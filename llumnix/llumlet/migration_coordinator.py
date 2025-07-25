# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import enum
from typing import List, Dict, Callable, Optional
import asyncio
import functools
import inspect

import ray.actor

from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import LlumnixRequest, RequestStatus
from llumnix.backends.backend_interface import BackendBaseInterface
from llumnix.utils import (
    asyncio_wait_for_ray_remote_call_with_timeout,
    RequestIDType,
    MigrationResponse,
    log_instance_exception,
    BackendType,
    MigrationType
)
from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler
from llumnix.constants import PENDING_MIGRATE_IN_TIMEOUT

logger = init_logger(__name__)


class MigrationStatus(enum.Enum):
    """Status of Migration."""
    RUNNING = enum.auto()
    ABORTED_DST = enum.auto()
    ABORTED_SRC = enum.auto()
    FINISHED = enum.auto()

    @staticmethod
    def is_finished(status: "MigrationStatus") -> bool:
        return status in [
            MigrationStatus.ABORTED_DST,
            MigrationStatus.ABORTED_SRC,
            MigrationStatus.FINISHED
        ]


def update_pending_migrate_in_request_decorator(func: Callable):
    # pylint: disable=trailing-whitespace
    """
    This decorator is used by migrate in related functions. This decorator watch migrate in request, 
    once one migrate in function (pre_alloc_cache, recv_cache, commit_dst_request) for a migrate in request is executed successfully, 
    it means that this migrate in request is waiting for next migrate in function to be called. 
    And this decorator records the time for the migrate in request when the migrate in function is executed successfully, 
    and remove the recorded migrate in request when next migrate in function for the migrate in request is called. 
    In the migration coordinator, there is a backgroud loop continuously watching the recorded migrate in request, 
    if the recorded migrate in request is not removed after PENDING_MIGRATE_IN_TIMEOUT seconds, 
    it indicates that the migrate out instance is dead, and next migrate in function will not be called. 
    And therefore, the backgroud loop remove the recorded migrate in request and clear the migration states of the migrate in request.
    """

    assert func.__name__ in ["_pre_alloc_cache", "recv_cache", "commit_dst_request", "free_pre_alloc_cache"]

    def inspect_is_start(func: Callable, self, *args, **kwargs) -> bool:
        if func.__name__ != "_pre_alloc_cache":
            return False
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        is_first_stage = bound_args.arguments.get("is_first_stage")
        return is_first_stage

    def inspect_is_stop(func: Callable) -> bool:
        return func.__name__ in ["commit_dst_request", "free_pre_alloc_cache"]

    def inspect_request_id(func: Callable, self, *args, **kwargs) -> RequestIDType:
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        request_id = bound_args.arguments.get("request_id")
        return request_id

    def pre_process_pending_migrate_in_request_time(
        is_start: bool,
        request_id: RequestIDType,
        pending_migrate_in_request_time: Dict[RequestIDType, float]
    ) -> bool:
        if not is_start:
            if request_id not in pending_migrate_in_request_time:
                return False
            del pending_migrate_in_request_time[request_id]
        return True

    def post_process_pending_migrate_in_request_time(
        response: MigrationResponse,
        is_stop: bool,
        request_id: RequestIDType,
        pending_migrate_in_request_time: Dict[RequestIDType, float]
    ) -> None:
        if response.success and not is_stop:
            pending_migrate_in_request_time[request_id] = time.time()

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        is_start = inspect_is_start(func, self, *args, **kwargs)
        is_stop = inspect_is_stop(func)
        request_id = inspect_request_id(func, self, *args, **kwargs)
        if not pre_process_pending_migrate_in_request_time(
            is_start, request_id, self.pending_migrate_in_request_time
        ):
            return MigrationResponse(success=False, return_value=None)
        response = await func(self, *args, **kwargs)
        post_process_pending_migrate_in_request_time(
            response, is_stop, request_id, self.pending_migrate_in_request_time
        )
        return response

    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        is_start = inspect_is_start(func, self, *args, **kwargs)
        is_stop = inspect_is_stop(func)
        request_id = inspect_request_id(func, self, *args, **kwargs)
        if not pre_process_pending_migrate_in_request_time(
            is_start, request_id, self.pending_migrate_in_request_time
        ):
            return MigrationResponse(success=False, return_value=None)
        response = func(self, *args, **kwargs)
        post_process_pending_migrate_in_request_time(
            response, is_stop, request_id, self.pending_migrate_in_request_time
        )
        return response

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def update_migrating_out_request_id_set_decorator(func: Callable):
    # pylint: disable=trailing-whitespace
    """
    This decorator is used to update the migrating out request id set, 
    which is used to decide whether to accept new migrate out/in requests.
    """

    assert func.__name__ == "_migrate_out_one_request"

    def inspect_request_id(func: Callable, self, *args, **kwargs) -> RequestIDType:
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        migrate_out_request = bound_args.arguments.get("migrate_out_request")
        return migrate_out_request.request_id

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        request_id = inspect_request_id(func, self, *args, **kwargs)
        if request_id not in self.migrating_out_request_id_set:
            self.migrating_out_request_id_set.add(request_id)
        try:
            return await func(self, *args, **kwargs)
        finally:
            assert request_id in self.migrating_out_request_id_set, \
                "request_id is added to migrating_out_request_id_set for each migration"
            self.migrating_out_request_id_set.remove(request_id)

    return async_wrapper


def update_migrating_in_request_id_set_decorator(func):
    # pylint: disable=trailing-whitespace
    """
    This decorator is used to update the migrating in request id set, 
    which is used to decide whether to accept new migrate out/in requests.
    """

    assert func.__name__ in ["_pre_alloc_cache", "recv_cache", "commit_dst_request", "free_pre_alloc_cache"]

    def inspect_request_id(func: Callable, self, *args, **kwargs) -> RequestIDType:
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        request_id = bound_args.arguments.get("request_id")
        return request_id

    def add_migrating_in_request_id_set(self, request_id: RequestIDType):
        if request_id not in self.migrating_in_request_id_set:
            self.migrating_in_request_id_set.add(request_id)

    def remove_migrating_in_request_id_set(self, request_id: RequestIDType):
        if request_id not in self.pending_migrate_in_request_time:
            assert request_id in self.migrating_in_request_id_set, \
                "request_id is added to migrating_in_request_id_set for each migration"
            self.migrating_in_request_id_set.remove(request_id)

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        request_id = inspect_request_id(func, self, *args, **kwargs)
        add_migrating_in_request_id_set(self, request_id)
        try:
            return await func(self, *args, **kwargs)
        finally:
            remove_migrating_in_request_id_set(self, request_id)

    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        request_id = inspect_request_id(func, self, *args, **kwargs)
        add_migrating_in_request_id_set(self, request_id)
        try:
            return func(self, *args, **kwargs)
        finally:
            remove_migrating_in_request_id_set(self, request_id)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class MigrationCoordinator:
    def __init__(self,
                 instance_id: str,
                 backend_engine: BackendBaseInterface,
                 backend_type: BackendType,
                 max_migration_concurrency: int,
                 request_migration_policy: str,
                 migration_last_stage_max_blocks: int,
                 migration_max_stages: int) -> None:
        self.instance_id = instance_id
        self.backend_engine = backend_engine
        self.backend_type = backend_type
        self.max_migration_concurrency = max_migration_concurrency
        self.migration_scheduler = LocalMigrationScheduler(request_migration_policy, self.backend_engine)
        self.migration_last_stage_max_blocks = migration_last_stage_max_blocks
        self.migration_max_stages = migration_max_stages
        self.pending_migrate_in_request_time: Dict[str, float] = {}
        self.migrating_out_request_id_set = set()
        self.migrating_in_request_id_set = set()
        asyncio.create_task(self._watch_pending_migrate_in_requests_loop())

    async def migrate_out(self,
                          dst_instance_actor: ray.actor.ActorHandle,
                          dst_instance_id: str,
                          migration_type: Optional[MigrationType] = None) -> List[RequestIDType]:
        if not self.has_migration_slot():
            logger.debug(
                "Max migration concurrency ({}) reached, reject new migrate out request attempt.".format(
                    self.max_migration_concurrency
                )
            )
            return []

        migrate_out_requests = self.migration_scheduler.get_migrate_out_requests(migration_type)

        if len(migrate_out_requests) == 0:
            return []

        def migration_request_callback(dst_instance_id, migrate_out_request, fut):
            ret = fut.result()[0]
            migrated_request_list.extend(migrated_request)
            if isinstance(ret, Exception):
                log_instance_exception(ret, dst_instance_id, "migrate_out", migrate_out_request.request_id)

        migrated_request_list = []
        for migrate_out_request in migrate_out_requests:
            migration_tasks = []
            while self.has_migration_slot():
                self.migrating_out_request_id_set.add(migrate_out_request.request_id)
                migrate_out_request.is_migrating = True
                migration_task = asyncio.gather(self._migrate_out_one_request(dst_instance_actor, dst_instance_id,
                    migrate_out_request, migration_type), return_exceptions=True)
                migration_task.add_done_callback(
                    partial(migration_request_callback, dst_instance_id, migrate_out_request))
                migration_tasks.append(migration_task)
            await asyncio.gather(*migration_tasks, return_exceptions=True)

        return migrated_request_list

    @update_migrating_out_request_id_set_decorator
    async def _migrate_out_one_request(self,
                                       dst_instance_actor: ray.actor.ActorHandle,
                                       dst_instance_id: str,
                                       migrate_out_request: LlumnixRequest,
                                       migration_type: Optional[MigrationType] = None) -> List[LlumnixRequest]:
        t0 = time.time()
        logger.info("{}->{} begin migrate out {}.".format(self.instance_id, dst_instance_id, migrate_out_request.request_id))
        migrated_request = []

        if migrate_out_request.llumnix_status == RequestStatus.RUNNING:
            migrate_out_request.migration_start_time = time.time()
            status = await self._migrate_out_running_request(dst_instance_actor, dst_instance_id, migrate_out_request)
        elif migrate_out_request.llumnix_status == RequestStatus.WAITING:
            migrate_out_request.migration_start_time = time.time()
            status = await self._migrate_out_waiting_request(dst_instance_actor, dst_instance_id, migrate_out_request)
        else:
            return migrated_request

        if status == MigrationStatus.FINISHED:
            response = await self._dst_commit_dst_request(dst_instance_actor, dst_instance_id, migrate_out_request)
            if response.success:
                self.backend_engine.free_src_request(migrate_out_request)
                self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
                migrated_request.append(migrate_out_request.request_id)
            else:
                await self.backend_engine.add_running_request(migrate_out_request)
                self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
                status = MigrationStatus.ABORTED_DST
        else: # ABORTED_SRC or ABORTED_DST
            migrate_out_request.reset_migration_states_src()
            migrate_out_request.reset_llumnix_status()
            # If src instance aborts itself, dst instance should free the pre allocated cache in pre_alloc_cache.
            if status == MigrationStatus.ABORTED_SRC:
                await self._dst_free_pre_alloc_cache(dst_instance_actor, dst_instance_id, migrate_out_request.request_id)

        t1 = time.time()
        logger.info(
            "Instance {}->{} migrate done, migration type: {}, migrate request {}, "
            "migration status: {}, len: {} blocks, cost: {} ms".format(
                self.instance_id, dst_instance_id, migration_type, migrated_request, status,
                sum(migrate_out_request.stage_num_blocks_list), (t1 - t0)*1000
            )
        )

        return migrated_request

    async def _migrate_out_running_request(self,
                                           dst_instance_actor: ray.actor.ActorHandle,
                                           dst_instance_id: str,
                                           migrate_out_request: LlumnixRequest) -> MigrationStatus:
        return await self._migrate_out_multistage(dst_instance_actor, dst_instance_id, migrate_out_request)

    async def _migrate_out_waiting_request(self,
                                           dst_instance_actor: ray.actor.ActorHandle,
                                           dst_instance_id: str,
                                           migrate_out_request: LlumnixRequest) -> MigrationStatus:
        found = self.backend_engine.remove_waiting_request(migrate_out_request.request_id)
        if not found:
            return MigrationStatus.ABORTED_SRC
        self.backend_engine.add_migrating_out_request_last_stage(migrate_out_request)
        response = await self._dst_pre_alloc_cache(
                        dst_instance_actor,
                        dst_instance_id,
                        migrate_out_request.request_id,
                        migrate_out_request.llumnix_status,
                        migrate_out_request.request_arrival_time,
                        migrate_out_request.prefill_num_blocks,
                        migrate_out_request.token_ids,
                        is_first_stage=True,
                    )
        if not response.success:
            self.backend_engine.add_waiting_request(migrate_out_request)
            self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
            return MigrationStatus.ABORTED_DST

        return MigrationStatus.FINISHED

    async def _migrate_out_multistage(self,
                                      dst_instance_actor: ray.actor.ActorHandle,
                                      dst_instance_id: str,
                                      migrate_out_request: LlumnixRequest) -> MigrationStatus:
        stage_count = 0
        while stage_count < self.migration_max_stages:
            stage_count += 1
            status = await self._migrate_out_onestage(
                        dst_instance_actor,
                        dst_instance_id,
                        migrate_out_request,
                        is_first_stage=(stage_count == 1),
                    )
            if MigrationStatus.is_finished(status):
                return status
        # exceed max stages
        return MigrationStatus.ABORTED_SRC

    async def _migrate_out_onestage(self,
                                    dst_instance_actor: ray.actor.ActorHandle,
                                    dst_instance_id: str,
                                    migrate_out_request: LlumnixRequest,
                                    is_first_stage: bool) -> MigrationStatus:
        if migrate_out_request.should_abort_migration():
            return MigrationStatus.ABORTED_SRC

        pre_stage_num_blocks = sum(migrate_out_request.stage_num_blocks_list)
        incremental_blocks, incremental_token_ids, is_last_stage = \
            await self.backend_engine.get_request_incremental_blocks(migrate_out_request, pre_stage_num_blocks)

        if migrate_out_request.should_abort_migration():
            return MigrationStatus.ABORTED_SRC

        # live migration, transfer all blocks except last one(currently updating)
        if not is_last_stage:
            migration_status = MigrationStatus.RUNNING
            src_blocks = incremental_blocks[:-1]
            if len(incremental_token_ids) > 0:
                incremental_token_ids = incremental_token_ids[:len(src_blocks)*migrate_out_request.block_size]
            stage_block_num = len(incremental_blocks) - 1
            response = await self._dst_pre_alloc_cache(
                            dst_instance_actor,
                            dst_instance_id,
                            migrate_out_request.request_id,
                            migrate_out_request.llumnix_status,
                            migrate_out_request.request_arrival_time,
                            stage_block_num,
                            incremental_token_ids,
                            is_first_stage,
                        )
        else:
            # last stage migration, stop inference, transfer all blocks
            migration_status = MigrationStatus.FINISHED
            found = await self.backend_engine.remove_running_request(migrate_out_request.request_id)
            # Request coule be finished by previous or current step.
            if not found or migrate_out_request.finished:
                return MigrationStatus.ABORTED_SRC
            self.backend_engine.add_migrating_out_request_last_stage(migrate_out_request)
            src_blocks = incremental_blocks[:]
            stage_block_num = len(incremental_blocks)
            response = await self._dst_pre_alloc_cache(
                            dst_instance_actor,
                            dst_instance_id,
                            migrate_out_request.request_id,
                            migrate_out_request.llumnix_status,
                            migrate_out_request.request_arrival_time,
                            stage_block_num,
                            incremental_token_ids,
                            is_first_stage,
                        )

        if not response.success:
            # migrate in instance failed to pre alloc
            if is_last_stage:
                await self.backend_engine.add_running_request(migrate_out_request)
                self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
            return MigrationStatus.ABORTED_DST

        if migrate_out_request.should_abort_migration():
            return MigrationStatus.ABORTED_SRC

        # do stage send/recv
        dst_blocks = response.return_value
        migrate_out_request.stage_timestamps.append(time.time())
        migrate_out_request.stage_num_blocks_list.append(stage_block_num)
        # TODO(ZeldaHuang): send_cache in pre_alloc_cache/migrate_in_last_stage
        response = await self._send_cache(
            dst_instance_actor, dst_instance_id, src_blocks, dst_blocks, migrate_out_request.request_id, is_last_stage
        )
        if not response.success:
            if is_last_stage:
                await self.backend_engine.add_running_request(migrate_out_request)
                self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
            return MigrationStatus.ABORTED_DST

        if not is_last_stage and migrate_out_request.should_abort_migration():
            # migrate-out request abort by scheduler during send/recv
            return MigrationStatus.ABORTED_SRC

        return migration_status

    # Add this function to implement migration lock outside the update_migrating_in_request_id_set_decorator.
    def pre_alloc_cache(self,
                        request_id: RequestIDType,
                        request_status: RequestStatus,
                        request_arrival_time: float,
                        block_num: int,
                        token_ids: List[int],
                        is_first_stage: bool) -> MigrationResponse:
        if is_first_stage and not self.has_migration_slot():
            logger.debug(
                "Max migration concurrency ({}) reached, reject new migrate in attempt.".format(
                    self.max_migration_concurrency
                )
            )
            return MigrationResponse(success=False, return_value=None)
        return self._pre_alloc_cache(
            request_id, request_status, request_arrival_time, block_num, token_ids, is_first_stage
        )

    # pylint: disable=unused-argument
    @update_migrating_in_request_id_set_decorator
    @update_pending_migrate_in_request_decorator
    def _pre_alloc_cache(self,
                         request_id: RequestIDType,
                         request_status: RequestStatus,
                         request_arrival_time: float,
                         block_num: int,
                         token_ids: List[int],
                         is_first_stage: bool) -> MigrationResponse:
        response = self.backend_engine.pre_alloc_cache(request_id,
                                                       request_status,
                                                       request_arrival_time,
                                                       block_num,
                                                       token_ids)
        if not response.success:
            # failed to alloc, abort request
            self.free_pre_alloc_cache(request_id)

        return response

    async def _dst_pre_alloc_cache(self,
                                   dst_instance_actor: ray.actor.ActorHandle,
                                   dst_instance_id: str,
                                   request_id: RequestIDType,
                                   request_status: RequestStatus,
                                   request_arrival_time: float,
                                   block_num: int,
                                   token_ids: List[int],
                                   is_first_stage: bool) -> MigrationResponse:
        try:
            return await asyncio_wait_for_ray_remote_call_with_timeout(
                dst_instance_actor.execute_migration_method,
                "pre_alloc_cache",
                request_id,
                request_status,
                request_arrival_time,
                block_num,
                token_ids,
                is_first_stage
            )
        # pylint: disable=W0703
        except Exception as e:
            log_instance_exception(e, dst_instance_id, "_dst_pre_alloc_cache", request_id)
            # Once return False response, the migration status will become ABORTED_DST,
            # which will stop and clear the migration.
            return MigrationResponse(success=False, return_value=None)

    @update_migrating_in_request_id_set_decorator
    @update_pending_migrate_in_request_decorator
    def free_pre_alloc_cache(self, request_id: RequestIDType) -> MigrationResponse:
        try:
            self.backend_engine.free_pre_alloc_cache(request_id)
            return MigrationResponse(success=True, return_value=None)
        # pylint: disable=W0703
        except Exception as e:
            log_instance_exception(e, self.instance_id, "free_pre_alloc_cache", request_id)
            return MigrationResponse(success=False, return_value=None)

    async def _dst_free_pre_alloc_cache(self,
                                        dst_instance_actor: ray.actor.ActorHandle,
                                        dst_instance_id: str,
                                        request_id: RequestIDType) -> bool:
        try:
            await asyncio_wait_for_ray_remote_call_with_timeout(
                dst_instance_actor.execute_migration_method, "free_pre_alloc_cache", request_id
            )
            return True
        # pylint: disable=W0703
        except Exception as e:
            log_instance_exception(e, dst_instance_id, "_dst_free_pre_alloc_cache", request_id)
            return False

    async def _send_cache(self,
                          dst_instance_actor: ray.actor.ActorHandle,
                          dst_instance_id: str,
                          src_blocks: List[int],
                          dst_blocks: List[int],
                          request_id: RequestIDType,
                          is_last_stage: bool) -> MigrationResponse:
        try:
            return await self.backend_engine.send_cache(
                dst_instance_actor, src_blocks, dst_blocks, request_id, is_last_stage
            )
        # pylint: disable=W0703
        except Exception as e:
            log_instance_exception(e, dst_instance_id, "_send_cache", request_id)
            return MigrationResponse(success=False, return_value=None)

    @update_migrating_in_request_id_set_decorator
    @update_pending_migrate_in_request_decorator
    async def recv_cache(self, request_id: RequestIDType, *args, **kwargs) -> MigrationResponse:
        # pylint: disable=protected-access
        return await self.backend_engine.recv_cache(request_id, *args, **kwargs)

    @update_migrating_in_request_id_set_decorator
    @update_pending_migrate_in_request_decorator
    async def commit_dst_request(self, request_id: RequestIDType, backend_request: LlumnixRequest) -> MigrationResponse:
        return await self.backend_engine.commit_dst_request(request_id, backend_request)

    async def _dst_commit_dst_request(self,
                                      dst_instance_actor: ray.actor.ActorHandle,
                                      dst_instance_id: str,
                                      migrate_out_request: LlumnixRequest) -> MigrationResponse:
        try:
            return await asyncio_wait_for_ray_remote_call_with_timeout(
                dst_instance_actor.execute_migration_method_async,
                "commit_dst_request", migrate_out_request.request_id, migrate_out_request
            )
        # pylint: disable=W0703
        except Exception as e:
            log_instance_exception(e, dst_instance_id, "_dst_commit_dst_request", migrate_out_request.request_id)
            return MigrationResponse(success=False, return_value=None)

    async def _watch_pending_migrate_in_requests_loop(self):
        while True:
            await asyncio.sleep(PENDING_MIGRATE_IN_TIMEOUT / 2)
            curr_time = time.time()
            new_pending_migrate_in_requests: Dict[str, float] = {}
            for request_id, last_migrate_in_stop_time in self.pending_migrate_in_request_time.items():
                if curr_time - last_migrate_in_stop_time > PENDING_MIGRATE_IN_TIMEOUT:
                    logger.error("Pending migrate in request {} timeout after {} seconds.".format(request_id, PENDING_MIGRATE_IN_TIMEOUT))
                    self.backend_engine.free_pre_alloc_cache(request_id)
                else:
                    new_pending_migrate_in_requests[request_id] = last_migrate_in_stop_time
            self.pending_migrate_in_request_time = new_pending_migrate_in_requests

    def has_migration_slot(self) -> bool:
        return len(self.migrating_in_request_id_set) + len(self.migrating_out_request_id_set) < self.max_migration_concurrency

    def is_migrating(self) -> bool:
        return len(self.migrating_in_request_id_set) + len(self.migrating_out_request_id_set) > 0
