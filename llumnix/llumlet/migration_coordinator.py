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
from typing import List, Dict
import asyncio

import ray.actor
import ray.exceptions

from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import LlumnixRequest, RequestStatus
from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.utils import asyncio_wait_for_with_timeout, RequestIDType
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


class MigrationCoordinator:
    def __init__(self,
                 instance_id: str,
                 backend_engine: BackendInterface,
                 backend_type: BackendType,
                 request_migration_policy: str,
                 migration_last_stage_max_blocks: int,
                 migration_max_stages: int) -> None:
        self.instance_id = instance_id
        self.backend_engine = backend_engine
        self.backend_type = backend_type
        self.migration_scheduler = LocalMigrationScheduler(request_migration_policy, self.backend_engine)
        self.migration_last_stage_max_blocks = migration_last_stage_max_blocks
        self.migration_max_stages = migration_max_stages
        self.pending_migrate_in_requests: Dict[str, float] = {}
        asyncio.create_task(self._watch_pending_migrate_in_requests_loop())

    async def migrate_out(self, dst_instance_actor: ray.actor.ActorHandle, dst_instance_id: str) -> List[RequestIDType]:
        # TODO(Failover): Currently, llumnix directly return if meeting exception during migration,
        # and handle migration exception through manager. In future, this should be handled by instance.
        try:
            migrate_out_requests = self.migration_scheduler.get_migrate_out_requests()

            if len(migrate_out_requests) == 0:
                return []

            for migrate_out_request in migrate_out_requests:
                migrate_out_request.is_migrating = True

            migrated_request_list = []
            for migrate_out_request in migrate_out_requests:
                migrated_request = await self._migrate_out_one_request(dst_instance_actor, dst_instance_id, migrate_out_request)
                migrated_request_list.extend(migrated_request)
                if len(migrated_request) == 0 and migrate_out_request.eom:
                    break
        # pylint: disable=W0703
        except Exception as e:
            # Not raise exception to ensure src instance won't die due to the death of dst instance.
            if isinstance(e, ray.exceptions.RayActorError):
                logger.info("Instance {} is dead.".format(dst_instance_id))
            elif isinstance(e, asyncio.TimeoutError, ray.exceptions.GetTimeoutError):
                logger.error("Instance {} is hang, please check the cause.".format(dst_instance_id))
            else:
                logger.exception("Failed to migrate out, unexpected exception: {}".format(e))
                raise

        return migrated_request_list

    async def _migrate_out_one_request(self,
                                       dst_instance_actor: ray.actor.ActorHandle,
                                       dst_instance_id: str,
                                       migrate_out_request: LlumnixRequest) -> List[LlumnixRequest]:
        t0 = time.time()
        logger.info("{}->{} begin migrate out".format(self.instance_id, dst_instance_id))
        migrated_request = []

        if migrate_out_request.status == RequestStatus.RUNNING:
            migrate_out_request.migration_start_time = time.time()
            status = await self._migrate_out_running_request(dst_instance_actor, dst_instance_id, migrate_out_request)
        elif migrate_out_request.status == RequestStatus.WAITING:
            migrate_out_request.migration_start_time = time.time()
            status = await self._migrate_out_waiting_request(dst_instance_actor, dst_instance_id, migrate_out_request)
        else:
            return migrated_request

        if status == MigrationStatus.FINISHED:
            success = await self._dst_commit_dst_request(dst_instance_actor, dst_instance_id, migrate_out_request)
            if success:
                self.backend_engine.free_src_request(migrate_out_request)
                self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
                migrated_request.append(migrate_out_request.request_id)
            else:
                self.backend_engine.add_running_request(migrate_out_request)
                self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
                status = MigrationStatus.ABORTED_DST
        else: # ABORTED_SRC or ABORTED_DST
            migrate_out_request.reset_migration_states_src()
            migrate_out_request.reset_status()
            # If src instance aborts itself, dst instance should free the pre allocated cache in pre_alloc_cache.
            if status == MigrationStatus.ABORTED_SRC:
                await self._dst_free_pre_alloc_cache(dst_instance_actor, dst_instance_id, migrate_out_request.request_id)

        t1 = time.time()
        logger.info("Instance {}->{} migrate done, migrate request {}, migration status: {}, len: {} blocks, cost: {} ms" \
                    .format(self.instance_id, dst_instance_id, migrated_request, status, \
                            sum(migrate_out_request.stage_num_blocks_list), (t1 - t0)*1000))

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
        dst_blocks = await self._dst_pre_alloc_cache(
                        dst_instance_actor,
                        dst_instance_id,
                        migrate_out_request.request_id,
                        migrate_out_request.status,
                        migrate_out_request.request_arrival_time,
                        migrate_out_request.prefill_num_blocks,
                        migrate_out_request.token_ids,
                        is_first_stage=True,
                    )
        if len(dst_blocks) != migrate_out_request.prefill_num_blocks:
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
        # TODO(s5u13b): Make migration codes of vLLM and BladeLLM uniform (some functions are not async).
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
            dst_blocks = await self._dst_pre_alloc_cache(
                            dst_instance_actor,
                            dst_instance_id,
                            migrate_out_request.request_id,
                            migrate_out_request.status,
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
            dst_blocks = await self._dst_pre_alloc_cache(
                            dst_instance_actor,
                            dst_instance_id,
                            migrate_out_request.request_id,
                            migrate_out_request.status,
                            migrate_out_request.request_arrival_time,
                            stage_block_num,
                            incremental_token_ids,
                            is_first_stage,
                        )

        if len(dst_blocks) != len(src_blocks):
            # migrate in instance failed to pre alloc
            if is_last_stage:
                self.backend_engine.add_running_request(migrate_out_request)
                self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
            return MigrationStatus.ABORTED_DST

        if migrate_out_request.should_abort_migration():
            return MigrationStatus.ABORTED_SRC

        # do stage send/recv
        migrate_out_request.stage_timestamps.append(time.time())
        migrate_out_request.stage_num_blocks_list.append(stage_block_num)
        # TODO(ZeldaHuang): send_cache in pre_alloc_cache/migrate_in_last_stage
        success = await self._send_cache(
            dst_instance_actor, dst_instance_id, src_blocks, dst_blocks, migrate_out_request.request_id, is_last_stage
        )
        if not success:
            if is_last_stage:
                self.backend_engine.add_running_request(migrate_out_request)
                self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
            return MigrationStatus.ABORTED_DST

        if not is_last_stage and migrate_out_request.should_abort_migration():
            # migrate-out request abort by scheduler during send/recv
            return MigrationStatus.ABORTED_SRC

        return migration_status

    def pre_alloc_cache(self,
                        request_id: RequestIDType,
                        request_status: RequestStatus,
                        request_arrival_time: float,
                        block_num: int,
                        token_ids: List[int],
                        is_first_stage: bool) -> List[int]:
        if not is_first_stage:
            if request_id not in self.pending_migrate_in_requests:
                return []
            del self.pending_migrate_in_requests[request_id]

        pre_alloc_blocks = self.backend_engine.pre_alloc_cache(request_id,
                                                               request_status,
                                                               request_arrival_time,
                                                               block_num,
                                                               token_ids)
        if len(pre_alloc_blocks) != block_num:
            # failed to alloc, abort request
            self.free_pre_alloc_cache(request_id)
            return pre_alloc_blocks

        self.pending_migrate_in_requests[request_id] = time.time()

        return pre_alloc_blocks

    async def _dst_pre_alloc_cache(self,
                                   dst_instance_actor: ray.actor.ActorHandle,
                                   dst_instance_id: str,
                                   request_id: RequestIDType,
                                   request_status: RequestStatus,
                                   request_arrival_time: float,
                                   block_num: int,
                                   token_ids: List[int],
                                   is_first_stage: bool) -> List[int]:
        try:
            return await asyncio_wait_for_with_timeout(
                dst_instance_actor.execute_migration_method.remote(
                    "pre_alloc_cache",
                    request_id,
                    request_status,
                    request_arrival_time,
                    block_num,
                    token_ids,
                    is_first_stage,
                )
            )
        # pylint: disable=W0703
        except Exception as e:
            if isinstance(e, ray.exceptions.RayActorError):
                logger.info("Instance {} is dead.".format(dst_instance_id))
            elif isinstance(e, asyncio.TimeoutError):
                logger.error("Instance {} is hang, please check the cause.".format(dst_instance_id))
            else:
                logger.exception("Failed to call dst instance {} to pre-allocate cache for request {}, "
                    "unexpected exception: {}".format(dst_instance_id, request_id, e))
            # Once return empty list, the migration status will become ABORTED_DST,
            # which will stop and clear the migration.
            return []

    def free_pre_alloc_cache(self, request_id: RequestIDType) -> None:
        return self.backend_engine.free_pre_alloc_cache(request_id)

    async def _dst_free_pre_alloc_cache(self,
                                        dst_instance_actor: ray.actor.ActorHandle,
                                        dst_instance_id: str,
                                        request_id: RequestIDType) -> bool:
        try:
            await asyncio_wait_for_with_timeout(
                dst_instance_actor.execute_migration_method.remote("free_pre_alloc_cache", request_id)
            )
            return True
        # pylint: disable=W0703
        except Exception as e:
            if isinstance(e, ray.exceptions.RayActorError):
                logger.info("Instance {} is dead.".format(dst_instance_id))
            elif isinstance(e, asyncio.TimeoutError):
                logger.error("Instance {} is hang, please check the cause.".format(dst_instance_id))
            else:
                logger.exception("Failed to call dst instance {} to free pre-allocate cache for request {}, "
                    "unexpected exception: {}".format(dst_instance_id, request_id, e))
            return False

    async def _send_cache(self,
                          dst_instance_actor: ray.actor.ActorHandle,
                          dst_instance_id: str,
                          src_blocks: List[int],
                          dst_blocks: List[int],
                          request_id: RequestIDType,
                          is_last_stage: bool) -> bool:
        try:
            return await self.backend_engine.send_cache(
                dst_instance_actor, src_blocks, dst_blocks, request_id, is_last_stage
            )
        # pylint: disable=W0703
        except Exception as e:
            if isinstance(e, ray.exceptions.RayActorError):
                logger.info("Instance {} is dead.".format(dst_instance_id))
            elif isinstance(e, asyncio.TimeoutError):
                logger.error("Instance {} is hang, please check the cause.".format(dst_instance_id))
            else:
                logger.exception("Failed to send cache to instance {} for request {}, "
                    "unexpected exception: {}".format(dst_instance_id, request_id, e))
            return False

    async def recv_cache(self, request_id: RequestIDType, *args, **kwargs) -> bool:
        if request_id not in self.pending_migrate_in_requests:
            return False
        del self.pending_migrate_in_requests[request_id]
        # pylint: disable=protected-access
        result = await self.backend_engine.recv_cache(request_id, *args, **kwargs)
        if result is True:
            self.pending_migrate_in_requests[request_id] = time.time()
        return result

    async def commit_dst_request(self, backend_request: LlumnixRequest) -> bool:
        if backend_request.request_id not in self.pending_migrate_in_requests:
            return False
        del self.pending_migrate_in_requests[backend_request.request_id]
        return await self.backend_engine.commit_dst_request(backend_request)

    async def _dst_commit_dst_request(self,
                                      dst_instance_actor: ray.actor.ActorHandle,
                                      dst_instance_id: str,
                                      migrate_out_request: LlumnixRequest) -> bool:
        try:
            return await asyncio_wait_for_with_timeout(
                dst_instance_actor.execute_migration_method_async.remote("commit_dst_request", migrate_out_request)
            )
        # pylint: disable=W0703
        except Exception as e:
            if isinstance(e, ray.exceptions.RayActorError):
                logger.info("Instance {} is dead.".format(dst_instance_id))
            elif isinstance(e, asyncio.TimeoutError):
                logger.error("Instance {} is hang, please check the cause.".format(dst_instance_id))
            else:
                logger.exception("Failed to call dst instance {} to commit dst request {}, "
                    "unexpected exception: {}".format(dst_instance_id, migrate_out_request.request_id, e))
            return False

    async def _watch_pending_migrate_in_requests_loop(self):
        while True:
            await asyncio.sleep(PENDING_MIGRATE_IN_TIMEOUT / 2)
            curr_time = time.time()
            new_pending_migrate_in_requests: Dict[str, float] = {}
            for request_id, last_migrate_in_stop_time in self.pending_migrate_in_requests.items():
                if curr_time - last_migrate_in_stop_time > PENDING_MIGRATE_IN_TIMEOUT:
                    logger.error("Pending migrate in request {} timeout after {} seconds.".format(request_id, PENDING_MIGRATE_IN_TIMEOUT))
                    self.backend_engine.free_pre_alloc_cache(request_id)
                else:
                    new_pending_migrate_in_requests[request_id] = last_migrate_in_stop_time
            self.pending_migrate_in_requests = new_pending_migrate_in_requests
