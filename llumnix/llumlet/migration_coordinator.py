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
from typing import List
import asyncio

import ray.actor
import ray.exceptions

from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import LlumnixRequest, RequestStatus
from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.utils import asyncio_wait_for_with_timeout, RequestIDType
from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler

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

    async def migrate_out(self, dst_instance_id: str, dst_instance_actor_handle: ray.actor.ActorHandle) -> List[RequestIDType]:
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
                migrated_request = await self._migrate_out_one_request(migrate_out_request, dst_instance_id, dst_instance_actor_handle)
                migrated_request_list.extend(migrated_request)
                if len(migrated_request) == 0 and migrate_out_request.eom:
                    break
        except ray.exceptions.RayActorError:
            # Not raise exception to ensure src instance won't die due to the death of dst instance.
            logger.info("Instance {} is dead.".format(dst_instance_id))
        except (asyncio.TimeoutError, ray.exceptions.GetTimeoutError):
            logger.error("Instance {} is hang, please check the cause.".format(dst_instance_id))
        # pylint: disable=W0703
        except Exception as e:
            logger.exception("Failed to migrate out, unexpected exception: {}".format(e))
            raise

        return migrated_request_list

    async def _migrate_out_one_request(self,
                                       migrate_out_request: LlumnixRequest,
                                       dst_instance_id: str,
                                       dst_instance_actor_handle: ray.actor.ActorHandle) -> List[LlumnixRequest]:
        t0 = time.time()
        logger.info("{}->{} begin migrate out".format(self.instance_id, dst_instance_id))
        migrated_request = []

        if migrate_out_request.status == RequestStatus.RUNNING:
            migrate_out_request.migration_start_time = time.time()
            status = await self._migrate_out_running_request(dst_instance_actor_handle, migrate_out_request)
        elif migrate_out_request.status == RequestStatus.WAITING:
            migrate_out_request.migration_start_time = time.time()
            status = await self._migrate_out_waiting_request(dst_instance_actor_handle, migrate_out_request)
        else:
            return migrated_request

        if status == MigrationStatus.FINISHED:
            await asyncio_wait_for_with_timeout(
                dst_instance_actor_handle.commit_dst_request.remote(migrate_out_request)
            )
            self.backend_engine.free_src_request(migrate_out_request)
            self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
            migrated_request.append(migrate_out_request.request_id)
        else: # ABORTED_SRC or ABORTED_DST
            migrate_out_request.reset_migration_args_src()
            migrate_out_request.reset_status()
            # If src aborts itself, dst should free the pre allocated cache in pre_alloc_cache.
            if status == MigrationStatus.ABORTED_SRC:
                await asyncio_wait_for_with_timeout(
                    dst_instance_actor_handle.free_pre_alloc_cache.remote(migrate_out_request.request_id)
                )

        t1 = time.time()
        logger.info("Instance {}->{} migrate done, migrate request {}, migration status: {}, len: {} blocks, cost: {} ms" \
                    .format(self.instance_id, dst_instance_id, migrated_request, status, \
                            sum(migrate_out_request.stage_num_blocks_list), (t1 - t0)*1000))

        return migrated_request

    async def _migrate_out_running_request(self,
                                           migrate_in_actor: ray.actor.ActorHandle,
                                           migrate_out_request: LlumnixRequest) -> MigrationStatus:
        return await self._migrate_out_multistage(migrate_in_actor, migrate_out_request)

    async def _migrate_out_waiting_request(self,
                                           migrate_in_actor: ray.actor.ActorHandle,
                                           migrate_out_request: LlumnixRequest) -> MigrationStatus:
        """
        one-stage migration for a waiting request
        """
        found = self.backend_engine.remove_waiting_request(migrate_out_request.request_id)
        if not found:
            return MigrationStatus.ABORTED_SRC
        self.backend_engine.add_migrating_out_request_last_stage(migrate_out_request)
        dst_blocks = await asyncio_wait_for_with_timeout(
            migrate_in_actor.pre_alloc_cache.remote(
                migrate_out_request.request_id,
                migrate_out_request.status,
                migrate_out_request.request_arrival_time,
                migrate_out_request.prefill_num_blocks,
                migrate_out_request.token_ids
            )
        )
        if len(dst_blocks) != migrate_out_request.prefill_num_blocks:
            self.backend_engine.add_waiting_request(migrate_out_request)
            self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
            return MigrationStatus.ABORTED_DST

        return MigrationStatus.FINISHED

    async def _migrate_out_multistage(self,
                                      migrate_in_actor: ray.actor.ActorHandle,
                                      migrate_out_request: LlumnixRequest) -> MigrationStatus:
        """
        Migrate out requests to a specified instance, return migrated request id.

        Args:
            migrate_in_actor: instance actor name, used to get ray actor handle.
            migrate_out_request: request to migrate out.
        """
        stage_count = 0
        while stage_count < self.migration_max_stages:
            stage_count += 1
            status = await self._migrate_out_onestage(migrate_in_actor, migrate_out_request)
            if MigrationStatus.is_finished(status):
                return status
        # exceed max stages
        return MigrationStatus.ABORTED_SRC

    async def _migrate_out_onestage(self,
                                    migrate_in_actor: ray.actor.ActorHandle,
                                    migrate_out_request: LlumnixRequest) -> MigrationStatus:
        """
        one-stage live migration until last stage for a running request
        """
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
            dst_blocks = await asyncio_wait_for_with_timeout(
                migrate_in_actor.pre_alloc_cache.remote(
                    migrate_out_request.request_id,
                    migrate_out_request.status,
                    migrate_out_request.request_arrival_time,
                    stage_block_num,
                    incremental_token_ids
                )
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
            dst_blocks = await asyncio_wait_for_with_timeout(
                migrate_in_actor.pre_alloc_cache.remote(
                    migrate_out_request.request_id,
                    migrate_out_request.status,
                    migrate_out_request.request_arrival_time,
                    stage_block_num,
                    incremental_token_ids
                )
            )

        if len(dst_blocks) != len(src_blocks):
            # migrate-in instance failed to pre alloc
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
        await self.backend_engine.send_cache(migrate_in_actor, src_blocks, dst_blocks, migrate_out_request.request_id, is_last_stage)

        if not is_last_stage and migrate_out_request.should_abort_migration():
            # migrate-out request abort by scheduler during send/recv
            return MigrationStatus.ABORTED_SRC

        return migration_status

    def pre_alloc_cache(self,
                        request_id: RequestIDType,
                        request_status: RequestStatus,
                        request_arrival_time: float,
                        block_num: int,
                        token_ids: List[int]) -> List[int]:
        """
        Pre-allocate blocks for migrate in request.
        """
        pre_alloc_blocks = self.backend_engine.pre_alloc_cache(request_id,
                                                               request_status,
                                                               request_arrival_time,
                                                               block_num,
                                                               token_ids)
        if len(pre_alloc_blocks) != block_num:
            # failed to alloc, abort request
            self.free_pre_alloc_cache(request_id)
        return pre_alloc_blocks

    def free_pre_alloc_cache(self, request_id: RequestIDType = None) -> None:
        return self.backend_engine.free_pre_alloc_cache(request_id)

    async def recv_cache(self, *args, **kwargs) -> None:
        # pylint: disable=protected-access
        return await self.backend_engine._run_workers_async("recv_cache", *args, **kwargs)

    async def commit_dst_request(self, backend_request: LlumnixRequest) -> None:
        return await self.backend_engine.commit_dst_request(backend_request)

    async def clear_migration_states(self, is_migrate_in: bool) -> None:
        logger.info("Instance {} clear_migration_states, is_migrate_in: {}".format(self.instance_id, is_migrate_in))
        if is_migrate_in:
            # If migrate out instance dies during migration, migrate in instance directly free the pre-allocated cache of the migrating in request.
            logger.info("clear_migration_states: free_pre_alloc_cache")
            self.backend_engine.free_pre_alloc_cache()
        else:
            # If migrate in instance dies during migration, migrate out instance should add the migrating out request in last stage.
            # back to the running request queue.
            migrating_out_requests_last_stage = self.backend_engine.free_migrating_out_requests_last_stage()
            for backend_request in migrating_out_requests_last_stage:
                logger.info("clear_migration_states: add request {} back to engine".format(backend_request.request_id))
                assert RequestStatus.is_migrating(backend_request.status), \
                    "The status of request in migrating_out_requests_last_stage should be \
                     RequestStatus.WAITING_MIGRATING or RequestStatus.RUNNING_MIGRATING"
                if backend_request.status == RequestStatus.RUNNING_MIGRATING:
                    self.backend_engine.add_running_request(backend_request)
                else: # WAITING_MIGRATING
                    self.backend_engine.add_waiting_request(backend_request)
