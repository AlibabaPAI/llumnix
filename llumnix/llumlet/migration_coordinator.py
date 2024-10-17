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

# pylint: disable=unused-import
import ray

from llumnix.logger import init_logger
from llumnix.llumlet.request import LlumnixRequest, RequestStatus
from llumnix.backends.backend_interface import BackendInterface

logger = init_logger(__name__)

class MigrationStatus(enum.Enum):
    """Status of Migration."""
    RUNNING = enum.auto()
    FINISHED_DST_ABORTED = enum.auto()
    FINISHED_SRC_ABORTED = enum.auto()
    FINISHED_DONE = enum.auto()

    @staticmethod
    def is_finished(status: "MigrationStatus") -> bool:
        return status in [
            MigrationStatus.FINISHED_DST_ABORTED,
            MigrationStatus.FINISHED_SRC_ABORTED,
            MigrationStatus.FINISHED_DONE
        ]

class MigrationCoordinator:
    def __init__(self,
                 backend_engine: BackendInterface,
                 last_stage_max_blocks: int,
                 max_stages: int) -> None:
        self.last_stage_max_blocks = last_stage_max_blocks
        self.max_stages = max_stages
        self.backend_engine = backend_engine
    
    async def migrate_out_running_request(self,
                                          migrate_in_ray_actor: "ray.actor.ActorHandle",
                                          migrate_out_request: LlumnixRequest) -> "MigrationStatus":
        return await self._migrate_out_multistage(migrate_in_ray_actor, migrate_out_request)

    async def migrate_out_waiting_request(self,
                                          migrate_in_ray_actor: "ray.actor.ActorHandle",
                                          migrate_out_request: LlumnixRequest) -> "MigrationStatus":
        """one-stage migration for a waiting request
        """
        self.backend_engine.remove_waiting_request(migrate_out_request.request_id)
        self.backend_engine.add_migrating_out_request_last_stage(migrate_out_request)
        prefill_num_blocks = migrate_out_request.prefill_num_blocks
        dst_blocks = await migrate_in_ray_actor.execute_migration_method \
                                .remote("migrate_in_pre_alloc", migrate_out_request.request_id,
                                                                migrate_out_request.status,
                                                                migrate_out_request.arrival_time,
                                                                prefill_num_blocks)
        if len(dst_blocks) != prefill_num_blocks:
            self.backend_engine.add_waiting_request(migrate_out_request)
            self.backend_engine.remove_migrating_out_request_last_stage(migrate_out_request)
            return MigrationStatus.FINISHED_DST_ABORTED

        return MigrationStatus.FINISHED_DONE

    async def _migrate_out_multistage(self,
                                      migrate_in_ray_actor: "ray.actor.ActorHandle",
                                      migrate_out_request: LlumnixRequest) -> "MigrationStatus":
        """Migrate out requests to a specified instance, return migrated request id.
        Args:
            migrate_in_ray_actor: instance actor name, used to get ray actor handle
        """
        stage_count = 0
        while stage_count < self.max_stages:
            stage_count += 1
            status = await self._migrate_out_onestage(migrate_in_ray_actor, migrate_out_request)
            if MigrationStatus.is_finished(status):
                return status
        # exceed max stages
        return MigrationStatus.FINISHED_SRC_ABORTED

    async def _migrate_out_onestage(self,
                                    migrate_in_ray_actor: "ray.actor.ActorHandle",
                                    migrate_out_request: LlumnixRequest) -> "MigrationStatus":
        """one-stage live migration until last stage for a running request
        """
        pre_stage_num_blocks = sum(migrate_out_request.stage_num_blocks_list)
        incremental_blocks = self.backend_engine.get_request_incremental_blocks(migrate_out_request, pre_stage_num_blocks)
        # live migration, transfer all blocks except last one(currently updating)
        is_last_stage = (len(incremental_blocks) <= self.last_stage_max_blocks) or migrate_out_request.blocking_migration
        if not is_last_stage:
            migration_status = MigrationStatus.RUNNING
            src_blocks = incremental_blocks[:-1]
            stage_block_num = len(incremental_blocks) - 1
            dst_blocks = await migrate_in_ray_actor.execute_migration_method \
                                    .remote("migrate_in_pre_alloc", migrate_out_request.request_id,
                                                                    migrate_out_request.status,
                                                                    migrate_out_request.arrival_time,
                                                                    stage_block_num)
        else:
            # last stage migration, stop inference, transfer all blocks
            migration_status = MigrationStatus.FINISHED_DONE
            self.backend_engine.remove_running_request(migrate_out_request.request_id)
            self.backend_engine.add_migrating_out_request_last_stage(migrate_out_request)
            stage_block_num = len(incremental_blocks)
            src_blocks = incremental_blocks[:]
            dst_blocks = await migrate_in_ray_actor.execute_migration_method \
                                    .remote("migrate_in_pre_alloc", migrate_out_request.request_id, 
                                                                    migrate_out_request.status,
                                                                    migrate_out_request.arrival_time,
                                                                    stage_block_num)

        if len(dst_blocks) != len(src_blocks):
            # migrate-in instance failed to pre alloc
            if is_last_stage:
                self.backend_engine.add_running_request(migrate_out_request)
                self.backend_engine.remove_migrating_out_request_last_stage(migrate_out_request)
            return MigrationStatus.FINISHED_DST_ABORTED
        # do stage send/recv
        migrate_out_request.stage_timestamps.append(time.time())
        migrate_out_request.stage_num_blocks_list.append(stage_block_num)
        # TODO(ZeldaHuang): send_blocks in migrate_in_pre_alloc/migrate_in_last_stage
        await self.backend_engine.send_blocks(migrate_in_ray_actor, src_blocks, dst_blocks)
        if not is_last_stage and migrate_out_request.should_abort_migration():
            # migrate-out request abort by scheduler during send/recv
            return MigrationStatus.FINISHED_SRC_ABORTED

        return migration_status

    def migrate_in_pre_alloc(self,
                             request_id: str,
                             request_status: RequestStatus,
                             request_arrival_time: float,
                             block_num: int) -> List[int]:
        """prev alloc blocks to migrate in request
        """
        pre_alloc_blocks = self.backend_engine.pre_alloc(request_id,
                                                         request_status,
                                                         request_arrival_time,
                                                         block_num)
        if len(pre_alloc_blocks) != block_num:
            # failed to alloc, abort request
            self.free_dst_pre_alloc_cache(request_id)
        return pre_alloc_blocks

    def free_dst_pre_alloc_cache(self, request_id: str = None) -> None:
        self.backend_engine.free_dst_pre_alloc_cache(request_id)
