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
from llumnix.llumlet.request import LlumnixRequest
from llumnix.backends.backend_interface import BackendInterface

logger = init_logger(__name__)

class MigrationStatus(enum.Enum):
    """Status of Migration."""
    RUNNING = enum.auto()
    # aborted by src instance
    ABORTED_SRC = enum.auto()
    # aborted by dst instance
    ABORTED_DST = enum.auto()
    FINISHED_DONE = enum.auto()

    @staticmethod
    def is_finished(status: "MigrationStatus") -> bool:
        return status in [
            MigrationStatus.ABORTED_SRC,
            MigrationStatus.ABORTED_DST,
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

    async def migrate_out_onestage(self, migrate_in_ray_actor: "ray.actor.ActorHandle",  migrate_out_request: LlumnixRequest, ) -> "MigrationStatus":
        """one-stage live migration until last stage
        """
        pre_stage_num_blocks = sum(migrate_out_request.stage_num_blocks_list)
        incremental_blocks = self.backend_engine.get_request_incremental_blocks(migrate_out_request, pre_stage_num_blocks)
        # live migration, transfer all blocks except last one(currently updating)
        migration_status = MigrationStatus.RUNNING
        is_last_stage = (len(incremental_blocks) <= self.last_stage_max_blocks) or migrate_out_request.blocking_migration
        if not is_last_stage:
            src_blocks = incremental_blocks[:-1]
            stage_block_num = len(incremental_blocks) - 1
            dst_blocks = await migrate_in_ray_actor.execute_migration_method \
                            .remote("migrate_in_pre_alloc", migrate_out_request.request_id, stage_block_num)
        else:
            # last stage migration, stop inference, transfer all blocks
            migration_status = MigrationStatus.FINISHED_DONE
            self.backend_engine.remove_running_request(migrate_out_request.request_id)
            self.backend_engine.add_migrating_out_request_last_stage(migrate_out_request)
            stage_block_num = len(incremental_blocks)
            src_blocks = incremental_blocks[:]
            dst_blocks = await migrate_in_ray_actor.execute_migration_method \
                            .remote("migrate_in_pre_alloc", migrate_out_request.request_id, stage_block_num)

        if len(dst_blocks) != len(src_blocks):
            # migrate-in instance failed to prev alloc
            if is_last_stage:
                self.backend_engine.add_running_request(migrate_out_request)
                self.backend_engine.remove_migrating_out_request_last_stage(migrate_out_request)
            migration_status = MigrationStatus.ABORTED_DST
            return migration_status
        # do stage send/recv
        migrate_out_request.stage_timestamps.append(time.time())
        migrate_out_request.stage_num_blocks_list.append(stage_block_num)
        # TODO(ZeldaHuang): send_blocks in migrate_in_pre_alloc/migrate_in_last_stage
        await self.backend_engine.send_blocks(migrate_in_ray_actor, src_blocks, dst_blocks)
        if not is_last_stage and migrate_out_request.should_abort_migration():
            # migrate-out request abort by scheduler during send/recv
            migration_status = MigrationStatus.ABORTED_SRC

        return migration_status

    async def migrate_out_multistage(self, migrate_in_ray_actor: "ray.actor.ActorHandle",  migrate_out_request: LlumnixRequest) -> "MigrationStatus":
        """Migrate out requests to a specified instance, return migrated request id.
        Args:
        dst_instance_name:instance actor name, used to get ray actor handle
        """
        state_count = 0
        while state_count < self.max_stages:
            state_count += 1
            status = await self.migrate_out_onestage(migrate_in_ray_actor, migrate_out_request)
            if MigrationStatus.is_finished(status):
                return status
        # exceed max stages
        return MigrationStatus.ABORTED_SRC

    def migrate_in_pre_alloc(self, request_id: str, block_num: int) -> List[int]:
        """prev alloc blocks to migrate in request
        """
        pre_alloc_blocks = self.backend_engine.pre_alloc(request_id ,block_num)
        if len(pre_alloc_blocks) != block_num:
            # failed to alloc, abort request
            self.free_dst_pre_alloc_cache(request_id)
        return pre_alloc_blocks

    def free_dst_pre_alloc_cache(self, request_id: str = None) -> None:
        self.backend_engine.free_dst_pre_alloc_cache(request_id)
