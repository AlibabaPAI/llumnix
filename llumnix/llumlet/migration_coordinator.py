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

import ray

from llumnix.logger import init_logger
from llumnix.llumlet.migrating_request import MigratingRequest
from llumnix.backends.backend_interface import BackendInterface

logger = init_logger(__name__)

class MigrationStatus(enum.Enum):
    """Status of Migration."""
    RUNNING = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_DONE = enum.auto()

    @staticmethod
    def is_finished(status: "MigrationStatus") -> bool:
        return status in [
            MigrationStatus.FINISHED_ABORTED,
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

    def migrate_out_onestage(self, migrate_in_ray_actor: "ray.actor.ActorHandle",  migrate_out_request: MigratingRequest, ) -> "MigrationStatus":
        """one-stage live migration until last stage
        """
        pre_stage_num_blocks = sum(migrate_out_request.stage_num_blocks_list)
        incremental_blocks = self.backend_engine.get_request_incremental_blocks(migrate_out_request.backend_request, pre_stage_num_blocks)
        # live migration, transfer all blocks except last one(currently updating)
        migration_status = MigrationStatus.RUNNING
        is_last_stage = (len(incremental_blocks) <= self.last_stage_max_blocks)
        if not is_last_stage:
            src_blocks = incremental_blocks[:-1]
            stage_block_num = len(incremental_blocks) - 1
            dst_blocks = ray.get(migrate_in_ray_actor.execute_migration_method \
                            .remote("migrate_in_pre_alloc", migrate_out_request.request_id, stage_block_num))
        else:
            # last stage migration, stop inference, transfer all blocks
            migration_status = MigrationStatus.FINISHED_DONE
            self.backend_engine.remove_running_request(migrate_out_request.request_id)
            self.backend_engine.add_migrating_out_request_last_stage(migrate_out_request.backend_request)
            stage_block_num = len(incremental_blocks)
            src_blocks = incremental_blocks[:]
            migrate_out_request.server_info = self.backend_engine.get_request_server_info(migrate_out_request.request_id)
            dst_blocks = ray.get(migrate_in_ray_actor.execute_migration_method \
                            .remote("migrate_in_last_stage", migrate_out_request, stage_block_num))

        if len(dst_blocks) != len(src_blocks):
            # migrate-in instance failed to prev alloc
            if is_last_stage:
                self.backend_engine.add_running_request(migrate_out_request.backend_request)
                self.backend_engine.remove_migrating_out_request_last_stage(migrate_out_request.backend_request)
            migration_status = MigrationStatus.FINISHED_ABORTED
            return migration_status
        # do stage send/recv
        migrate_out_request.stage_timestamps.append(time.time())
        migrate_out_request.stage_num_blocks_list.append(stage_block_num)
        # TODO(ZeldaHuang): send_blocks in migrate_in_pre_alloc/migrate_in_last_stage
        self.backend_engine.send_blocks(migrate_in_ray_actor, src_blocks, dst_blocks)
        if not is_last_stage and self.backend_engine.should_abort_migration(migrate_out_request.backend_request, \
                                                                            migrate_out_request.stage_timestamps[-1]):
            # migrate-out request abort by scheduler during send/recv
            migration_status = MigrationStatus.FINISHED_ABORTED

        return migration_status

    def migrate_out_multistage(self, migrate_in_ray_actor: "ray.actor.ActorHandle",  migrate_out_request: MigratingRequest) -> "MigrationStatus":
        """Migrate out requests to a specified instance, return migrated request id.
        Args:
        dst_instance_name:instance actor name, used to get ray actor handle
        """
        state_count = 0
        while state_count < self.max_stages:
            state_count += 1
            status = self.migrate_out_onestage(migrate_in_ray_actor, migrate_out_request)
            if MigrationStatus.is_finished(status):
                break
        return status

    def migrate_in_last_stage(self, request: MigratingRequest, block_num: int) -> List[int]:
        """last stage migrate in, update migrate in dict
        """
        pre_alloc_blocks = self.migrate_in_pre_alloc(request.request_id, block_num)
        if len(pre_alloc_blocks) == block_num:
            # Pass the server information of the request to dst instance.
            self.backend_engine.commit_dst_request(request.backend_request, request.server_info)
        return pre_alloc_blocks

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
