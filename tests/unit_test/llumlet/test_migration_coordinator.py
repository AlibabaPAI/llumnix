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

from unittest.mock import MagicMock, patch, AsyncMock

import math
import ray
import pytest

from llumnix.llumlet.migration_coordinator import MigrationCoordinator
from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.llumlet.migration_coordinator import MigrationStatus

# pylint: disable=unused-import
from tests.conftest import ray_env

from .test_local_migration_scheduler import MockRequest

@ray.remote
def ray_remote_call(ret):
    return ret

def init_migration_coordinator(backend_engine,
                               migration_last_stage_max_blocks=1,
                               migration_max_stages=3):
    # Create an instance of MigrationCoordinator
    coordinator = MigrationCoordinator(
        "0",
        backend_engine,
        BackendType.VLLM,
        request_migration_policy="SR",
        migration_last_stage_max_blocks=migration_last_stage_max_blocks,
        migration_max_stages=migration_max_stages,
    )
    return coordinator

@pytest.mark.asyncio
async def test_migrate_out_onestage(ray_env):
    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    dst_instance_actor = MagicMock()
    migrate_out_request = MagicMock()
    coordinator = init_migration_coordinator(backend_engine)

    # Mock method return values and test data
    src_blocks = [1, 2, 3]
    dst_blocks = [1, 2]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks, [], False
    migrate_out_request.n_blocks = 3
    migrate_out_request.should_abort_migration.return_value = False
    migrate_out_request.blocking_migration = False
    dst_instance_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)

    # Test normal migration scenario
    status = await coordinator._migrate_out_onestage(dst_instance_actor, migrate_out_request)
    assert status == MigrationStatus.RUNNING

    # Test the last stage of migration
    backend_engine.remove_running_request = AsyncMock()
    backend_engine.remove_running_request.return_value = True
    migrate_out_request.finished = False
    src_blocks = [3]
    dst_blocks = [3]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks, [], True
    migrate_out_request.should_abort_migration.return_value = False
    migrate_out_request.blocking_migration = False
    dst_instance_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)
    status = await coordinator._migrate_out_onestage(dst_instance_actor, migrate_out_request)
    assert status == MigrationStatus.FINISHED

    migrate_out_request = MagicMock()
    # Test migration dst aborted scenario
    src_blocks = [1, 2, 3]
    dst_blocks = []
    backend_engine.get_request_incremental_blocks.return_value = src_blocks, [], False
    migrate_out_request.n_blocks = 3
    migrate_out_request.should_abort_migration.return_value = False
    migrate_out_request.blocking_migration = False
    dst_instance_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)
    status = await coordinator._migrate_out_onestage(dst_instance_actor, migrate_out_request)
    assert status == MigrationStatus.ABORTED_DST

    # Test migration src aborted scenario
    migrate_out_request = MagicMock()
    src_blocks = [1, 2, 3]
    dst_blocks = [1, 2]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks, [], False
    migrate_out_request.n_blocks = 3
    migrate_out_request.should_abort_migration.return_value = True
    migrate_out_request.blocking_migration = False
    dst_instance_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)
    status = await coordinator._migrate_out_onestage(dst_instance_actor, migrate_out_request)
    assert status == MigrationStatus.ABORTED_SRC

# ray_env should be passed after _migrate_out_onestage
@patch.object(MigrationCoordinator, '_migrate_out_onestage')
@pytest.mark.asyncio
async def test_migrate_out_running_request(_, ray_env):
    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    dst_instance_actor = MagicMock()
    migrate_out_request = MockRequest("1", 1, math.inf)

    # Create an instance of MigrationCoordinator
    migration_max_stages = 3
    coordinator = init_migration_coordinator(backend_engine, migration_max_stages=migration_max_stages)
    dst_instance_actor = MagicMock()
    dst_instance_actor.execute_engine_method = MagicMock()
    dst_instance_actor.execute_engine_method.remote = MagicMock()
    dst_instance_actor.execute_engine_method.remote.return_value = ray_remote_call.remote([1])
    dst_instance_actor.execute_migration_method.remote.return_value = ray_remote_call.remote([1])
    coordinator._migrate_out_onestage.side_effect = [MigrationStatus.FINISHED]
    status = await coordinator._migrate_out_running_request(dst_instance_actor, migrate_out_request)
    assert coordinator._migrate_out_onestage.call_count == 1
    assert status == MigrationStatus.FINISHED

    coordinator._migrate_out_onestage.side_effect = [MigrationStatus.RUNNING,
                                                     MigrationStatus.RUNNING,
                                                     MigrationStatus.RUNNING,
                                                     MigrationStatus.RUNNING]
    status = await coordinator._migrate_out_running_request(dst_instance_actor, migrate_out_request)
    assert coordinator._migrate_out_onestage.call_count == migration_max_stages + 1
    assert status == MigrationStatus.ABORTED_SRC

@pytest.mark.asyncio
async def test_migrate_out_waiting_request():
    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    dst_instance_actor = MagicMock()
    migrate_out_request = MagicMock()

    # Create an instance of MigrationCoordinator
    coordinator = init_migration_coordinator(backend_engine)

    # Test FINISHED
    migrate_out_request.prefill_num_blocks = 3
    dst_blocks = [1, 2, 3]
    dst_instance_actor.execute_engine_method = MagicMock()
    dst_instance_actor.execute_engine_method.remote = MagicMock()
    dst_instance_actor.execute_engine_method.remote.return_value = ray_remote_call.remote(dst_blocks)
    dst_instance_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)
    status = await coordinator._migrate_out_waiting_request(dst_instance_actor, migrate_out_request)
    assert status == MigrationStatus.FINISHED

    # Test FINISHED_ABORTED
    migrate_out_request.prefill_num_blocks = 2
    status = await coordinator._migrate_out_waiting_request(dst_instance_actor, migrate_out_request)
    assert status == MigrationStatus.ABORTED_DST
