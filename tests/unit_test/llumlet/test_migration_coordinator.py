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
import math
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import ray
import pytest

from llumnix.llumlet.migration_coordinator import MigrationCoordinator
from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.llumlet.migration_coordinator import MigrationStatus
from llumnix.constants import PENDING_MIGRATE_IN_TIMEOUT
from llumnix.utils import random_uuid, MigrationResponse
from llumnix.llumlet.request import RequestStatus

# pylint: disable=unused-import
from tests.conftest import ray_env

from .test_local_migration_scheduler import MockRequest

@ray.remote
def ray_remote_call(ret):
    return ret

async def async_return(ret):
    return ret

def init_migration_coordinator(backend_engine,
                               migration_last_stage_max_blocks=1,
                               migration_max_stages=3):
    # Create an instance of MigrationCoordinator
    migration_coordinator = MigrationCoordinator(
        "0",
        backend_engine,
        BackendType.VLLM,
        request_migration_policy="SR",
        migration_last_stage_max_blocks=migration_last_stage_max_blocks,
        migration_max_stages=migration_max_stages,
    )
    return migration_coordinator

@pytest.mark.asyncio
async def test_migrate_out_onestage(ray_env):
    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    dst_instance_actor = MagicMock()
    migrate_out_request = MagicMock()
    migration_coordinator = init_migration_coordinator(backend_engine)
    migrate_out_request.request_id = "0"

    # Test normal migration scenario
    src_blocks = [1, 2, 3]
    dst_blocks = [1, 2, 3]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks, [], False
    migrate_out_request.n_blocks = 3
    migrate_out_request.should_abort_migration.return_value = False
    migrate_out_request.blocking_migration = False
    dst_instance_actor.execute_migration_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=True, return_value=dst_blocks))
    migration_coordinator.pending_migrate_in_requests["0"] = time.time()
    status = await migration_coordinator._migrate_out_onestage(dst_instance_actor, "0", migrate_out_request, True)
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
    dst_instance_actor.execute_migration_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=True, return_value=dst_blocks))
    migration_coordinator.pending_migrate_in_requests["0"] = time.time()
    status = await migration_coordinator._migrate_out_onestage(dst_instance_actor, "0", migrate_out_request, True)
    assert status == MigrationStatus.FINISHED

    # Test migration dst aborted scenario

    # 1: dst aborted in pre_alloc
    migrate_out_request = MagicMock()
    src_blocks = [1, 2, 3]
    dst_blocks = []
    backend_engine.get_request_incremental_blocks.return_value = src_blocks, [], False
    migrate_out_request.n_blocks = 3
    migrate_out_request.should_abort_migration.return_value = False
    migrate_out_request.blocking_migration = False
    dst_instance_actor.execute_migration_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=False, return_value=None))
    migration_coordinator.pending_migrate_in_requests["0"] = time.time()
    status = await migration_coordinator._migrate_out_onestage(dst_instance_actor, "0", migrate_out_request, True)
    assert status == MigrationStatus.ABORTED_DST

    # 2: dst aborted in send_cache
    migrate_out_request = MagicMock()
    src_blocks = [1, 2, 3]
    dst_blocks = [1, 2, 3]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks, [], False
    migrate_out_request.n_blocks = 3
    migrate_out_request.should_abort_migration.return_value = False
    migrate_out_request.blocking_migration = False
    dst_instance_actor.execute_migration_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=True, return_value=dst_blocks))
    backend_engine.send_cache.return_value = MigrationResponse(success=False, return_value=None)
    migration_coordinator.pending_migrate_in_requests["0"] = time.time()
    status = await migration_coordinator._migrate_out_onestage(dst_instance_actor, "0", migrate_out_request, True)
    assert status == MigrationStatus.ABORTED_DST

    # Test migration src aborted scenario
    migrate_out_request = MagicMock()
    src_blocks = [1, 2, 3]
    dst_blocks = [1, 2, 3]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks, [], False
    migrate_out_request.n_blocks = 3
    migrate_out_request.should_abort_migration.return_value = True
    migrate_out_request.blocking_migration = False
    dst_instance_actor.execute_migration_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=True, return_value=dst_blocks))
    migration_coordinator.pending_migrate_in_requests["0"] = time.time()
    status = await migration_coordinator._migrate_out_onestage(dst_instance_actor, "0", migrate_out_request, True)
    assert status == MigrationStatus.ABORTED_SRC

# ray_env should be passed after _migrate_out_onestage
@patch.object(MigrationCoordinator, '_migrate_out_onestage')
@pytest.mark.asyncio
async def test_migrate_out_running_request(_, ray_env):
    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    dst_instance_actor = MagicMock()
    migrate_out_request = MockRequest("1", 1, math.inf)
    migration_max_stages = 3
    migration_coordinator = init_migration_coordinator(backend_engine, migration_max_stages=migration_max_stages)

    # Test FINISHED
    dst_instance_actor.execute_engine_method = MagicMock()
    dst_instance_actor.execute_engine_method.remote = MagicMock()
    dst_instance_actor.execute_engine_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=True, return_value=[1]))
    dst_instance_actor.execute_migration_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=True, return_value=[1]))
    migration_coordinator._migrate_out_onestage.side_effect = [MigrationStatus.FINISHED]
    status = await migration_coordinator._migrate_out_running_request(dst_instance_actor, "0", migrate_out_request)
    assert migration_coordinator._migrate_out_onestage.call_count == 1
    assert status == MigrationStatus.FINISHED

    # Test ABORTED_SRC
    migration_coordinator._migrate_out_onestage.side_effect = [
        MigrationStatus.RUNNING, MigrationStatus.RUNNING, MigrationStatus.RUNNING, MigrationStatus.RUNNING]
    status = await migration_coordinator._migrate_out_running_request(dst_instance_actor, "0", migrate_out_request)
    assert migration_coordinator._migrate_out_onestage.call_count == migration_max_stages + 1
    assert status == MigrationStatus.ABORTED_SRC

@pytest.mark.asyncio
async def test_migrate_out_waiting_request():
    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    dst_instance_actor = MagicMock()
    migrate_out_request = MagicMock()
    migration_coordinator = init_migration_coordinator(backend_engine)

    # Test FINISHED
    dst_blocks = [1, 2, 3]
    dst_instance_actor.execute_migration_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=True, return_value=dst_blocks))
    status = await migration_coordinator._migrate_out_waiting_request(dst_instance_actor, "0", migrate_out_request)
    assert status == MigrationStatus.FINISHED

    # Test ABORTED_DST
    dst_instance_actor.execute_migration_method.remote.return_value = \
        ray_remote_call.remote(MigrationResponse(success=False, return_value=None))
    status = await migration_coordinator._migrate_out_waiting_request(dst_instance_actor, "0", migrate_out_request)
    assert status == MigrationStatus.ABORTED_DST

@pytest.mark.asyncio
async def test_migrate_out_one_request():
    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    dst_instance_actor = MagicMock()
    migrate_out_request = MagicMock()
    request_id = random_uuid()
    migration_coordinator = init_migration_coordinator(backend_engine)
    migration_coordinator._migrate_out_running_request = MagicMock()
    migration_coordinator._dst_commit_dst_request = MagicMock()
    migrate_out_request.request_id = request_id
    migrate_out_request.status = RequestStatus.RUNNING
    migration_coordinator._dst_free_pre_alloc_cache = MagicMock()

    # setting the return value for each _migrate_out_running_request call, because different coroutines are required.
    migration_coordinator._migrate_out_running_request.side_effect = \
        [async_return(MigrationStatus.FINISHED), async_return(MigrationStatus.FINISHED),
         async_return(MigrationStatus.ABORTED_SRC), async_return(MigrationStatus.ABORTED_DST)]

    # 1: dst_commit_dst_request success
    migration_coordinator._dst_commit_dst_request.return_value = async_return(MigrationResponse(success=True, return_value=None))
    migrated_request = await migration_coordinator._migrate_out_one_request(dst_instance_actor, request_id, migrate_out_request)
    assert len(migrated_request) == 1 and migrated_request[0] == request_id
    # 2: dst_commit_dst_request not success
    migration_coordinator._dst_commit_dst_request.return_value = async_return(MigrationResponse(success=False, return_value=None))
    migrated_request = await migration_coordinator._migrate_out_one_request(dst_instance_actor, request_id, migrate_out_request)
    assert len(migrated_request) == 0

    assert migration_coordinator._dst_free_pre_alloc_cache.call_count == 0
    migration_coordinator._dst_free_pre_alloc_cache.return_value = async_return(MigrationResponse(success=True, return_value=None))
    migrated_request = await migration_coordinator._migrate_out_one_request(dst_instance_actor, request_id, migrate_out_request)
    assert migration_coordinator._dst_free_pre_alloc_cache.call_count == 1

    migrated_request = await migration_coordinator._migrate_out_one_request(dst_instance_actor, request_id, migrate_out_request)
    assert migration_coordinator._dst_free_pre_alloc_cache.call_count == 1

@pytest.mark.asyncio
async def test_pending_migrate_in_timeout():
     # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    migrate_out_request = MagicMock()
    migration_coordinator = init_migration_coordinator(backend_engine)
    request_id = random_uuid()
    migrate_out_request.request_id = request_id

    backend_engine.free_pre_alloc_cache.return_value = None

    # test first_stage pre_alloc_cache timeout
    backend_engine.pre_alloc_cache.return_value = MigrationResponse(success=True, return_value=[0, 1, 2])
    assert backend_engine.free_pre_alloc_cache.call_count == 0
    response = migration_coordinator.pre_alloc_cache(request_id, "running_migrating", 0.0, 3, [], True)
    assert response.return_value == [0, 1, 2]
    assert request_id in migration_coordinator.pending_migrate_in_requests
    await asyncio.sleep(PENDING_MIGRATE_IN_TIMEOUT + 6.0)
    assert backend_engine.free_pre_alloc_cache.call_count == 1
    assert request_id not in migration_coordinator.pending_migrate_in_requests

    # test recv_cache timeout
    backend_engine.recv_cache.return_value = MigrationResponse(success=True, return_value=None)
    response = await migration_coordinator.recv_cache(request_id)
    assert response.success is False
    migration_coordinator.pending_migrate_in_requests[request_id] = time.time()
    response = await migration_coordinator.recv_cache(request_id)
    assert response.success is True
    assert request_id in migration_coordinator.pending_migrate_in_requests
    await asyncio.sleep(PENDING_MIGRATE_IN_TIMEOUT + 6.0)
    assert backend_engine.free_pre_alloc_cache.call_count == 2

    # test non-first_stage pre_alloc_cache timeout
    backend_engine.pre_alloc_cache.return_value = MigrationResponse(success=True, return_value=[0, 1, 2])
    assert request_id not in migration_coordinator.pending_migrate_in_requests
    response = migration_coordinator.pre_alloc_cache(request_id, "running_migrating", 0.0, 3, [], False)
    assert response.return_value == []
    migration_coordinator.pending_migrate_in_requests[request_id] = time.time()
    response = migration_coordinator.pre_alloc_cache(request_id, "running_migrating", 0.0, 3, [], False)
    assert response.return_value == [0, 1, 2]
    assert request_id in migration_coordinator.pending_migrate_in_requests
    await asyncio.sleep(PENDING_MIGRATE_IN_TIMEOUT + 6.0)
    assert backend_engine.free_pre_alloc_cache.call_count == 3
    assert request_id not in migration_coordinator.pending_migrate_in_requests

    # test commit_dst_request_timeout
    response = await migration_coordinator.commit_dst_request(migrate_out_request)
    assert response.success is False
    migration_coordinator.pending_migrate_in_requests[request_id] = time.time()
    backend_engine.commit_dst_request.return_value = MigrationResponse(success=True, return_value=None)
    response = await migration_coordinator.commit_dst_request(migrate_out_request)
    assert response.success is True
    assert request_id not in migration_coordinator.pending_migrate_in_requests
