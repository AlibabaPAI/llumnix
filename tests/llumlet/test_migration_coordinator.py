import ray
from unittest.mock import MagicMock, patch

from llumnix.llumlet.migration_coordinator import MigrationCoordinator
from llumnix.llumlet.migrating_request import MigratingRequest
from llumnix.backends.backend_interface import BackendInterface
from llumnix.llumlet.llumlet import MigrationStatus

@ray.remote
def ray_remote_call(ret):
    return ret
# 测试用例
def test_migrate_out_onestage():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    migrate_in_ray_actor = MagicMock()
    migrate_out_request = MigratingRequest(1, "test_request")

    # Create an instance of MigrationCoordinator
    coordinator = MigrationCoordinator(backend_engine, 1, 3)

    # Mock method return values and test data
    src_blocks = [1, 2, 3]
    dst_blocks = [1, 2]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks
    backend_engine.should_abort_migration.return_value = False
    migrate_in_ray_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)

    # Test normal migration scenario
    status = coordinator.migrate_out_onestage(migrate_in_ray_actor, migrate_out_request)
    assert status == MigrationStatus.RUNNING

    # Test the last stage of migration
    src_blocks = [3]
    dst_blocks = [3]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks
    backend_engine.should_abort_migration.return_value = False
    migrate_in_ray_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)
    status = coordinator.migrate_out_onestage(migrate_in_ray_actor, migrate_out_request)
    assert status == MigrationStatus.FINISHED_DONE

    migrate_out_request = MigratingRequest(2, "test_request")
    # Test migration aborted scenario
    src_blocks = [1, 2, 3]
    dst_blocks = []
    backend_engine.get_request_incremental_blocks.return_value = src_blocks
    backend_engine.should_abort_migration.return_value = False
    migrate_in_ray_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)
    status = coordinator.migrate_out_onestage(migrate_in_ray_actor, migrate_out_request)
    assert status == MigrationStatus.FINISHED_ABORTED

    migrate_out_request = MigratingRequest(3, "test_request")
    src_blocks = [1, 2, 3]
    dst_blocks = [1, 2]
    backend_engine.get_request_incremental_blocks.return_value = src_blocks
    backend_engine.should_abort_migration.return_value = True
    migrate_in_ray_actor.execute_migration_method.remote.return_value = ray_remote_call.remote(dst_blocks)
    status = coordinator.migrate_out_onestage(migrate_in_ray_actor, migrate_out_request)
    assert status == MigrationStatus.FINISHED_ABORTED
    ray.shutdown()

@patch.object(MigrationCoordinator, 'migrate_out_onestage')
def test_migrate_out_multistage(migrate_out_onestage):
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Create mock objects
    backend_engine = MagicMock(spec=BackendInterface)
    migrate_in_ray_actor = MagicMock()
    migrate_out_request = MigratingRequest(1, "test_request")

    # Create an instance of MigrationCoordinator
    max_stages = 3
    coordinator = MigrationCoordinator(backend_engine, 1, max_stages)
    migrate_in_ray_actor = MagicMock()
    migrate_in_ray_actor.execute_engine_method = MagicMock()
    migrate_in_ray_actor.execute_engine_method.remote = MagicMock()
    migrate_in_ray_actor.execute_engine_method.remote.return_value = ray_remote_call.remote([1])
    migrate_in_ray_actor.execute_migration_method.remote.return_value = ray_remote_call.remote([1])
    coordinator.migrate_out_onestage.side_effect = [MigrationStatus.FINISHED_DONE]
    status = coordinator.migrate_out_multistage(migrate_in_ray_actor, migrate_out_request)
    assert coordinator.migrate_out_onestage.call_count == 1
    assert status == MigrationStatus.FINISHED_DONE

    max_stages = 3
    coordinator.migrate_out_onestage.side_effect = [MigrationStatus.RUNNING,
                                                    MigrationStatus.RUNNING,
                                                    MigrationStatus.RUNNING,
                                                    MigrationStatus.RUNNING]
    status = coordinator.migrate_out_multistage(migrate_in_ray_actor, migrate_out_request)
    assert coordinator.migrate_out_onestage.call_count == max_stages + 1
    assert status == MigrationStatus.FINISHED_ABORTED

    ray.shutdown()