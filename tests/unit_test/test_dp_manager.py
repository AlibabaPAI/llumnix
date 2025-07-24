import time
import pytest

import ray
import torch

from vllm import EngineArgs

from llumnix.dp_manager import DPManager
from llumnix.utils import random_uuid, InstanceType, BackendType
from llumnix.arg_utils import EntrypointsArgs, InstanceArgs
from llumnix.entrypoints.vllm.arg_utils import VLLMEngineArgs
from llumnix.ray_utils import (
    initialize_placement_group,
    get_placement_group_name,
    list_actor_names_by_actor_type,
    LlumnixActor,
    get_llumnix_actor_name,
    get_actor_handle,
    list_placement_group_infos_by_state,
)

# pylint: disable=unused-import
from tests.conftest import ray_env
from tests.utils import try_convert_to_local_path


@ray.remote(num_cpus=0)
class MockManager:
    def __init__(self):
        pass

    def scale_up(self, instance_ids, instance_actor_handles, instance_types):
        pass


@ray.remote(num_cpus=0)
class MockInstance:
    def __init__(self):
        pass

    def is_ready(self):
        return True

    def stop(self):
        return True


@ray.remote(num_cpus=0)
class MockServer:
    def __init__(self):
        time.sleep(5.0)
        self.dead_instance_ids = []

    def is_ready(self):
        return True

    def stop(self):
        return True

    def cancel_dead_instance_requests(self, dead_instance_ids):
        self.dead_instance_ids.extend(dead_instance_ids)

    def get_dead_instance_ids(self):
        return self.dead_instance_ids


def init_dp_manager(unit_id, scaler = None):
    manager = MockManager.options(name="mock_manager", namespace="llumnix").remote()
    instance_id = f"{unit_id}_{random_uuid()}"
    engine_args = VLLMEngineArgs(
        engine_args=EngineArgs(
            model=try_convert_to_local_path("facebook/opt-125m"),
            download_dir="/mnt/model",
            worker_use_ray=True,
            enforce_eager=True,
        )
    )
    placement_group = initialize_placement_group(get_placement_group_name(unit_id), num_cpus=3, num_gpus=1)
    dp_manager = DPManager.from_args(
        unit_id,
        InstanceType("neutral"),
        1,
        1,
        [instance_id],
        [EntrypointsArgs(port=8000)],
        [InstanceArgs()],
        [engine_args],
        placement_group,
        BackendType.VLLM,
        scaler,
        manager,
    )

    return dp_manager

def init_server(unit_id):
    instance_id = f"{unit_id}_{random_uuid()}"
    server = MockServer.options(name=get_llumnix_actor_name(LlumnixActor.SERVER, instance_id), namespace="llumnix").remote()
    ray.get(server.is_ready.remote())

    return instance_id, server

def init_instance(unit_id):
    instance_id = f"{unit_id}_{random_uuid()}"
    instance = MockInstance.options(name=get_llumnix_actor_name(LlumnixActor.INSTANCE, instance_id), namespace="llumnix").remote()
    ray.get(instance.is_ready.remote())

    return instance_id, instance

def list_curr_cluster_deployments():
    curr_pg_infos = list_placement_group_infos_by_state("CREATED")
    curr_instance_names = list_actor_names_by_actor_type(LlumnixActor.INSTANCE)
    curr_server_names = list_actor_names_by_actor_type(LlumnixActor.SERVER)
    return curr_pg_infos, curr_instance_names, curr_server_names

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required")
def test_connect_to_instances_and_servers(ray_env):
    # 1 instance, 0 server
    # test _clear_instance_ray_resources and stop
    unit_id = random_uuid()
    # NOTE(s5u13b): avoid gc.
    _, instance = init_instance(unit_id)
    dp_manager = init_dp_manager(unit_id)
    dp_manager_created = False
    try:
        ray.get(dp_manager.is_ready.remote())
        dp_manager_created = True
    # pylint: disable=bare-except
    except:
        pass
    assert dp_manager_created is False
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 0 and len(curr_instances) == 0 and len(curr_servers) == 0

    # 0 instance, 1 server
    # test _clear_instance_ray_resources and stop
    unit_id = random_uuid()
    _, server = init_server(unit_id)
    dp_manager = init_dp_manager(unit_id)
    dp_manager_created = False
    try:
        ray.get(dp_manager.is_ready.remote())
        dp_manager_created = True
    # pylint: disable=bare-except
    except:
        pass
    assert dp_manager_created is False
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 0 and len(curr_instances) == 0 and len(curr_servers) == 0

    # 1 instance, 1 server
    # test _connect_to_instances_and_servers
    unit_id = random_uuid()
    _, instance = init_instance(unit_id)
    _, server = init_server(unit_id)
    dp_manager = init_dp_manager(unit_id)
    dp_manager_created = False
    try:
        ray.get(dp_manager.is_ready.remote())
        dp_manager_created = True
    # pylint: disable=bare-except
    except:
        pass
    assert dp_manager_created is True
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 1 and len(curr_instances) == 1 and len(curr_servers) == 1

    # test _scale_up and stop
    dp_manager.stop.remote()
    time.sleep(5.0)
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 0 and len(curr_instances) == 0 and len(curr_servers) == 0

    # 0 instance, 0 server
    # test _init_instances_and_servers and _wait_for_instances_and_servers_ready
    unit_id = random_uuid()
    dp_manager = init_dp_manager(unit_id)
    dp_manager_alive = True
    try:
        ray.get(dp_manager.is_ready.remote())
    # pylint: disable=bare-except
    except:
        dp_manager_alive = False
    assert dp_manager_alive is True
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 1 and len(curr_instances) == 1 and len(curr_servers) == 1

    # test _scale_up and stop
    dp_manager.stop.remote()
    time.sleep(5.0)
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 0 and len(curr_instances) == 0 and len(curr_servers) == 0

    # 0 instance, 0 server
    # test _wait_for_instances_and_servers_ready
    unit_id = random_uuid()
    dp_manager = init_dp_manager(unit_id)
    ray.get(dp_manager.is_ready.remote())
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 1 and len(curr_instances) == 1 and len(curr_servers) == 1
    server = get_actor_handle(curr_servers[0])
    instance = get_actor_handle(curr_instances[0])
    start = time.time()
    ray.get(server.is_ready.remote())
    ray.get(instance.is_ready.remote())
    stop = time.time()
    # much less than 5.0 (server init time)
    assert stop - start < 1.0

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required")
@pytest.mark.parametrize("test_mode", ["instance", "server"])
def test_heartbeat_loop(ray_env, test_mode):
    unit_id = random_uuid()
    dp_manager = init_dp_manager(unit_id)
    ray.get(dp_manager.is_ready.remote())
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 1 and len(curr_instances) == 1 and len(curr_servers) == 1
    server = get_actor_handle(curr_servers[0])
    instance = get_actor_handle(curr_instances[0])
    if test_mode == "server":
        ray.kill(server)
    else:
        ray.kill(instance)
    time.sleep(30.0)
    curr_pgs, curr_instances, curr_servers = list_curr_cluster_deployments()
    assert len(curr_pgs) == 0 and len(curr_instances) == 0 and len(curr_servers) == 0
    dp_manager_alive = True
    try:
        ray.get(dp_manager.is_ready.remote())
    # pylint: disable=bare-except
    except:
        dp_manager_alive = False
    assert dp_manager_alive is False

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required")
def test_update_cached_cluster_servers_loop_and_update_cluster_actors(ray_env):
    dp_manager = init_dp_manager(random_uuid())
    ray.get(dp_manager.is_ready.remote())
    _, server1 = init_server(random_uuid())
    _, server2 = init_server(random_uuid())
    dead_instance_ids = ["0", "1"]
    ray.get(dp_manager._broadcast_dead_instances_to_cluster_servers.remote(dead_instance_ids))
    assert dead_instance_ids == ray.get(server1.get_dead_instance_ids.remote())
    assert dead_instance_ids == ray.get(server2.get_dead_instance_ids.remote())
