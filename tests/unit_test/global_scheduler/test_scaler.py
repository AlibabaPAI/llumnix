import asyncio
import os
import copy
import subprocess
import shutil

import ray
import pytest
import torch

from vllm import EngineArgs

from llumnix.arg_utils import ManagerArgs, InstanceArgs, EntrypointsArgs, LaunchArgs
from llumnix.queue.queue_type import QueueType
from llumnix.entrypoints.vllm.arg_utils import VllmEngineArgs
from llumnix.ray_utils import get_scaler_name
from llumnix.scaler import Scaler
from llumnix.entrypoints.utils import LaunchMode
from llumnix.ray_utils import (
    get_server_name,
    remove_placement_group,
    INSTANCE_NAME_PREFIX,
    kill_server,
    kill_instance,
    get_placement_group_name,
    get_instance_name,
    get_manager_name,
)
from llumnix.arg_utils import save_engine_args
from llumnix.backends.backend_interface import BackendType
from llumnix.utils import random_uuid
from llumnix.instance_info import InstanceType
from llumnix.manager import Manager

# pylint: disable=unused-import
from tests.conftest import ray_env, cleanup_ray_env_func, ray_stop, ray_start
from tests.utils import try_convert_to_local_path
from tests.unit_test.global_scheduler.test_manager import init_instances

def init_scaler(manager_args = None):
    try:
        manager_args = ManagerArgs(enable_migration=True)
        manager_args.log_instance_info = False
        scaler = Scaler.from_args(
            entrypoints_args=None,
            manager_args=manager_args,
            instance_args=InstanceArgs(migration_backend="rayrpc"),
            engine_args=None,
            launch_args=None,
        )
    except ValueError:
        scaler = ray.get_actor(get_scaler_name(), namespace='llumnix')
    ray.get(scaler.is_ready.remote())

    return scaler

@pytest.fixture
def scaler():
    scaler = init_scaler()
    ray.get(scaler.is_ready.remote())
    yield scaler

@ray.remote
class DummyScaler:
    def __init__(self):
        pass

def is_actor_exists(actor_name):
    try:
        ray.get_actor(actor_name, namespace='llumnix')
        return True
    except ValueError:
        return False

def is_placement_group_exists(pg_name):
    try:
        ray.util.get_placement_group(pg_name)
        return True
    except ValueError:
        return False

def test_init_instances(ray_env, scaler):
    engine_args = VllmEngineArgs(
        engine_args=EngineArgs(
            model=try_convert_to_local_path("facebook/opt-125m"),
            download_dir="/mnt/model",
            worker_use_ray=True,
            enforce_eager=True,
        )
    )
    node_id = ray.get_runtime_context().get_node_id()
    _, instances = ray.get(scaler.init_instances.remote(
        QueueType("rayqueue"), InstanceArgs(), engine_args, node_id))
    num_instances = len(instances)
    manager_args = ManagerArgs()
    assert num_instances == manager_args.initial_instances

def test_init_instances_sim(ray_env, scaler):
    # pylint: disable=import-outside-toplevel
    # cannot catch by pytest.raises
    try:
        engine_args = VllmEngineArgs(
            engine_args=EngineArgs(
                model=try_convert_to_local_path("facebook/opt-125m"),
                download_dir="/mnt/model",
                worker_use_ray=True,
                enforce_eager=True,
            )
        )
        node_id = ray.get_runtime_context().get_node_id()
        _, _ = ray.get(
            scaler.init_instances.remote(
                QueueType("rayqueue"),
                InstanceArgs(profiling_result_file_path="/"),
                engine_args,
                node_id
            )
        )
    # pylint: disable=broad-except
    except Exception as e:
        assert isinstance(e, IsADirectoryError)

def init_scaler_with_launch_mode(launch_mode, enable_pd_disagg=False, pd_ratio="1:3", max_instances=-1,
                                 enable_pdd_node_affinity_scheduling=False):
    manager_args = ManagerArgs(enable_port_increment=True, enable_pd_disagg=enable_pd_disagg,
                               pd_ratio=pd_ratio, max_instances=max_instances,
                               enable_pdd_node_affinity_scheduling=enable_pdd_node_affinity_scheduling)
    instance_args = InstanceArgs(migration_backend="rayrpc")
    entrypoints_args = EntrypointsArgs(host="127.0.0.1", port=8000)
    engine_args = VllmEngineArgs(
        engine_args=EngineArgs(
            model=try_convert_to_local_path("facebook/opt-125m"),
            download_dir="/mnt/model",
            worker_use_ray=True,
            enforce_eager=True,
        )
    )
    launch_args = LaunchArgs(launch_mode=launch_mode, backend_type=BackendType.VLLM)

    scaler: Scaler = Scaler.from_args(entrypoints_args=entrypoints_args, manager_args=manager_args,
                                      instance_args=instance_args, engine_args=engine_args,
                                      launch_args=launch_args)
    ray.get(scaler.is_ready.remote())
    manager: Manager = ray.get_actor(get_manager_name(), namespace='llumnix')

    return scaler, manager, manager_args, entrypoints_args, engine_args, launch_args

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required")
async def test_init_server_and_get_instance_deployment_states_and_instance_and_clear_instance_ray_resources(ray_env):
    scaler, manager, _, _, engine_args, _ = init_scaler_with_launch_mode(LaunchMode.LOCAL)
    instance_id = random_uuid()
    pg = ray.get(
        scaler._init_placement_group.remote(
            get_placement_group_name(instance_id), engine_args, init_server=True
        )
    )
    pg = ray.util.get_placement_group(get_placement_group_name(instance_id))
    ray.get(pg.ready())
    ray.get(scaler._init_server_and_instance.remote(instance_id, EntrypointsArgs(), InstanceArgs(), engine_args, pg))

    # wait for scale up
    await asyncio.sleep(5.0)
    instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
    ray.get(instance.is_ready.remote())
    await asyncio.sleep(5.0)
    server = ray.get_actor(get_server_name(instance_id), namespace="llumnix")
    ray.get(server.is_ready.remote())
    num_instances = ray.get(manager.scale_up.remote(instance_id, instance, InstanceType("no_constraints"), pg))
    assert num_instances == 1

    pg_created, server_alive, instance_alive = ray.get(scaler._get_instance_deployment_states.remote(instance_id))
    assert pg_created and server_alive and instance_alive

    # test clear_instance_ray_resources
    ray.get(scaler.clear_instance_ray_resources.remote(instance_id))
    # wait for remove and kill
    await asyncio.sleep(5.0)

    pg_exists = is_placement_group_exists(get_placement_group_name(instance_id))
    assert not pg_exists
    server_exists = is_actor_exists(get_server_name(instance_id))
    assert not server_exists
    instance_exists = is_actor_exists(get_instance_name(instance_id))
    assert not instance_exists

    pg_created, server_alive, instance_alive = ray.get(scaler._get_instance_deployment_states.remote(instance_id))
    assert not pg_created and not server_alive and not instance_alive

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_auto_scale_up_loop_and_get_cluster_deployment_states(ray_env):
    scaler, manager, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.GLOBAL, max_instances=4)
    await asyncio.sleep(60.0)

    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    actor_infos = ray.util.list_named_actors(all_namespaces=True)
    instance_ids = [actor_info['name'].split("_")[-1] for actor_info in actor_infos
                    if actor_info['name'].startswith(INSTANCE_NAME_PREFIX)]
    assert len(instance_ids) == 4
    ray.get(scaler.clear_instance_ray_resources.remote(instance_ids[0]))
    ray.get(scaler.clear_instance_ray_resources.remote(instance_ids[1]))
    await asyncio.sleep(60.0)

    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_check_deployment_states_loop_and_auto_scale_up_loop(ray_env):
    scaler, manager, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.GLOBAL, max_instances=4)
    await asyncio.sleep(60.0)

    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    actor_infos = ray.util.list_named_actors(all_namespaces=True)
    instance_ids = [actor_info['name'].split("_")[-1] for actor_info in actor_infos
                    if actor_info['name'].startswith(INSTANCE_NAME_PREFIX)]
    assert len(instance_ids) == 4
    remove_placement_group(instance_ids[0])
    await kill_server(instance_ids[1])
    await kill_instance(instance_ids[2])
    # Wait for check deployment states, scale down instance and auto scale up.
    await asyncio.sleep(90.0)

    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

# Scaler requires to get manager actor.
def test_pd_disagg_gloal_launch_instance_type(ray_env):
    manager_args = ManagerArgs(
        enable_pd_disagg=True,
        enable_engine_pd_disagg=False,
        pd_ratio="1:2",
        enable_pdd_node_affinity_scheduling=False,
        enable_port_increment=True,
        enable_port_offset_store=False,
        load_registered_service=False,
    )
    # Scaler will get scaler actor handle inside the constructor.
    _ = DummyScaler.options(name=get_scaler_name()).remote()
    scaler = Scaler(None, manager_args, None, None, None)

    assert scaler._get_next_instance_type(0, 0, [1, 2]) == InstanceType.PREFILL
    scaler.inflight_num_prefill_instances += 1

    assert scaler._get_next_instance_type(0, 0, [1, 2]) == InstanceType.DECODE
    scaler.inflight_num_decode_instances += 1

    scaler.inflight_num_prefill_instances = 0
    scaler.inflight_num_decode_instances = 0
    assert scaler._get_next_instance_type(1, 1, [1, 2]) == InstanceType.DECODE
    assert scaler._get_next_instance_type(1, 2, [1, 2]) == InstanceType.PREFILL

    assert scaler._get_next_instance_type(3, 5, [1, 2]) == InstanceType.DECODE
    assert scaler._get_next_instance_type(3, 6, [1, 2]) == InstanceType.PREFILL
    assert scaler._get_next_instance_type(3, 7, [1, 2]) == InstanceType.PREFILL

@pytest.mark.parametrize("load_registered_service", [False, True])
@pytest.mark.parametrize("enable_pd_disagg", [False, True])
def test_load_registered_service(ray_env, load_registered_service, enable_pd_disagg):
    engine_args = VllmEngineArgs(engine_args=EngineArgs(model="no_constraints"))
    save_path = 'test'
    save_key = "test"
    load_registered_service_path = os.path.join(save_path, save_key)
    instance_type_list = ['no_constraints']
    if load_registered_service:
        if enable_pd_disagg:
            instance_type_list = ['prefill', 'decode']
        for instance_type in instance_type_list:
            put_engine_args = copy.deepcopy(engine_args)
            put_engine_args.engine_args.model = instance_type
            save_engine_args(instance_type, save_path, put_engine_args, save_key)
    manager_args = ManagerArgs(
        enable_pd_disagg=enable_pd_disagg,
        enable_engine_pd_disagg=False,
        pd_ratio="1:2",
        enable_pdd_node_affinity_scheduling=False,
        enable_port_increment=True,
        enable_port_offset_store=False,
        load_registered_service=load_registered_service,
        load_registered_service_path=load_registered_service_path,
    )
    # Scaler will get scaler actor handle inside the constructor.
    _ = DummyScaler.options(name=get_scaler_name()).remote()
    scaler = Scaler(None, manager_args, None, None, None)
    for instance_type in instance_type_list:
        get_engine_args = scaler.llumnix_engine_args_factory.gen_next_engine_args(engine_args, instance_type)
        if load_registered_service:
            assert get_engine_args.load_engine_args().model == instance_type
        else:
            assert get_engine_args.load_engine_args().model == 'no_constraints'
    if load_registered_service:
        shutil.rmtree(load_registered_service_path)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_pd_disagg_gloal_launch_deployment_and_auto_scale_up_loop(ray_env):
    scaler, manager, _, _, _, _ = init_scaler_with_launch_mode(
        LaunchMode.GLOBAL, max_instances=4, enable_pd_disagg=True, pd_ratio="1:1")
    await asyncio.sleep(60.0)
    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    num_prefill_instances = 0
    num_decode_instances = 0
    prefill_instance_ids = []
    decode_instance_ids = []
    for instance_id in curr_instances:
        instance_handle = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
        instance_type = ray.get(instance_handle.get_instance_type.remote())
        if instance_type == InstanceType.PREFILL:
            num_prefill_instances += 1
            prefill_instance_ids.append(ray.get(instance_handle.get_instance_info.remote()).instance_id)
        elif instance_type == InstanceType.DECODE:
            num_decode_instances += 1
            decode_instance_ids.append(ray.get(instance_handle.get_instance_info.remote()).instance_id)

    assert num_prefill_instances == 2 and num_decode_instances == 2
    assert set(prefill_instance_ids).union(set(decode_instance_ids)) == set(curr_instances)

    await kill_instance(prefill_instance_ids[0])
    await asyncio.sleep(10.0)

    await kill_instance(prefill_instance_ids[1])
    await asyncio.sleep(10.0)

    await kill_instance(decode_instance_ids[1])
    await asyncio.sleep(90.0)
    alive_decode_instance_id = decode_instance_ids[0]

    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    num_prefill_instances = 0
    num_decode_instances = 0
    decode_instance_ids = []
    for instance_id in curr_instances:
        instance_handle = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
        instance_type = ray.get(instance_handle.get_instance_type.remote())
        if instance_type == InstanceType.PREFILL:
            num_prefill_instances += 1
        elif instance_type == InstanceType.DECODE:
            num_decode_instances += 1
            decode_instance_ids.append(instance_id)

    assert num_prefill_instances == 2 and num_decode_instances == 2
    assert alive_decode_instance_id in decode_instance_ids

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_pd_disagg_deployment_states(ray_env):
    manager_args = ManagerArgs(enable_migration=True, enable_pd_disagg=True, pd_ratio="1:2")
    engine_args = VllmEngineArgs(
        engine_args=EngineArgs(
            model=try_convert_to_local_path("facebook/opt-125m"),
            download_dir="/mnt/model",
            worker_use_ray=True,
            enforce_eager=True,
        )
    )
    scaler: Scaler = Scaler.from_args(entrypoints_args=EntrypointsArgs(), manager_args=manager_args,
                                      instance_args=InstanceArgs(migration_backend="rayrpc"),
                                      engine_args=engine_args, launch_args=LaunchArgs(LaunchMode.LOCAL, BackendType.VLLM),)
    ray.get(scaler.is_ready.remote())
    manager: Manager = ray.get_actor(get_manager_name(), namespace="llumnix")
    ray.get(manager.is_ready.remote())
    assert not ray.get(scaler._check_pd_deployment_states.remote())

    prefill_instance_ids, prefill_instances = init_instances(3)
    decode_instance_ids, decode_instances = init_instances(3)

    ray.get(manager.scale_up.remote(prefill_instance_ids, prefill_instances,
                     [InstanceType("prefill")]*len(prefill_instance_ids), [None]*len(prefill_instance_ids)))
    assert ray.get(scaler._check_pd_deployment_states.remote()) in prefill_instance_ids
    ray.get(manager.scale_down.remote(prefill_instance_ids))
    ray.get(manager.scale_up.remote(decode_instance_ids, decode_instances,
                     [InstanceType("decode")]*len(decode_instance_ids), [None]*len(decode_instance_ids)))
    assert ray.get(scaler._check_pd_deployment_states.remote()) in decode_instance_ids

    prefill_instance_ids2, prefill_instances2 = init_instances(3)
    ray.get(manager.scale_up.remote(prefill_instance_ids2, prefill_instances2,
                     [InstanceType("prefill")]*len(prefill_instance_ids2), [None]*len(prefill_instance_ids2)))
    assert not ray.get(scaler._check_pd_deployment_states.remote())

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_auto_scale_up_loop_max_instances(ray_env):
    _, manager, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.GLOBAL, "rayqueue", max_instances=2)
    await asyncio.sleep(60.0)
    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 2

# 1. Resources cannot be specified in ray.init() when there is a existing ray cluster.
# 2. If specify resources in ray_start() of pytest_sessionstart, when running serveral tests, it's fine.
# However, when running the whole unit test, placement group with resources cannot be scheduled for unknown reason.
# 3. If restart ray cluster, it will raise restart gcs error between tests sometimes.
@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_auto_scale_up_loop_enable_pdd_node_affinity_scheduling():
    ray_stop()
    subprocess.run(["ray", "start", "--head", "--port=6379", "--resources={\"PREFILL_GPU\": 2, \"DECODE_GPU\": 2}"],
            check=False, stdout=subprocess.DEVNULL)
    ray.init(ignore_reinit_error=True, namespace="llumnix")

    scaler, _, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.GLOBAL,
                                                         max_instances=4,
                                                         enable_pd_disagg=True,
                                                         enable_pdd_node_affinity_scheduling=True)
    await asyncio.sleep(60.0)

    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    num_prefill_instances = 0
    num_decode_instances = 0
    num_no_constraints_instances = 0
    for instance_id in curr_instances:
        instance_handle = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
        instance_type = ray.get(instance_handle.get_instance_type.remote())
        if instance_type == InstanceType.PREFILL:
            num_prefill_instances += 1
        elif instance_type == InstanceType.DECODE:
            num_decode_instances += 1
        else:
            num_no_constraints_instances += 1

    assert num_prefill_instances == 2 and num_decode_instances == 2 and num_no_constraints_instances == 0

    cleanup_ray_env_func()
    ray_start()
