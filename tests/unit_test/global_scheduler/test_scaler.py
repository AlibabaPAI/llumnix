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
from llumnix.entrypoints.vllm.arg_utils import VLLMEngineArgs
from llumnix.scaler import Scaler
from llumnix.ray_utils import (
    remove_placement_group,
    kill_dp_manager,
    get_placement_group_name,
    get_llumnix_actor_name,
    list_actor_names_by_actor_type,
    get_llumnix_actor_id,
    LlumnixActor,
    get_llumnix_actor_handle,
)
from llumnix.arg_utils import save_engine_args
from llumnix.utils import random_uuid, BackendType, LaunchMode, InstanceType
from llumnix.manager import Manager

# pylint: disable=unused-import
from tests.conftest import ray_env, cleanup_ray_env_func, ray_stop, ray_start
from tests.utils import try_convert_to_local_path
from tests.unit_test.global_scheduler.test_manager import init_instances
from tests.unit_test.test_dp_manager import init_dp_manager

def init_scaler(manager_args = None):
    try:
        manager_args = ManagerArgs(enable_routine_migration=True, enable_pre_stop_migration=False)
        manager_args.log_instance_info = False
        scaler = Scaler.from_args(
            entrypoints_args=None,
            manager_args=manager_args,
            instance_args=InstanceArgs(migration_backend="rayrpc"),
            engine_args=None,
            launch_args=LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=BackendType.VLLM),
        )
    except ValueError:
        scaler = get_llumnix_actor_handle(LlumnixActor.SCALER)
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
    engine_args = VLLMEngineArgs(
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
        engine_args = VLLMEngineArgs(
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

def init_scaler_with_launch_mode(launch_mode, enable_pd_disagg=False, pd_ratio="1:3", max_units=-1,
                                 enable_pdd_node_affinity_scheduling=False):
    manager_args = ManagerArgs(enable_port_increment=True, enable_pd_disagg=enable_pd_disagg,
                               pd_ratio=pd_ratio, max_units=max_units,
                               enable_pdd_node_affinity_scheduling=enable_pdd_node_affinity_scheduling)
    instance_args = InstanceArgs(migration_backend="rayrpc")
    entrypoints_args = EntrypointsArgs(host="127.0.0.1", port=8000)
    engine_args = VLLMEngineArgs(
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
    manager: Manager = get_llumnix_actor_handle(LlumnixActor.MANAGER)

    return scaler, manager, manager_args, entrypoints_args, engine_args, launch_args

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required")
async def test_init_dp_manager_and_get_unit_deployment_states_and_instance_and_clear_dp_manager_ray_resources(ray_env):
    scaler, _, _, _, engine_args, _ = init_scaler_with_launch_mode(LaunchMode.LOCAL)
    unit_id = random_uuid()
    pg = ray.get(
        scaler._init_placement_group.remote(
            get_placement_group_name(unit_id), engine_args, init_server=True
        )
    )
    pg = ray.util.get_placement_group(get_placement_group_name(unit_id))
    ray.get(pg.ready())
    dp_manager = ray.get(
        scaler._init_dp_manager.remote(
            unit_id, EntrypointsArgs(), InstanceArgs(), engine_args, pg, InstanceType("neutral"), BackendType.VLLM
        )
    )

    # wait for scale up
    await asyncio.sleep(10.0)
    num_units = ray.get(scaler._scale_up.remote(unit_id, dp_manager, InstanceType("neutral")))
    assert num_units == 1

    pg_created, dp_manager_alive = ray.get(scaler._get_unit_deployment_states.remote(unit_id))
    assert pg_created and dp_manager_alive

    ray.get(scaler._clear_unit_ray_resources.remote(unit_id))
    # wait for remove and kill
    await asyncio.sleep(5.0)

    pg_created, dp_manager_alive = ray.get(scaler._get_unit_deployment_states.remote(unit_id))
    assert not pg_created and not dp_manager_alive

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_auto_scale_up_loop_and_get_cluster_deployment_states(ray_env):
    scaler, _, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.GLOBAL, max_units=4)
    await asyncio.sleep(30.0)

    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 4
    curr_pg_uids, curr_dp_manager_uids = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pg_uids) == 4 and len(curr_dp_manager_uids) == 4

    dp_manager_names = list_actor_names_by_actor_type(LlumnixActor.DP_MANAGER)
    unit_ids = [get_llumnix_actor_id(LlumnixActor.DP_MANAGER, dp_manager_name) for dp_manager_name in dp_manager_names]
    assert len(unit_ids) == 4
    ray.get(scaler._clear_unit_ray_resources.remote(unit_ids[0]))
    ray.get(scaler._clear_unit_ray_resources.remote(unit_ids[1]))
    await asyncio.sleep(60.0)

    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 4
    curr_pg_uids, curr_dp_manager_uids = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pg_uids) == 4 and len(curr_dp_manager_uids) == 4

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_check_deployment_states_loop_and_auto_scale_up_loop(ray_env):
    scaler, _, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.GLOBAL, max_units=4)
    await asyncio.sleep(30.0)

    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 4
    curr_pg_uids, curr_dp_manager_uids = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pg_uids) == 4 and len(curr_dp_manager_uids) == 4

    dp_manager_names = list_actor_names_by_actor_type(LlumnixActor.DP_MANAGER)
    unit_ids = [get_llumnix_actor_id(LlumnixActor.DP_MANAGER, dp_manager_name) for dp_manager_name in dp_manager_names]
    assert len(unit_ids) == 4
    remove_placement_group(unit_ids[0])
    await kill_dp_manager(unit_ids[1])
    # Wait for check deployment states, scale down instance and auto scale up.
    await asyncio.sleep(60.0)

    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 4
    curr_pg_uids, curr_dp_manager_uids = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pg_uids) == 4 and len(curr_dp_manager_uids) == 4

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
    launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=BackendType.VLLM)
    # Scaler will get scaler actor handle inside the constructor.
    _ = DummyScaler.options(name=get_llumnix_actor_name(LlumnixActor.SCALER)).remote()
    scaler = Scaler(EntrypointsArgs(), manager_args, InstanceArgs(), EngineArgs(), launch_args)

    scaler.prefill_unit_id_set = set()
    scaler.decode_unit_id_set = set()

    assert scaler._get_next_instance_type() == InstanceType.PREFILL
    scaler.inflight_num_prefill_units += 1

    assert scaler._get_next_instance_type() == InstanceType.DECODE
    scaler.inflight_num_decode_units += 1

    scaler.inflight_num_prefill_units = 0
    scaler.inflight_num_decode_units = 0
    scaler.prefill_unit_id_set.add(random_uuid())
    scaler.decode_unit_id_set.add(random_uuid())
    assert scaler._get_next_instance_type() == InstanceType.DECODE
    scaler.decode_unit_id_set.add(random_uuid())
    assert scaler._get_next_instance_type() == InstanceType.PREFILL

    scaler.prefill_unit_id_set.update([random_uuid()] * 2)
    scaler.decode_unit_id_set.update([random_uuid()] * 2)
    assert scaler._get_next_instance_type() == InstanceType.DECODE
    scaler.decode_unit_id_set.add(random_uuid())
    assert scaler._get_next_instance_type() == InstanceType.PREFILL
    scaler.decode_unit_id_set.add(random_uuid())
    assert scaler._get_next_instance_type() == InstanceType.PREFILL

@pytest.mark.parametrize("load_registered_service", [False, True])
@pytest.mark.parametrize("enable_pd_disagg", [False, True])
def test_load_registered_service(ray_env, load_registered_service, enable_pd_disagg):
    engine_args = VLLMEngineArgs(engine_args=EngineArgs(model="neutral"))
    save_path = 'test'
    save_key = "test"
    load_registered_service_path = os.path.join(save_path, save_key)
    instance_type_list = ['neutral']
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
    _ = DummyScaler.options(name=get_llumnix_actor_name(LlumnixActor.SCALER)).remote()
    scaler = Scaler(None, manager_args, None, None, LaunchArgs(backend_type=BackendType.VLLM, launch_mode=LaunchMode.LOCAL))
    for instance_type in instance_type_list:
        get_engine_args = scaler.llumnix_engine_args_factory.gen_next_engine_args(
            engine_args, InstanceArgs(instance_type=instance_type), 0)
        if load_registered_service:
            assert get_engine_args.load_engine_args().model == instance_type
        else:
            assert get_engine_args.load_engine_args().model == 'neutral'
    if load_registered_service:
        shutil.rmtree(load_registered_service_path)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_pd_disagg_gloal_launch_deployment_and_auto_scale_up_loop(ray_env):
    scaler, _, _, _, _, _ = init_scaler_with_launch_mode(
        LaunchMode.GLOBAL, max_units=4, enable_pd_disagg=True, pd_ratio="1:1")
    await asyncio.sleep(30.0)
    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 4
    curr_pg_uids, curr_dp_manager_uids = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pg_uids) == 4 and len(curr_dp_manager_uids) == 4

    num_prefill_units = 0
    num_decode_units = 0
    prefill_unit_ids = []
    decode_unit_ids = []
    for unit_id in curr_dp_manager_uids:
        dp_manager = get_llumnix_actor_handle(LlumnixActor.DP_MANAGER, unit_id)
        instance_type = ray.get(dp_manager.get_instance_type.remote())
        if instance_type == InstanceType.PREFILL:
            num_prefill_units += 1
            prefill_unit_ids.append(unit_id)
        elif instance_type == InstanceType.DECODE:
            num_decode_units += 1
            decode_unit_ids.append(unit_id)

    assert num_prefill_units == 2 and num_decode_units == 2
    assert set(prefill_unit_ids).union(set(decode_unit_ids)) == set(curr_dp_manager_uids)

    ray.get(scaler._scale_down.remote(prefill_unit_ids[0]))
    await asyncio.sleep(10.0)

    ray.get(scaler._scale_down.remote(prefill_unit_ids[1]))
    await asyncio.sleep(10.0)

    ray.get(scaler._scale_down.remote(decode_unit_ids[1]))
    await asyncio.sleep(60.0)
    alive_decode_unit_id = decode_unit_ids[0]

    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 4
    curr_pg_uids, curr_dp_manager_uids = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pg_uids) == 4 and len(curr_dp_manager_uids) == 4

    num_prefill_units = 0
    num_decode_units = 0
    decode_unit_ids = []
    for unit_id in curr_dp_manager_uids:
        dp_manager = get_llumnix_actor_handle(LlumnixActor.DP_MANAGER, unit_id)
        instance_type = ray.get(dp_manager.get_instance_type.remote())
        if instance_type == InstanceType.PREFILL:
            num_prefill_units += 1
        elif instance_type == InstanceType.DECODE:
            num_decode_units += 1
            decode_unit_ids.append(unit_id)

    assert num_prefill_units == 2 and num_decode_units == 2
    assert alive_decode_unit_id in decode_unit_ids

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_pd_disagg_deployment_states(ray_env):
    manager_args = ManagerArgs(enable_routine_migration=True, enable_pre_stop_migration=True, enable_pd_disagg=True, pd_ratio="1:2")
    engine_args = VLLMEngineArgs(
        engine_args=EngineArgs(
            model=try_convert_to_local_path("facebook/opt-125m"),
            download_dir="/mnt/model",
            worker_use_ray=True,
            enforce_eager=True,
        )
    )
    scaler: Scaler = Scaler.from_args(
        entrypoints_args=EntrypointsArgs(),
        manager_args=manager_args,
        instance_args=InstanceArgs(migration_backend="rayrpc"),
        engine_args=engine_args,
        launch_args=LaunchArgs(LaunchMode.LOCAL, BackendType.VLLM),
    )
    ray.get(scaler.is_ready.remote())
    manager: Manager = get_llumnix_actor_handle(LlumnixActor.MANAGER)
    ray.get(manager.is_ready.remote())
    assert not ray.get(scaler._check_pd_deployment_states.remote())

    prefill_instance_ids, prefill_instances = init_instances(3)
    decode_instance_ids, decode_instances = init_instances(3)

    ray.get(
        scaler._scale_up.remote(
            prefill_instance_ids, prefill_instances, [InstanceType("prefill")]*len(prefill_instance_ids)
        )
    )
    assert ray.get(scaler._check_pd_deployment_states.remote())
    ray.get(scaler._scale_down.remote(prefill_instance_ids))
    ray.get(
        scaler._scale_up.remote(
            decode_instance_ids, decode_instances, [InstanceType("decode")]*len(decode_instance_ids)
        )
    )
    assert ray.get(scaler._check_pd_deployment_states.remote())

    prefill_instance_ids2, prefill_instances2 = init_instances(3)
    ray.get(
        scaler._scale_up.remote(
            prefill_instance_ids2, prefill_instances2, [InstanceType("prefill")]*len(prefill_instance_ids2)
        )
    )
    assert not ray.get(scaler._check_pd_deployment_states.remote())

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_auto_scale_up_loop_max_units(ray_env):
    scaler, _, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.GLOBAL, "rayqueue", max_units=2)
    await asyncio.sleep(30.0)
    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 2

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

    scaler, _, _, _, _, _ = init_scaler_with_launch_mode(
        LaunchMode.GLOBAL,
        max_units=4,
        enable_pd_disagg=True,
        enable_pdd_node_affinity_scheduling=True
    )
    await asyncio.sleep(30.0)

    curr_pg_uids, curr_dp_manager_uids = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pg_uids) == 4 and len(curr_dp_manager_uids) == 4

    num_prefill_units = 0
    num_decode_units = 0
    num_neutral_units = 0
    for unit_id in curr_dp_manager_uids:
        dp_manager = get_llumnix_actor_handle(LlumnixActor.DP_MANAGER, unit_id)
        instance_type = ray.get(dp_manager.get_instance_type.remote())
        if instance_type == InstanceType.PREFILL:
            num_prefill_units += 1
        elif instance_type == InstanceType.DECODE:
            num_decode_units += 1
        else:
            num_neutral_units += 1

    assert num_prefill_units == 2 and num_decode_units == 2 and num_neutral_units == 0

    cleanup_ray_env_func()
    ray_start()

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="at least 2 gpus required")
def test_connect_to_dp_managers(ray_env):
    dp_manager = init_dp_manager(random_uuid())
    ray.get(dp_manager.is_ready.remote())
    scaler, _, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.LOCAL)
    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 1

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required")
async def test_heartbeat_loop(ray_env):
    scaler, _, _, _, _, _ = init_scaler_with_launch_mode(LaunchMode.GLOBAL, "rayqueue", max_units=1)
    await asyncio.sleep(30.0)
    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 1
    dp_manager_names = list_actor_names_by_actor_type(LlumnixActor.DP_MANAGER)
    unit_ids = [get_llumnix_actor_id(LlumnixActor.DP_MANAGER, dp_manager_name) for dp_manager_name in dp_manager_names]
    assert len(unit_ids) == 1
    await kill_dp_manager(unit_ids[0])
    await asyncio.sleep(60.0)
    num_units = ray.get(scaler._scale_up.remote([], [], []))
    assert num_units == 1
    dp_manager_names = list_actor_names_by_actor_type(LlumnixActor.DP_MANAGER)
    new_unit_ids = [get_llumnix_actor_id(LlumnixActor.DP_MANAGER, dp_manager_name) for dp_manager_name in dp_manager_names]
    assert unit_ids[0] != new_unit_ids[0]
