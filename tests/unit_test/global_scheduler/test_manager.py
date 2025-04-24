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

import asyncio
import os
import copy
import subprocess
import time
import math
import shutil

import ray
import pytest
import numpy as np
import torch

from vllm import EngineArgs

from llumnix.scaler import Scaler
from llumnix.arg_utils import ManagerArgs, EntrypointsArgs, LaunchArgs, InstanceArgs
from llumnix.manager import Manager
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.server_info import ServerInfo
from llumnix.queue.queue_type import QueueType
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode
from llumnix.utils import random_uuid, try_convert_to_local_path
from llumnix.ray_utils import (get_placement_group_name, get_server_name, get_instance_name,
                               remove_placement_group, INSTANCE_NAME_PREFIX, kill_server,
                               kill_instance, get_manager_name, get_scaler_name,
                               initialize_placement_group)
from llumnix.internal_config import PDDConfig
from llumnix.entrypoints.vllm.register_service import save_engine_args

# pylint: disable=unused-import
from tests.conftest import ray_env, cleanup_ray_env_func, ray_stop, ray_start


@ray.remote(num_cpus=1)
class MockLlumlet:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.engine_disagg_inst_id = instance_id
        self.actor_name = get_instance_name(instance_id)
        self.num_requests = 0
        self.request_id_set = set()
        self.instance_info = None
        self.num_migrate_out = 0
        self.num_migrate_in = 0

    def get_instance_id(self) -> str:
        return self.instance_id

    def set_instance_info(self, instance_info):
        self.instance_info = instance_info

    def get_instance_info(self):
        return self.instance_info

    def is_ready(self) -> bool:
        return True

    def get_instance_type(self) -> InstanceType:
        return InstanceType("no_constraints")

    def get_all_request_ids(self):
        return list(self.request_id_set)

    def get_num_requests(self):
        return self.num_requests

    def generate(self, request_id, server_info, expected_steps, *args, **kwargs):
        self.request_id_set.add(request_id)
        self.num_requests = len(self.request_id_set)
        return self.num_requests

    def abort(self, request_id):
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)

        for req_id in request_ids:
            if req_id in self.request_id_set:
                self.request_id_set.remove(req_id)
                self.num_requests = len(self.request_id_set)
        return self.num_requests

    def migrate_out(self, dst_instance_id, dst_instance_actor_handle):
        self.num_migrate_out += 1
        ray.get(dst_instance_actor_handle.migrate_in.remote(self.actor_name))
        time.sleep(0.1)
        return []

    def migrate_in(self, src_instance_name):
        self.num_migrate_in += 1
        return self.num_migrate_in

    def get_num_migrate_out(self):
        return self.num_migrate_out

    def get_num_migrate_in(self):
        return self.num_migrate_in

    def get_engine_disagg_inst_id(self) -> str:
        return self.engine_disagg_inst_id

def init_manager():
    try:
        manager_args = ManagerArgs(enable_migration=True)
        manager_args.log_instance_info = False
        manager = Manager.from_args(
            entrypoints_args=None,
            manager_args=manager_args,
            instance_args=InstanceArgs(migration_backend="rayrpc"),
            engine_args=None,
            launch_args=None,
        )
    except ValueError:
        manager = ray.get_actor(get_manager_name(), namespace='llumnix')
    ray.get(manager.is_ready.remote())
    return manager

def init_manager_with_launch_mode(launch_mode, enable_pd_disagg=False, pd_ratio="1:3", max_instances=-1,
                                  enable_pdd_node_affinity_scheduling=False):
    manager_args = ManagerArgs(enable_port_increment=True, enable_pd_disagg=enable_pd_disagg,
                               pd_ratio=pd_ratio, max_instances=max_instances,
                               enable_pdd_node_affinity_scheduling=enable_pdd_node_affinity_scheduling)
    instance_args = InstanceArgs(migration_backend="rayrpc")
    entrypoints_args = EntrypointsArgs(host="127.0.0.1", port=8000)
    engine_args = EngineArgs(model=try_convert_to_local_path("facebook/opt-125m"), download_dir="/mnt/model", worker_use_ray=True, enforce_eager=True)
    launch_args = LaunchArgs(launch_mode=launch_mode, backend_type=BackendType.VLLM)

    # As mock_manager can not be initialized to ray actor, it is initialized as local variable.
    # But, some place need to get the manager actor, so create the dummy manager actor here.
    manager = Manager.from_args(entrypoints_args=entrypoints_args, manager_args=manager_args,
                                instance_args=instance_args, engine_args=engine_args,
                                launch_args=launch_args)
    ray.get(manager.is_ready.remote())
    scaler = ray.get_actor(get_scaler_name(), namespace='llumnix')

    return manager, scaler, manager_args, entrypoints_args, engine_args, launch_args

def init_instances(initial_instances):
    instance_ids = []
    instances = []
    for _ in range(initial_instances):
        instance_id = random_uuid()
        # In order to make manager connect to instances sucessfully, we need to create placement group for each instance.
        initialize_placement_group(get_placement_group_name(instance_id), num_cpus=1, num_gpus=0)
        llumlet = MockLlumlet.options(name=get_instance_name(instance_id),
                                      namespace='llumnix').remote(instance_id)
        instance_ids.append(instance_id)
        instances.append(llumlet)
    ray.get([instance.is_ready.remote() for instance in instances])
    return instance_ids, instances

@pytest.fixture
def manager():
    manager = init_manager()
    ray.get(manager.is_ready.remote())
    yield manager

@pytest.fixture
def llumlet():
    instance_id = random_uuid()
    instance_name = get_instance_name(instance_id)
    llumlet = MockLlumlet.options(name=instance_name,
                                  namespace='llumnix').remote(instance_id)
    ray.get(llumlet.is_ready.remote())
    return llumlet

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

def test_init_manager(ray_env, manager):
    assert manager is not None
    manager_actor_handle = ray.get_actor(get_manager_name(), namespace='llumnix')
    assert manager_actor_handle is not None
    assert manager == manager_actor_handle

def test_init_llumlet(ray_env, llumlet):
    assert llumlet is not None
    ray.get(llumlet.is_ready.remote())

def test_init_instances(ray_env, manager):
    engine_args = EngineArgs(model=try_convert_to_local_path("facebook/opt-125m"), download_dir="/mnt/model", worker_use_ray=True, enforce_eager=True)
    node_id = ray.get_runtime_context().get_node_id()
    _, instances = ray.get(manager.init_instances.remote(QueueType("rayqueue"), BackendType.VLLM, InstanceArgs(), engine_args,
                                                         node_id))
    num_instances = len(instances)
    manager_args = ManagerArgs()
    assert num_instances == manager_args.initial_instances

def test_init_instances_sim(ray_env, manager):
    # pylint: disable=import-outside-toplevel
    # cannot catch by pytest.raises
    try:
        engine_args = EngineArgs(model=try_convert_to_local_path("facebook/opt-125m"), download_dir="/mnt/model",
                                 worker_use_ray=True, enforce_eager=True)
        node_id = ray.get_runtime_context().get_node_id()
        _, _ = ray.get(manager.init_instances.remote(QueueType("rayqueue"), BackendType.SIM_VLLM,
                                                     InstanceArgs(profiling_result_file_path="/"), engine_args, node_id))
    # pylint: disable=broad-except
    except Exception as e:
        assert isinstance(e, IsADirectoryError)

def test_scale_up_and_down(ray_env, manager):
    initial_instances = 4
    instance_ids, instances = init_instances(initial_instances)
    num_instances = ray.get(manager.scale_up.remote(instance_ids, instances, [InstanceType("no_constraints")]*initial_instances,
                                                    [None]*initial_instances))
    assert num_instances == initial_instances
    instance_ids_1, _ = init_instances(initial_instances)
    num_instances = ray.get(manager.scale_down.remote(instance_ids_1))
    assert num_instances == initial_instances
    instance_ids_2, instances_2 = init_instances(initial_instances)
    num_instances = ray.get(
        manager.scale_up.remote(
            instance_ids_2,
            instances_2,
            [InstanceType("no_constraints")] * initial_instances,
            [None] * initial_instances,
        )
    )
    assert num_instances == initial_instances * 2
    num_instances = ray.get(manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances
    num_instances = ray.get(manager.scale_down.remote(instance_ids_2))
    assert num_instances == 0

def test_connect_to_instances(ray_env):
    initial_instances = 4
    instance_ids, instances = init_instances(initial_instances)
    ray.get([instance.is_ready.remote() for instance in instances])
    manager = init_manager()
    instance_ids_1, instances_1 = init_instances(initial_instances)
    num_instances = ray.get(manager.scale_up.remote(instance_ids_1, instances_1, [InstanceType("no_constraints")]*initial_instances,
                                                    [None]*initial_instances))
    assert num_instances == initial_instances * 2
    num_instances = ray.get(manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances

def test_generate_and_abort(ray_env, manager, llumlet):
    instance_id = ray.get(llumlet.get_instance_id.remote())
    ray.get(manager.scale_up.remote(instance_id, llumlet, InstanceType("no_constraints"), [None]))
    request_id = random_uuid()
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 0
    server_info = ServerInfo(None, None, None, None, None)
    ray.get(manager.generate.remote(request_id, server_info, math.inf, None, None))
    time.sleep(1.0)
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 1
    ray.get(manager.abort.remote(request_id))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 0
    request_id_1 = random_uuid()
    request_id_2 = random_uuid()
    request_ids = [request_id_1, request_id_2]
    ray.get(manager.abort.remote(request_ids))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 0

def test_get_request_instance(ray_env):
    _, instances = init_instances(2)
    llumlet, llumlet_1 = instances[0], instances[1]
    manager = init_manager()
    request_id = random_uuid()
    request_id_1 = random_uuid()
    ray.get(manager.generate.remote(request_id, None, math.inf, None, None))
    ray.get(manager.generate.remote(request_id_1, None, math.inf, None, None))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    num_requests_1 = ray.get(llumlet_1.get_num_requests.remote())
    assert num_requests + num_requests_1 == 2
    ray.get(manager.abort.remote(request_id))
    ray.get(manager.abort.remote(request_id_1))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    num_requests_1 = ray.get(llumlet_1.get_num_requests.remote())
    assert num_requests == 0
    assert num_requests_1 == 0

def get_instance_info_migrate_in(instance_id):
    instance_info = InstanceInfo(
        instance_id=instance_id,
        instance_type=InstanceType.NO_CONSTRAINTS,
        num_available_gpu_blocks=np.inf,
        num_running_requests=1,
        num_blocks_first_waiting_request=0,
        num_killed_requests=0
    )

    return instance_info

def get_instance_info_migrate_out(instance_id):
    instance_info = InstanceInfo(
        instance_id=instance_id,
        instance_type=InstanceType.NO_CONSTRAINTS,
        num_available_gpu_blocks=0,
        num_running_requests=1,
        num_blocks_first_waiting_request=np.inf,
        num_killed_requests=np.inf
    )
    return instance_info

def test_poll_instance_info_loop_and_migrate(ray_env, manager):
    num_instances = 5
    instance_ids, instances = init_instances(num_instances)

    for i in range(num_instances):
        instance_info = InstanceInfo(
            instance_id=instance_ids[i],
            instance_type=InstanceType.NO_CONSTRAINTS,
            num_free_gpu_blocks=40-i*10,
            num_running_requests=2*(i+1),
            num_blocks_first_waiting_request=20,
            migration_load_metric=-5+i
        )
        ray.get(instances[i].set_instance_info.remote(instance_info))

    for i in range(num_instances):
        num_migrate_out = ray.get(instances[i].get_num_migrate_out.remote())
        assert num_migrate_out == 0

    ray.get(manager.scale_up.remote(instance_ids, instances, [InstanceType("no_constraints")]*len(instance_ids), [None]*len(instance_ids)))
    time.sleep(3)

    for i in range(num_instances):
        num_migrate_out = ray.get(instances[i].get_num_migrate_out.remote())
        num_migrate_in = ray.get(instances[i].get_num_migrate_in.remote())
        if i == 0:
            assert num_migrate_in > 1 and num_migrate_out == 0
        elif i == num_instances - 1:
            assert num_migrate_in == 0 and num_migrate_out > 1

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required")
async def test_init_server_and_get_instance_deployment_states_and_instance_and_clear_instance_ray_resources(ray_env):
    manager, scaler, _, _, engine_args, _ = init_manager_with_launch_mode(LaunchMode.LOCAL)
    instance_id = random_uuid()
    pg = ray.get(scaler._init_placement_group.remote(get_placement_group_name(instance_id),
                                                     engine_args, BackendType.VLLM, init_server=True))
    pg = ray.util.get_placement_group(get_placement_group_name(instance_id))
    ray.get(pg.ready())
    ray.get(scaler._init_server_and_instance.remote(instance_id, EntrypointsArgs(), InstanceArgs(), engine_args, BackendType.VLLM, pg))

    # wait for scale up
    await asyncio.sleep(5.0)
    server = ray.get_actor(get_server_name(instance_id), namespace="llumnix")
    ray.get(server.is_ready.remote())
    instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
    ray.get(instance.is_ready.remote())
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
    manager, scaler, _, _, _, _ = init_manager_with_launch_mode(LaunchMode.GLOBAL, max_instances=4)
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
    manager, scaler, _, _, _, _ = init_manager_with_launch_mode(LaunchMode.GLOBAL, max_instances=4)
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
    kill_server(instance_ids[1])
    kill_instance(instance_ids[2])
    # Wait for check deployment states, scale down instance and auto scale up.
    await asyncio.sleep(90.0)

    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

# Scaler requires to get manager actor.
def test_pd_disagg_gloal_launch_instance_type(ray_env, manager):
    pdd_config = PDDConfig(True, False, [1, 2], False)
    scaler = Scaler(None, ManagerArgs(), None, None, None, True, False, False, None, pdd_config)

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
def test_load_registered_service(ray_env, manager, load_registered_service, enable_pd_disagg):
    engine_args = EngineArgs()
    engine_args.model = 'no_constraints'
    save_path = 'test'
    save_key = "test"
    load_registered_service_path = os.path.join(save_path, save_key)
    instance_type_list = ['no_constraints']
    if load_registered_service:
        if enable_pd_disagg:
            instance_type_list = ['prefill', 'decode']
        for instance_type in instance_type_list:
            put_engine_args = copy.deepcopy(engine_args)
            put_engine_args.model = instance_type
            save_engine_args(instance_type, save_path, put_engine_args, save_key)
    pdd_config = PDDConfig(enable_pd_disagg, False, [1, 2], False)
    scaler = Scaler(None, ManagerArgs(), None, None, None, True, False, load_registered_service, load_registered_service_path, pdd_config)
    for instance_type in instance_type_list:
        get_engine_args = scaler._get_next_engine_args(engine_args, instance_type)
        if load_registered_service:
            assert get_engine_args.model == instance_type
        else:
            assert get_engine_args.model == 'no_constraints'
    if load_registered_service:
        shutil.rmtree(load_registered_service_path)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required")
async def test_pd_disagg_gloal_launch_deployment_and_auto_scale_up_loop(ray_env):
    manager, scaler, _, _, _, _ = init_manager_with_launch_mode(LaunchMode.GLOBAL, max_instances=4,
                                                                enable_pd_disagg=True, pd_ratio="1:1")
    await asyncio.sleep(60.0)
    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    num_prefill_instances = 0
    num_decode_instances = 0
    prefill_instance_ids = []
    decode_instance_ids = []
    for _, instance_handle in curr_instances.items():
        instance_type = ray.get(instance_handle.get_instance_type.remote())
        if instance_type == InstanceType.PREFILL:
            num_prefill_instances += 1
            prefill_instance_ids.append(ray.get(instance_handle.get_instance_info.remote()).instance_id)
        elif instance_type == InstanceType.DECODE:
            num_decode_instances += 1
            decode_instance_ids.append(ray.get(instance_handle.get_instance_info.remote()).instance_id)

    assert num_prefill_instances == 2 and num_decode_instances == 2
    assert set(prefill_instance_ids).union(set(decode_instance_ids)) == set(curr_instances.keys())

    kill_instance(prefill_instance_ids[0])
    await asyncio.sleep(10.0)

    kill_instance(prefill_instance_ids[1])
    await asyncio.sleep(10.0)

    kill_instance(decode_instance_ids[1])
    await asyncio.sleep(90.0)
    alive_decode_instance_id = decode_instance_ids[0]

    num_instances = ray.get(manager.scale_up.remote([], [], [], []))
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    num_prefill_instances = 0
    num_decode_instances = 0
    decode_instance_ids = []
    for instance_id, instance_handle in curr_instances.items():
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
    engine_args = EngineArgs(model=try_convert_to_local_path("facebook/opt-125m"), download_dir="/mnt/model", worker_use_ray=True, enforce_eager=True)
    manager = Manager.from_args(entrypoints_args=EntrypointsArgs(), manager_args=manager_args,
                                instance_args=InstanceArgs(migration_backend="rayrpc"),
                                engine_args=engine_args, launch_args=LaunchArgs(LaunchMode.LOCAL, BackendType.VLLM))
    ray.get(manager.is_ready.remote())
    scaler = ray.get_actor(get_scaler_name(), namespace="llumnix")
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
    manager, _, _, _, _, _ = init_manager_with_launch_mode(LaunchMode.GLOBAL, "rayqueue", max_instances=2)
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

    _, scaler, _, _, _, _ = init_manager_with_launch_mode(LaunchMode.GLOBAL,
                                                          max_instances=4,
                                                          enable_pd_disagg=True,
                                                          enable_pdd_node_affinity_scheduling=True)
    await asyncio.sleep(60.0)

    curr_pgs, curr_servers, curr_instances = ray.get(scaler._get_cluster_deployment_states.remote())
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    num_prefill_instances = 0
    num_decode_instances = 0
    num_no_constraints_instances = 0
    for instance_handle in curr_instances.values():
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
