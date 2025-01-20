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
import time
import math
import ray
import pytest
import numpy as np

import torch
from vllm import EngineArgs

from llumnix.launcher import Launcher
from llumnix.arg_utils import ManagerArgs, EntrypointsArgs, LaunchArgs, InstanceArgs
from llumnix.manager import Manager
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator
from llumnix.server_info import ServerInfo
from llumnix.queue.queue_type import QueueType
from llumnix.instance_info import InstanceType
from llumnix.backends.vllm.simulator import BackendSimVLLM
from llumnix.backends.backend_interface import BackendType
from llumnix.backends.profiling import LatencyMemData
from llumnix.entrypoints.utils import LaunchMode
from llumnix.utils import (get_placement_group_name, get_server_name, get_instance_name,
                           remove_placement_group, INSTANCE_NAME_PREFIX, kill_server,
                           kill_instance, random_uuid, get_manager_name)

# pylint: disable=unused-import
from tests.conftest import ray_env


@ray.remote(num_cpus=1, max_concurrency=4)
class MockLlumlet:
    def __init__(self, instance_id):
        self.instance_id = instance_id
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

    def get_instance_args(self) -> InstanceArgs:
        return InstanceArgs()

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

    def migrate_out(self, dst_instance_name):
        self.num_migrate_out += 1
        migrate_in_ray_actor = ray.get_actor(dst_instance_name, namespace='llumnix')
        ray.get(migrate_in_ray_actor.migrate_in.remote(self.actor_name))
        time.sleep(0.1)
        return []

    def migrate_in(self, src_instance_name):
        self.num_migrate_in += 1
        return self.num_migrate_in

    def get_num_migrate_out(self):
        return self.num_migrate_out

    def get_num_migrate_in(self):
        return self.num_migrate_in

class MockBackendSim(BackendSimVLLM):
    def _get_lantecy_mem(self, *args, **kwargs):
        latency_mem = LatencyMemData({}, {}, {})
        latency_mem.prefill_model_params = (0,0)
        latency_mem.decode_model_params = (0,0,0)
        return latency_mem

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

class MockManager(Manager):
    async def init_placement_group(self, *args, **kwargs):
        return self.launcher.init_placement_group(*args, **kwargs)

    async def init_server_and_instance(self, *args, **kwargs):
        return self.launcher.init_server_and_instance(*args, **kwargs)

    async def clear_instance_ray_resources(self, instance_id: str):
        return self.launcher.clear_instance_ray_resources(instance_id)

    async def get_cluster_deployment(self):
        return self.launcher.get_cluster_deployment()

    async def get_instance_deployment_states(self, instance_id: str):
        return self.launcher.get_instance_deployment_states(instance_id)

def init_manager_with_launch_mode(launch_mode, request_output_queue_type="rayqueue",
                                  enable_pd_disagg=False, pd_ratio="1:3"):
    manager_args = ManagerArgs(enable_port_increment=True, enable_port_offset_store=True,
                               enable_pd_disagg=enable_pd_disagg, pd_ratio=pd_ratio)
    instance_args = InstanceArgs(migration_backend="rayrpc")
    entrypoints_args = EntrypointsArgs(host="127.0.0.1", port=8000, request_output_queue_type=request_output_queue_type)
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    launch_args = LaunchArgs(launch_mode=launch_mode, backend_type=BackendType.VLLM)

    # As mock_manager can not be initialized to ray actor, it is initialized as local variable.
    # But, some place need to get the manager actor, so create the dummy manager actor here.
    dummy_manager_actor = init_manager()
    ray.get(dummy_manager_actor.is_ready.remote())
    manager = MockManager(entrypoints_args=entrypoints_args, manager_args=manager_args,
                      instance_args=instance_args, engine_args=engine_args,
                      launch_args=launch_args, work_dir=os.getcwd())

    return manager, manager_args, entrypoints_args, engine_args, launch_args

def init_instances(initial_instances):
    instance_ids = []
    instances = []
    for _ in range(initial_instances):
        instance_id = random_uuid()
        instance_name = get_instance_name(instance_id)
        llumlet = MockLlumlet.options(name=instance_name,
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
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    _, instances = ray.get(manager.init_instances.remote(QueueType("rayqueue"), BackendType.VLLM, InstanceArgs(), engine_args))
    num_instances = len(instances)
    manager_args = ManagerArgs()
    assert num_instances == manager_args.initial_instances

def test_init_instances_sim(ray_env, manager):
    manager.profiling_result_file_path="//"
    # pylint: disable=import-outside-toplevel
    import llumnix.backends.vllm.simulator
    llumnix.backends.vllm.simulator.BackendSimVLLM = MockBackendSim
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    _, instances = ray.get(manager.init_instances.remote(QueueType("rayqueue"), BackendType.SIM_VLLM, InstanceArgs(), engine_args))
    num_instances = len(instances)
    manager_args = ManagerArgs()
    assert num_instances == manager_args.initial_instances

def test_scale_up_and_down(ray_env, manager):
    initial_instances = 4
    instance_ids, instances = init_instances(initial_instances)
    num_instances = ray.get(manager.scale_up.remote(instance_ids, instances, [InstanceArgs()]*initial_instances))
    assert num_instances == initial_instances
    instance_ids_1, instances_1 = init_instances(initial_instances)
    num_instances = ray.get(manager.scale_down.remote(instance_ids_1))
    assert num_instances == initial_instances
    num_instances = ray.get(manager.scale_up.remote(instance_ids_1, instances_1, [InstanceArgs()]*initial_instances))
    assert num_instances == initial_instances * 2
    num_instances = ray.get(manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances
    num_instances = ray.get(manager.scale_down.remote(instance_ids_1))
    assert num_instances == 0

def test_connect_to_instances(ray_env):
    initial_instances = 4
    instance_ids, instances = init_instances(initial_instances)
    ray.get([instance.is_ready.remote() for instance in instances])
    manager = init_manager()
    instance_ids_1, instances_1 = init_instances(initial_instances)
    num_instances = ray.get(manager.scale_up.remote(instance_ids_1, instances_1, [InstanceArgs()]*initial_instances))
    assert num_instances == initial_instances * 2
    num_instances = ray.get(manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances

def test_generate_and_abort(ray_env, manager, llumlet):
    instance_id = ray.get(llumlet.get_instance_id.remote())
    ray.get(manager.scale_up.remote(instance_id, llumlet, InstanceArgs()))
    request_id = random_uuid()
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 0
    server_info = ServerInfo(None, None, None, None, None)
    ray.get(manager.generate.remote(request_id, server_info, math.inf, None, None))
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
        for _ in range(2*(i+1)):
            ray.get(instances[i].generate.remote(random_uuid(), None, math.inf, None, None))

    instance_load_calculator = InstanceLoadCalculator("remaining_steps", "remaining_steps", True)
    for i in range(num_instances):
        instance_info = InstanceInfo(
            instance_id=instance_ids[i],
            instance_type=InstanceType.NO_CONSTRAINTS,
            num_free_gpu_blocks=40-i*10,
            num_running_requests=i+1,
            num_blocks_first_waiting_request=i,
        )
        instance_load_calculator.compute_instance_load(instance_info)
        ray.get(instances[i].set_instance_info.remote(instance_info))

    for i in range(num_instances):
        num_migrate_out = ray.get(instances[i].get_num_migrate_out.remote())
        assert num_migrate_out == 0

    ray.get(manager.scale_up.remote(instance_ids, instances, [InstanceArgs()]*len(instance_ids)))
    time.sleep(3)

    for i in range(num_instances):
        num_migrate_out = ray.get(instances[i].get_num_migrate_out.remote())
        num_migrate_in = ray.get(instances[i].get_num_migrate_in.remote())
        if i == 0:
            assert num_migrate_in > 1 and num_migrate_out == 0
        elif i == num_instances - 1:
            assert num_migrate_in == 0 and num_migrate_out > 1

@pytest.mark.asyncio
async def test_init_server_and_get_instance_deployment_states_and_instance_and_clear_instance_ray_resources(ray_env):
    manager, _, _, engine_args, _ = init_manager_with_launch_mode(LaunchMode.LOCAL)
    instance_id = random_uuid()
    pg = await manager.init_placement_group(get_placement_group_name(instance_id),
                                                      engine_args, BackendType.VLLM, init_server=True)
    pg = ray.util.get_placement_group(get_placement_group_name(instance_id))
    ray.get(pg.ready())
    await manager.init_server_and_instance(instance_id, EntrypointsArgs(), InstanceArgs(), engine_args, BackendType.VLLM, pg)

    # wait for scale up
    await asyncio.sleep(5.0)
    server = ray.get_actor(get_server_name(instance_id), namespace="llumnix")
    ray.get(server.is_ready.remote())
    instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
    ray.get(instance.is_ready.remote())
    num_instances = manager.scale_up(instance_id, instance, InstanceArgs())
    assert num_instances == 1

    pg_created, server_alive, instance_alive = await manager.get_instance_deployment_states(instance_id)
    assert pg_created and server_alive and instance_alive

    # test clear_instance_ray_resources
    await manager.clear_instance_ray_resources(instance_id)
    # wait for remove and kill
    await asyncio.sleep(5.0)

    pg_exists = is_placement_group_exists(get_placement_group_name(instance_id))
    assert not pg_exists
    server_exists = is_actor_exists(get_server_name(instance_id))
    assert not server_exists
    instance_exists = is_actor_exists(get_instance_name(instance_id))
    assert not instance_exists

    pg_created, server_alive, instance_alive = await manager.get_instance_deployment_states(instance_id)
    assert not pg_created and not server_alive and not instance_alive

@pytest.mark.asyncio
@pytest.mark.parametrize("request_output_queue_type", ['rayqueue', 'zmq'])
async def test_auto_scale_up_loop_and_get_cluster_deployment(ray_env, request_output_queue_type):
    manager, _, _, _, _ = init_manager_with_launch_mode(LaunchMode.GLOBAL, request_output_queue_type)
    await asyncio.sleep(60.0)

    num_instances = manager.scale_up([], [], [])
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = await manager.get_cluster_deployment()
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    actor_names_dict = ray.util.list_named_actors(all_namespaces=True)
    instance_ids = [actor_name_dict['name'].split("_")[-1] for actor_name_dict in actor_names_dict
                    if actor_name_dict['name'].startswith(INSTANCE_NAME_PREFIX)]
    assert len(instance_ids) == 4
    await manager.clear_instance_ray_resources(instance_ids[0])
    await manager.clear_instance_ray_resources(instance_ids[1])
    await asyncio.sleep(60.0)

    num_instances = manager.scale_up([], [], [])
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = await manager.get_cluster_deployment()
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

@pytest.mark.asyncio
@pytest.mark.parametrize("request_output_queue_type", ['rayqueue', 'zmq'])
async def test_check_deployment_states_loop_and_auto_scale_up_loop(ray_env, request_output_queue_type):
    manager, _, _, _, _ = init_manager_with_launch_mode(LaunchMode.GLOBAL, request_output_queue_type)
    await asyncio.sleep(60.0)

    num_instances = manager.scale_up([], [], [])
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = await manager.get_cluster_deployment()
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    actor_names_dict = ray.util.list_named_actors(all_namespaces=True)
    instance_ids = [actor_name_dict['name'].split("_")[-1] for actor_name_dict in actor_names_dict
                    if actor_name_dict['name'].startswith(INSTANCE_NAME_PREFIX)]
    assert len(instance_ids) == 4
    remove_placement_group(instance_ids[0])
    kill_server(instance_ids[1])
    kill_instance(instance_ids[2])
    # Wait for check deployment states, scale down instance and auto scale up.
    await asyncio.sleep(90.0)

    num_instances = manager.scale_up([], [], [])
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = await manager.get_cluster_deployment()
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

def test_pd_disagg_gloal_launch_instance_type():
    launcher = Launcher(None, True, False, True, False, [1, 2])

    assert launcher._get_next_instance_type(0, 0, [1, 2]) == InstanceType.PREFILL
    launcher.inflight_num_prefill += 1

    assert launcher._get_next_instance_type(0, 0, [1, 2]) == InstanceType.DECODE
    launcher.inflight_num_decode += 1

    launcher.inflight_num_prefill = 0
    launcher.inflight_num_decode = 0
    assert launcher._get_next_instance_type(1, 1, [1, 2]) == InstanceType.DECODE
    assert launcher._get_next_instance_type(1, 2, [1, 2]) == InstanceType.PREFILL

    assert launcher._get_next_instance_type(3, 5, [1, 2]) == InstanceType.DECODE
    assert launcher._get_next_instance_type(3, 6, [1, 2]) == InstanceType.PREFILL
    assert launcher._get_next_instance_type(3, 7, [1, 2]) == InstanceType.PREFILL

@pytest.mark.asyncio
@pytest.mark.parametrize("request_output_queue_type", ['rayqueue', 'zmq'])
async def test_pd_disagg_gloal_launch_deployment_and_auto_scale_up_loop(ray_env, request_output_queue_type):
    manager, _, _, _, _ = init_manager_with_launch_mode(LaunchMode.GLOBAL, request_output_queue_type,
                                                        enable_pd_disagg=True, pd_ratio="1:1")
    await asyncio.sleep(60.0)

    num_instances = manager.scale_up([], [], [])
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = await manager.get_cluster_deployment()
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    num_prefill_instances = 0
    num_decode_instances = 0
    prefill_instance_ids = []
    decode_instance_ids = []
    for _, instance_handle in curr_instances.items():
        instance_type = ray.get(instance_handle.get_instance_args.remote()).instance_type
        if instance_type == InstanceType.PREFILL:
            num_prefill_instances += 1
            prefill_instance_ids.append(ray.get(instance_handle.get_instance_info.remote()).instance_id)
        elif instance_type == InstanceType.DECODE:
            num_decode_instances += 1
            decode_instance_ids.append(ray.get(instance_handle.get_instance_info.remote()).instance_id)

    assert torch.cuda.device_count() == 4
    assert num_prefill_instances == 2 and num_decode_instances == 2
    assert set(prefill_instance_ids).union(set(decode_instance_ids)) == set(curr_instances.keys())

    kill_instance(prefill_instance_ids[0])
    await asyncio.sleep(10.0)

    kill_instance(prefill_instance_ids[1])
    await asyncio.sleep(10.0)

    kill_instance(decode_instance_ids[1])
    await asyncio.sleep(90.0)
    alive_decode_instance_id = decode_instance_ids[0]

    num_instances = manager.scale_up([], [], [])
    assert num_instances == 4
    curr_pgs, curr_servers, curr_instances = await manager.get_cluster_deployment()
    assert len(curr_pgs) == 4 and len(curr_servers) == 4 and len(curr_instances) == 4

    num_prefill_instances = 0
    num_decode_instances = 0
    decode_instance_ids = []
    for instance_id, instance_handle in curr_instances.items():
        instance_type = ray.get(instance_handle.get_instance_args.remote()).instance_type
        if instance_type == InstanceType.PREFILL:
            num_prefill_instances += 1
        elif instance_type == InstanceType.DECODE:
            num_decode_instances += 1
            decode_instance_ids.append(instance_id)

    assert num_prefill_instances == 2 and num_decode_instances == 2
    assert alive_decode_instance_id in decode_instance_ids

@pytest.mark.asyncio
async def test_pd_disagg_deployment_states():
    manager_args = ManagerArgs(enable_migration=True, enable_pd_disagg=True, pd_ratio="1:2")
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    manager = Manager(entrypoints_args=EntrypointsArgs(), manager_args=manager_args,
                      instance_args=InstanceArgs(migration_backend="rayrpc"),
                      engine_args=engine_args, launch_args=LaunchArgs(LaunchMode.LOCAL, BackendType.VLLM),
                      work_dir=os.getcwd())
    assert not manager._inner_check_pd_deployment()

    prefill_instance_ids = [random_uuid() for _ in range(3)]
    decode_instance_ids = [random_uuid() for _ in range(3)]

    manager.scale_up(prefill_instance_ids, [None]*len(prefill_instance_ids),
                     [InstanceArgs(instance_type="prefill")]*len(prefill_instance_ids))
    assert manager._inner_check_pd_deployment() in prefill_instance_ids

    manager.scale_down(prefill_instance_ids)
    manager.scale_up(decode_instance_ids, [None]*len(decode_instance_ids),
                     [InstanceArgs(instance_type="decode")]*len(decode_instance_ids))
    assert manager._inner_check_pd_deployment() in decode_instance_ids

    manager.scale_up(prefill_instance_ids, [None]*len(prefill_instance_ids),
                     [InstanceArgs(instance_type="prefill")]*len(prefill_instance_ids))
    assert not manager._inner_check_pd_deployment()
