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
import ray
import pytest
import numpy as np

from vllm import EngineArgs

from llumnix.utils import random_uuid, get_instance_name, MANAGER_NAME
from llumnix.arg_utils import ManagerArgs
from llumnix.manager import Manager
from llumnix.instance_info import InstanceInfo
from llumnix.server_info import ServerInfo
from llumnix.queue.queue_type import QueueType
from llumnix.global_scheduler.scaling_scheduler import InstanceType
from llumnix.backends.vllm.simulator import BackendSimVLLM
from llumnix.backends.backend_interface import BackendType
from llumnix.backends.profiling import LatencyMemData

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
        return self.num_migrate_out

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
        manager_args = ManagerArgs(migration_backend="rayrpc", enable_migration=True)
        manager_args.log_instance_info = False
        manager = Manager.from_args(manager_args)
    except ValueError:
        manager = ray.get_actor(MANAGER_NAME, namespace='llumnix')
    ray.get(manager.is_ready.remote())
    return manager

def init_llumlets(initial_instances):
    instance_ids = []
    llumlets = []
    for _ in range(initial_instances):
        instance_id = random_uuid()
        instance_name = get_instance_name(instance_id)
        llumlet = MockLlumlet.options(name=instance_name,
                                      namespace='llumnix').remote(instance_id)
        instance_ids.append(instance_id)
        llumlets.append(llumlet)
    ray.get([llumlet.is_ready.remote() for llumlet in llumlets])
    return instance_ids, llumlets

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

def test_init_manager(ray_env, manager):
    assert manager is not None
    manager_actor_handle = ray.get_actor(MANAGER_NAME, namespace='llumnix')
    assert manager_actor_handle is not None
    assert manager == manager_actor_handle

def test_init_llumlet(ray_env, llumlet):
    assert llumlet is not None
    ray.get(llumlet.is_ready.remote())

def test_init_llumlets(ray_env, manager):
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    instance_ids, llumlets = ray.get(manager.init_llumlets.remote(engine_args, QueueType("rayqueue"), BackendType.VLLM))
    num_instances = ray.get(manager.scale_up.remote(instance_ids, llumlets))
    manager_args = ManagerArgs()
    assert num_instances == manager_args.initial_instances

def test_init_llumlets_sim(ray_env, manager):
    manager.profiling_result_file_path="//"
    # pylint: disable=import-outside-toplevel
    import llumnix.backends.vllm.simulator
    llumnix.backends.vllm.simulator.BackendSimVLLM = MockBackendSim
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    instance_ids, llumlets = ray.get(manager.init_llumlets.remote(engine_args, QueueType("rayqueue"), BackendType.VLLM))
    num_instances = ray.get(manager.scale_up.remote(instance_ids, llumlets))
    manager_args = ManagerArgs()
    assert num_instances == manager_args.initial_instances

def test_scale_up_and_down(ray_env, manager):
    initial_instances = 4
    instance_ids, llumlets = init_llumlets(initial_instances)
    num_instances = ray.get(manager.scale_up.remote(instance_ids, llumlets))
    assert num_instances == initial_instances
    instance_ids_1, llumlets_1 = init_llumlets(initial_instances)
    num_instances = ray.get(manager.scale_down.remote(instance_ids_1))
    assert num_instances == initial_instances
    num_instances = ray.get(manager.scale_up.remote(instance_ids_1, llumlets_1))
    assert num_instances == initial_instances * 2
    num_instances = ray.get(manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances
    num_instances = ray.get(manager.scale_down.remote(instance_ids_1))
    assert num_instances == 0

def test_connect_to_instances(ray_env):
    initial_instances = 4
    instance_ids, llumlets = init_llumlets(initial_instances)
    ray.get([llumlet.is_ready.remote() for llumlet in llumlets])
    manager = init_manager()
    instance_ids_1, llumlets_1 = init_llumlets(initial_instances)
    num_instances = ray.get(manager.scale_up.remote(instance_ids_1, llumlets_1))
    assert num_instances == initial_instances * 2
    num_instances = ray.get(manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances

def test_generate_and_abort(ray_env, manager, llumlet):
    instance_id = ray.get(llumlet.get_instance_id.remote())
    ray.get(manager.scale_up.remote(instance_id, [llumlet]))
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
    _, llumlets = init_llumlets(2)
    llumlet, llumlet_1 = llumlets[0], llumlets[1]
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
    instance_info = InstanceInfo()
    instance_info.instance_id = instance_id
    instance_info.num_available_gpu_blocks = np.inf
    instance_info.num_running_requests = 1
    instance_info.num_blocks_first_waiting_request = 0
    instance_info.instance_type = InstanceType.NO_CONSTRAINTS
    return instance_info

def get_instance_info_migrate_out(instance_id):
    instance_info = InstanceInfo()
    instance_info.instance_id = instance_id
    instance_info.num_available_gpu_blocks = 0
    instance_info.num_running_requests = 1
    instance_info.num_blocks_first_waiting_request = np.inf
    instance_info.instance_type = InstanceType.NO_CONSTRAINTS
    return instance_info

def test_update_instance_info_loop_and_migrate(ray_env, manager):
    num_llumlets = 5
    instance_ids, llumlets = init_llumlets(num_llumlets)

    for i in range(num_llumlets):
        for _ in range(2*(i+1)):
            ray.get(llumlets[i].generate.remote(random_uuid(), None, math.inf, None, None))

    instance_info = InstanceInfo()
    instance_info.instance_type = InstanceType.NO_CONSTRAINTS

    for i in range(num_llumlets):
        instance_info.instance_id = instance_ids[i]
        instance_info.num_available_gpu_blocks = 40 - i * 10
        instance_info.num_running_requests = i
        instance_info.num_blocks_first_waiting_request = i
        ray.get(llumlets[i].set_instance_info.remote(instance_info))

    for i in range(num_llumlets):
        num_migrate_out = ray.get(llumlets[i].get_num_migrate_out.remote())
        assert num_migrate_out == 0

    ray.get(manager.scale_up.remote(instance_ids, llumlets))
    time.sleep(2)

    for i in range(num_llumlets):
        num_migrate_out = ray.get(llumlets[i].get_num_migrate_out.remote())
        num_migrate_in = ray.get(llumlets[i].get_num_migrate_in.remote())

        if i == 0:
            assert num_migrate_in > 1 and num_migrate_out == 0
        elif i == num_llumlets - 1:
            assert num_migrate_in == 0 and num_migrate_out > 1
        else:
            assert num_migrate_in == 0 and num_migrate_out == 0
