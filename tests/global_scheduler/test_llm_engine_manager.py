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

import ray
import pytest
import numpy as np

from vllm.utils import random_uuid

from llumnix.arg_utils import EngineManagerArgs
from llumnix.llm_engine_manager import LLMEngineManager, MANAGER_ACTOR_NAME
from llumnix.instance_info import InstanceInfo


@ray.remote(num_cpus=1, max_concurrency=4)
class MockLlumlet:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.actor_name = f"instance_{instance_id}"
        self.num_requests = 0
        self.request_id_set = set()
        self.instance_info = None
        self.num_migrate_out = 0

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

    def generate(self, request_id, server_info, *args, **kwargs):
        self.request_id_set.add(request_id)
        self.num_requests = len(self.request_id_set)
        return self.num_requests

    def abort(self, request_id):
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for request_id in request_ids:
            if request_id in self.request_id_set:
                self.request_id_set.remove(request_id)
                self.num_requests = len(self.request_id_set)
        return self.num_requests

    def migrate_out(self, dst_instance_name):
        self.num_migrate_out += 1

    def get_num_migrate_out(self):
        return self.num_migrate_out


def init_manager():
    ray.init(ignore_reinit_error=True, namespace='llumnix')
    try:
        engine_manager_args = EngineManagerArgs()
        engine_manager_args.log_instance_info = False
        engine_manager = LLMEngineManager.from_args(engine_manager_args, None)
    except ValueError:
        engine_manager = ray.get_actor(MANAGER_ACTOR_NAME, namespace='llumnix')
    ray.get(engine_manager.is_ready.remote())
    return engine_manager

def init_llumlets(initial_instances):
    instance_ids = []
    llumlets = []
    for _ in range(initial_instances):
        instance_id = random_uuid()
        instance_name = 'instance_{}'.format(instance_id)
        llumlet = MockLlumlet.options(name=instance_name,
                                      namespace='llumnix').remote(instance_id)
        instance_ids.append(instance_id)
        llumlets.append(llumlet)
    ray.get([llumlet.is_ready.remote() for llumlet in llumlets])
    return instance_ids, llumlets

@pytest.fixture
def engine_manager():
    engine_manager = init_manager()
    ray.get(engine_manager.is_ready.remote())
    yield engine_manager
    ray.kill(engine_manager)
    ray.shutdown()

@pytest.fixture
def llumlet():
    instance_id = random_uuid()
    instance_name = 'instance_{}'.format(instance_id)
    llumlet = MockLlumlet.options(name=instance_name,
                                  namespace='llumnix').remote(instance_id)
    ray.get(llumlet.is_ready.remote())
    return llumlet

def test_init_manager(engine_manager):
    assert engine_manager is not None
    engine_manager_actor_handle = ray.get_actor(MANAGER_ACTOR_NAME, namespace='llumnix')
    assert engine_manager_actor_handle is not None
    assert engine_manager == engine_manager_actor_handle

def test_init_llumlet(llumlet):
    assert llumlet is not None
    ray.get(llumlet.is_ready.remote())

# TODO(s5u13b): Add init_llumlets test.

def test_scale_up_and_down(engine_manager):
    initial_instances = 4
    instance_ids, llumlets = init_llumlets(initial_instances)
    num_instances = ray.get(engine_manager.scale_up.remote(instance_ids, llumlets))
    assert num_instances == initial_instances
    instance_ids_1, llumlets_1 = init_llumlets(initial_instances)
    num_instances = ray.get(engine_manager.scale_down.remote(instance_ids_1))
    assert num_instances == initial_instances
    num_instances = ray.get(engine_manager.scale_up.remote(instance_ids_1, llumlets_1))
    assert num_instances == initial_instances * 2
    num_instances = ray.get(engine_manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances
    num_instances = ray.get(engine_manager.scale_down.remote(instance_ids_1))
    assert num_instances == 0

def test_connect_to_instances():
    initial_instances = 4
    instance_ids, llumlets = init_llumlets(initial_instances)
    ray.get([llumlet.is_ready.remote() for llumlet in llumlets])
    engine_manager = init_manager()
    instance_ids_1, llumlets_1 = init_llumlets(initial_instances)
    num_instances = ray.get(engine_manager.scale_up.remote(instance_ids_1, llumlets_1))
    assert num_instances == initial_instances * 2
    num_instances = ray.get(engine_manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances
    ray.kill(engine_manager)
    ray.shutdown()

def test_generate_and_abort(engine_manager, llumlet):
    instance_id = ray.get(llumlet.get_instance_id.remote())
    ray.get(engine_manager.scale_up.remote(instance_id, [llumlet]))
    request_id = random_uuid()
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 0
    ray.get(engine_manager.generate.remote(request_id, None, None, None))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 1
    ray.get(engine_manager.abort.remote(request_id))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 0
    request_id_1 = random_uuid()
    request_id_2 = random_uuid()
    request_ids = [request_id_1, request_id_2]
    ray.get(engine_manager.abort.remote(request_ids))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 0

def test_get_request_instance():
    _, llumlets = init_llumlets(2)
    llumlet, llumlet_1 = llumlets[0], llumlets[1]
    request_id = random_uuid()
    request_id_1 = random_uuid()
    ray.get(llumlet.generate.remote(request_id, None, None, None))
    ray.get(llumlet_1.generate.remote(request_id_1, None, None, None))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    num_requests_1 = ray.get(llumlet_1.get_num_requests.remote())
    assert num_requests == 1
    assert num_requests_1 == 1
    engine_manager = init_manager()
    ray.get(engine_manager.abort.remote(request_id))
    ray.get(engine_manager.abort.remote(request_id_1))
    num_requests = ray.get(llumlet.get_num_requests.remote())
    num_requests_1 = ray.get(llumlet_1.get_num_requests.remote())
    assert num_requests == 0
    assert num_requests_1 == 0
    ray.kill(engine_manager)
    ray.shutdown()

def get_instance_info_migrate_in(instance_id):
    instance_info = InstanceInfo()
    instance_info.instance_id = instance_id
    instance_info.num_available_gpu_blocks = np.inf
    instance_info.num_running_requests = 1
    instance_info.num_blocks_first_waiting_request = 0
    return instance_info

def get_instance_info_migrate_out(instance_id):
    instance_info = InstanceInfo()
    instance_info.instance_id = instance_id
    instance_info.num_available_gpu_blocks = 0
    instance_info.num_running_requests = 1
    instance_info.num_blocks_first_waiting_request = np.inf
    return instance_info

def test_update_instance_info_loop_and_migrate(engine_manager):
    instance_ids, llumlets = init_llumlets(2)
    instance_id, instance_id_1 = instance_ids[0], instance_ids[1]
    llumlet, llumlet_1 = llumlets[0], llumlets[1]
    request_id = random_uuid()
    request_id_1 = random_uuid()
    ray.get(llumlet.generate.remote(request_id, None, None, None))
    ray.get(llumlet_1.generate.remote(request_id_1, None, None, None))
    instance_info_migrate_out = get_instance_info_migrate_out(instance_id)
    instance_info_migrate_in = get_instance_info_migrate_in(instance_id_1)
    ray.get(llumlet.set_instance_info.remote(instance_info_migrate_out))
    ray.get(llumlet_1.set_instance_info.remote(instance_info_migrate_in))
    num_migrate_out = ray.get(llumlet.get_num_migrate_out.remote())
    assert num_migrate_out == 0
    ray.get(engine_manager.scale_up.remote(instance_ids, llumlets))
    time.sleep(0.2)
    num_migrate_out = ray.get(llumlet.get_num_migrate_out.remote())
    assert num_migrate_out != 0
