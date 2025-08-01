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

from llumnix.arg_utils import ManagerArgs, InstanceArgs, LaunchArgs
from llumnix.manager import Manager
from llumnix.instance_info import InstanceInfo
from llumnix.request_processing_context import RequestProcessingContext
from llumnix.utils import random_uuid, LaunchMode, BackendType, InstanceType, InstanceContext
from llumnix.ray_utils import (
    get_placement_group_name,
    LlumnixActor,
    get_llumnix_actor_name,
    get_llumnix_actor_handle,
    initialize_placement_group
)
from llumnix.scaler import Scaler

# pylint: disable=unused-import
from tests.conftest import ray_env


@ray.remote(num_cpus=1)
class MockLlumlet:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.engine_disagg_inst_id = instance_id
        self.actor_name = get_llumnix_actor_name(LlumnixActor.INSTANCE, instance_id)
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
        return InstanceType("neutral")

    def get_num_requests(self):
        return self.num_requests

    def generate(self, request_id, server_info, expected_steps, *args, **kwargs):
        self.request_id_set.add(request_id)
        self.num_requests = len(self.request_id_set)

        if 'prefill_instance_id' in kwargs:
            assert self.instance_id == kwargs['prefill_instance_id']

        if 'semi_d_inst_id' in kwargs:
            assert self.instance_id == kwargs['semi_d_inst_id']

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

    def migrate_out(self, dst_instance_actor, dst_instance_id, instance_type):
        self.num_migrate_out += 1
        ray.get(dst_instance_actor.migrate_in.remote(self.actor_name))
        time.sleep(0.1)
        return []

    def migrate_in(self, src_instance_name):
        self.num_migrate_in += 1
        return self.num_migrate_in

    def get_num_migrate_out(self):
        return self.num_migrate_out

    def get_num_migrate_in(self):
        return self.num_migrate_in

    def get_engine_context(self):
        return InstanceContext(local_engine_id=self.instance_id)


def init_manager(
        enable_pd_disagg: bool = False,
        enable_engine_pd_disagg: bool = False,
        enable_engine_semi_pd_disagg: bool = False):
    manager_args = ManagerArgs(
        enable_routine_migration=True,
        enable_pre_stop_migration=False,
        enable_pd_disagg=enable_pd_disagg,
        enable_engine_pd_disagg=enable_engine_pd_disagg,
        enable_engine_semi_pd_disagg=enable_engine_semi_pd_disagg)
    backend_type = BackendType.VLLM if not enable_engine_semi_pd_disagg else BackendType.BLADELLM
    manager_args.log_instance_info = False
    # manager is initialized by scaler
    scaler: Scaler = Scaler.from_args(
        entrypoints_args=None,
        manager_args=manager_args,
        instance_args=InstanceArgs(migration_backend="rayrpc"),
        engine_args=None,
        launch_args=LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=backend_type),
    )
    ray.get(scaler.is_ready.remote())
    manager: Manager = get_llumnix_actor_handle(LlumnixActor.MANAGER)
    ray.get(manager.is_ready.remote())
    return manager

def init_instances(initial_instances):
    instance_ids = []
    instances = []
    for _ in range(initial_instances):
        instance_id = random_uuid()
        # In order to make manager connect to instances sucessfully, we need to create placement group for each instance.
        initialize_placement_group(get_placement_group_name(instance_id), num_cpus=1, num_gpus=0)
        llumlet = MockLlumlet.options(name=get_llumnix_actor_name(LlumnixActor.INSTANCE, instance_id),
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

def generate_llumlet() -> MockLlumlet:
    instance_id = random_uuid()
    instance_name = get_llumnix_actor_name(LlumnixActor.INSTANCE, instance_id)
    llumlet = MockLlumlet.options(name=instance_name,
                                  namespace='llumnix').remote(instance_id)
    ray.get(llumlet.is_ready.remote())
    return llumlet

@pytest.fixture
def llumlet():
    return generate_llumlet()

def test_init_manager(ray_env, manager):
    assert manager is not None
    manager_actor_handle = get_llumnix_actor_handle(LlumnixActor.MANAGER, raise_exc=False)
    assert manager_actor_handle is not None
    assert manager == manager_actor_handle

def test_init_llumlet(ray_env, llumlet):
    assert llumlet is not None
    ray.get(llumlet.is_ready.remote())

def test_scale_up_and_down(ray_env, manager: Manager):
    initial_instances = 4
    instance_ids, instances = init_instances(initial_instances)
    num_instances = ray.get(manager.scale_up.remote(instance_ids, instances, [InstanceType("neutral")]*initial_instances))
    assert num_instances == initial_instances
    instance_ids_1, _ = init_instances(initial_instances)
    num_instances = ray.get(manager.scale_down.remote(instance_ids_1))
    assert num_instances == initial_instances
    instance_ids_2, instances_2 = init_instances(initial_instances)
    num_instances = ray.get(
        manager.scale_up.remote(
            instance_ids_2,
            instances_2,
            [InstanceType("neutral")] * initial_instances,
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
    num_instances = ray.get(manager.scale_up.remote(instance_ids_1, instances_1, [InstanceType("neutral")]*initial_instances))
    assert num_instances == initial_instances * 2
    num_instances = ray.get(manager.scale_down.remote(instance_ids))
    assert num_instances == initial_instances

def test_generate(ray_env, manager: Manager, llumlet):
    instance_id = ray.get(llumlet.get_instance_id.remote())
    ray.get(manager.scale_up.remote(instance_id, llumlet, InstanceType("neutral")))
    request_id = random_uuid()
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 0
    request_processing_context = RequestProcessingContext(None, None, None, None, None)
    ray.get(manager.generate.remote(request_id, request_processing_context, math.inf, None, None))
    time.sleep(1.0)
    num_requests = ray.get(llumlet.get_num_requests.remote())
    assert num_requests == 1

def test_generate_pdd(ray_env):
    manager = init_manager(enable_engine_pd_disagg=True)
    prefill_llumlet: MockLlumlet = generate_llumlet()
    prefill_instance_id = ray.get(prefill_llumlet.get_instance_id.remote())
    ray.get(manager.scale_up.remote(prefill_instance_id, prefill_llumlet, InstanceType("prefill")))
    decode_llumlet: MockLlumlet = generate_llumlet()
    decode_instance_id = ray.get(decode_llumlet.get_instance_id.remote())
    ray.get(manager.scale_up.remote(decode_instance_id, decode_llumlet, InstanceType("decode")))

    request_id = random_uuid()
    num_requests = ray.get(prefill_llumlet.get_num_requests.remote())
    assert num_requests == 0
    request_processing_context = RequestProcessingContext(None, None, None, None, None)
    ray.get(manager.generate.remote(request_id, request_processing_context, math.inf, None, None,
                                    prefill_instance_id=prefill_instance_id))
    time.sleep(1.0)
    num_requests = ray.get(prefill_llumlet.get_num_requests.remote())
    assert num_requests == 1

def test_generate_semi_pdd(ray_env):
    manager = init_manager(enable_engine_semi_pd_disagg=True)
    prefill_llumlet: MockLlumlet = generate_llumlet()
    instance_id = ray.get(prefill_llumlet.get_instance_id.remote())
    ray.get(manager.scale_up.remote(instance_id, prefill_llumlet, InstanceType("prefill")))
    decode_llumlet: MockLlumlet = generate_llumlet()
    instance_id = ray.get(decode_llumlet.get_instance_id.remote())
    ray.get(manager.scale_up.remote(instance_id, decode_llumlet, InstanceType("decode")))

    request_id = random_uuid()
    num_requests = ray.get(decode_llumlet.get_num_requests.remote())
    assert num_requests == 0
    request_processing_context = RequestProcessingContext(None, None, None, None, None)
    ray.get(manager.generate.remote(request_id, request_processing_context, math.inf, None, None))
    time.sleep(1.0)
    num_requests = ray.get(decode_llumlet.get_num_requests.remote())
    assert num_requests == 1

def get_instance_info_migrate_in(instance_id):
    instance_info = InstanceInfo(
        instance_id=instance_id,
        instance_type=InstanceType.NEUTRAL,
        num_total_gpu_blocks=np.inf,
        num_used_gpu_blocks=0,
        num_running_requests=1,
        num_blocks_first_waiting_request=0,
        num_killed_requests=0
    )

    return instance_info

def get_instance_info_migrate_out(instance_id):
    instance_info = InstanceInfo(
        instance_id=instance_id,
        instance_type=InstanceType.NEUTRAL,
        num_total_gpu_blocks=0,
        num_used_gpu_blocks=0,
        num_running_requests=1,
        num_blocks_first_waiting_request=np.inf,
        num_killed_requests=np.inf
    )
    return instance_info

def test_poll_instance_info_loop_and_migrate(ray_env, manager: Manager):
    num_instances = 5
    instance_ids, instances = init_instances(num_instances)

    for i in range(num_instances):
        instance_info = InstanceInfo(
            instance_id=instance_ids[i],
            instance_type=InstanceType.NEUTRAL,
            num_free_gpu_blocks=40-i*10,
            num_running_requests=2*(i+1),
            num_blocks_first_waiting_request=20,
            migration_load_metric=-5+i
        )
        ray.get(instances[i].set_instance_info.remote(instance_info))

    for i in range(num_instances):
        num_migrate_out = ray.get(instances[i].get_num_migrate_out.remote())
        assert num_migrate_out == 0

    ray.get(manager.scale_up.remote(instance_ids, instances, [InstanceType("neutral")]*len(instance_ids)))
    time.sleep(3)

    for i in range(num_instances):
        num_migrate_out = ray.get(instances[i].get_num_migrate_out.remote())
        num_migrate_in = ray.get(instances[i].get_num_migrate_in.remote())
        if i == 0:
            assert num_migrate_in > 1 and num_migrate_out == 0
        elif i == num_instances - 1:
            assert num_migrate_in == 0 and num_migrate_out > 1

@pytest.mark.asyncio
async def test_get_engine_context(ray_env, manager: Manager):
    instance_id = random_uuid()
    env_instance_id = random_uuid()
    llumlet_actor = MockLlumlet.remote(env_instance_id)
    engine_context = await manager._get_engine_context.remote(instance_id, llumlet_actor)
    assert engine_context.local_engine_id == env_instance_id
