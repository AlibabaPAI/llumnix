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
import time
import ray
import torch
import pytest

from vllm.engine.arg_utils import EngineArgs

from llumnix.arg_utils import InstanceArgs
from llumnix.backends.backend_interface import BackendType
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_type import QueueType
from llumnix.utils import initialize_placement_group, get_placement_group_name

# pylint: disable=unused-import
from tests.conftest import ray_env


@ray.remote(num_cpus=1)
class MockLlumlet(Llumlet):
    def __init__(self, *args, **kwargs) -> None:
        instance_id = kwargs["instance_id"]
        placement_group = initialize_placement_group(get_placement_group_name(instance_id), num_cpus=3, num_gpus=1, detached=True)
        kwargs["placement_group"] = placement_group
        super().__init__(*args, **kwargs)
        self.origin_step = self.backend_engine.engine.step_async

    def set_error_step(self, broken: bool):
        self.backend_engine._stop_event.set()

        async def raise_error_step():
            await self.origin_step()
            raise ValueError("Mock engine step error")

        if broken:
            self.backend_engine.engine.step_async = raise_error_step
        else:
            self.backend_engine.engine.step_async = self.origin_step

        asyncio.create_task(self.backend_engine._start_engine_step_loop())

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need at least 1 GPU to run the test.")
def test_engine_step_exception(ray_env):
    engine_args = EngineArgs(model="facebook/opt-125m", max_model_len=8, worker_use_ray=True)

    # wait previous test to release the GPU memory
    time.sleep(5.0)
    device_count = torch.cuda.device_count()
    origin_free_memory_list = []
    for device_id in range(device_count):
        origin_free_memory, _ = torch.cuda.mem_get_info(device_id)
        origin_free_memory_list.append(origin_free_memory)

    actor_name = "instance_0"
    llumlet = MockLlumlet.options(name=actor_name, namespace='llumnix').remote(
        instance_id="0",
        instance_args=InstanceArgs(),
        request_output_queue_type=QueueType.RAYQUEUE,
        backend_type=BackendType.VLLM,
        engine_args=engine_args,
    )
    ray.get(llumlet.is_ready.remote())

    all_actors = ray.util.list_named_actors(True)
    all_actor_names = [actor["name"] for actor in all_actors]
    assert actor_name in all_actor_names

    cur_free_memory_list = []
    for device_id in range(device_count):
        cur_free_memory, _ = torch.cuda.mem_get_info(device_id)
        cur_free_memory_list.append(cur_free_memory)
    assert origin_free_memory_list != cur_free_memory_list

    ray.get(llumlet.set_error_step.remote(True))
    time.sleep(3)

    all_actors = ray.util.list_named_actors(True)
    all_actor_names = [actor["name"] for actor in all_actors]
    assert actor_name not in all_actor_names

    cur_free_memory_list = []
    for device_id in range(device_count):
        cur_free_memory, _ = torch.cuda.mem_get_info(device_id)
        cur_free_memory_list.append(cur_free_memory)
    assert origin_free_memory_list == cur_free_memory_list
