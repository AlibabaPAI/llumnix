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
import torch
import pytest

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from vllm.engine.arg_utils import EngineArgs

from llumnix.backends.backend_interface import BackendType
from llumnix.llumlet.llumlet import Llumlet
from llumnix.internal_config import MigrationConfig
from llumnix.queue.queue_type import QueueType
# pylint: disable=unused-import
from tests.conftest import setup_ray_env

@ray.remote(num_cpus=1, max_concurrency=4)
class MockLlumlet(Llumlet):
    def set_error_step(self):
        async def raise_error_step():
            await self.origin_step()
            raise ValueError("Mock engine step error")

        self.backend_engine.engine.step_async = raise_error_step

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need at least 1 GPU to run the test.")
def test_engine_step_exception(setup_ray_env):
    engine_args = EngineArgs(model="facebook/opt-125m", max_model_len=8, worker_use_ray=True)
    migration_config = MigrationConfig("SR", "rpc", 16, 1, 4, 5, 20, 2)
    node_id = ray.get_runtime_context().get_node_id()
    scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    origin_free_memory, _ = torch.cuda.mem_get_info()

    actor_name = "instance_0"
    llumlet = MockLlumlet.options(name=actor_name, namespace='llumnix',
                                  scheduling_strategy=scheduling_strategy).remote(
        output_queue_type=QueueType.RAYQUEUE,
        instance_id="0",
        backend_type=BackendType.VLLM,
        migration_config=migration_config,
        engine_args=engine_args,
        node_id=node_id
    )
    ray.get(llumlet.is_ready.remote())

    all_actors = ray.util.list_named_actors(True)
    all_actor_names = [actor["name"] for actor in all_actors]
    assert actor_name in all_actor_names

    cur_free_memory, _ = torch.cuda.mem_get_info()
    assert cur_free_memory < origin_free_memory

    ray.get(llumlet.set_error_step.remote())
    time.sleep(3)

    all_actors = ray.util.list_named_actors(True)
    all_actor_names = [actor["name"] for actor in all_actors]
    assert actor_name not in all_actor_names

    cur_free_memory, _ = torch.cuda.mem_get_info()
    assert origin_free_memory == cur_free_memory
