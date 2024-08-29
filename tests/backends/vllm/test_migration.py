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

import pytest
import torch
import ray
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from vllm import EngineArgs, SamplingParams
from vllm.utils import random_uuid

from llumnix.backends.vllm.llm_engine import BackendVLLM
from llumnix.llumlet.llumlet import Llumlet
from llumnix.backends.utils import BackendType
from llumnix.config import MigrationConfig
from llumnix.server_info import ServerInfo

from .test_llm_engine import MockEngine
from .utils import create_dummy_prompt

TEST_PROMPTS = ["hello world, ",
                "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.\n",
                "Write a short story about a robot that dreams for the first time.\n",
                "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.\n",
                "Swahili: 'The early bird catches the worm.'\n"]

class MockBackendVLLM(BackendVLLM):
    def __init__(self):
        self.engine = MockEngine()

class MockLlumlet(Llumlet):
    def __init__(self):
        self.instance_id = "0"
        self.backend_engine = MockBackendVLLM()

@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_migration_correctness():
    ray.init(namespace="llumnix", ignore_reinit_error=True)
    engine_args = EngineArgs(model="facebook/opt-125m",worker_use_ray=True)
    id_rank_map = {"0":0,"1":1}
    migration_config = MigrationConfig("LCFS", "gloo",16,1,4,5,20)
    que = RayQueue(actor_options={
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,)
    })
    server_info = ServerInfo("0",que)

    llumlet_0:Llumlet = Llumlet.from_args(
                            False,
                            True,
                            ray.get_runtime_context().get_node_id(),
                            "0",
                            BackendType.VLLM,
                            1,
                            migration_config,
                            engine_args,)

    llumlet_1:Llumlet = Llumlet.from_args(
                            False,
                            True,
                            ray.get_runtime_context().get_node_id(),
                            "1",
                            BackendType.VLLM,
                            1,
                            migration_config,
                            engine_args,
                     )
    while True:
        res = ray.get([llumlet_0.is_ready.remote(),llumlet_1.is_ready.remote()])
        if all(res):
            break
    ray.get([llumlet_0.execute_engine_method.remote("_run_workers","rebuild_migration_backend", id_rank_map, "llumnix"),
            llumlet_1.execute_engine_method.remote("_run_workers","rebuild_migration_backend", id_rank_map, "llumnix")])
    # empty instance migrate out
    res = ray.get(llumlet_0.migrate_out.remote("instance_1"))
    assert not res

    # running without migration
    def test_correctness(prompt):
        sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
        request_id0 = random_uuid()
        llumlet_0.generate.remote(request_id0, server_info, prompt, sampling_params)
        request_output_queue = que
        origin_output = None
        finished = False
        while not finished:
            qsize = ray.get(request_output_queue.actor.qsize.remote())
            request_outputs = ray.get(request_output_queue.actor.get_nowait_batch.remote(qsize))
            for request_output in request_outputs:
                origin_output = request_output.outputs[0]
                finished = request_output.finished

        request_id1 = random_uuid()
        llumlet_0.generate.remote(request_id1, server_info, prompt, sampling_params)
        # wait prefill done
        while True:
            if ray.get(llumlet_0.execute_engine_method.remote("get_last_running_request")):
                break
        # migrate request
        res = ray.get(llumlet_0.migrate_out.remote("instance_1"))
        assert len(res) == 1
        request_output_queue = que
        output = None
        finished = False
        while not finished:
            qsize = ray.get(request_output_queue.actor.qsize.remote())
            request_outputs = ray.get(request_output_queue.actor.get_nowait_batch.remote(qsize))
            for request_output in request_outputs:
                if request_output.request_id != request_id1:
                    continue
                output = request_output.outputs[0]
                finished = request_output.finished
        assert output.text == origin_output.text
        assert output.cumulative_logprob == origin_output.cumulative_logprob
    for prompt in TEST_PROMPTS:
        test_correctness(prompt)
    ray.shutdown()

def test_clear_migration_states():
    llumlet = MockLlumlet()
    llumlet.backend_engine.pre_alloc("0", 1)
    num_gpu_blocks = 8
    block_size = 4

    llumlet.clear_migration_states(is_migrate_in=True)
    assert len(llumlet.backend_engine.pre_alloc("0", num_gpu_blocks)) == num_gpu_blocks
    _, seq_group = create_dummy_prompt("0",7,block_size)
    llumlet.backend_engine.add_migrating_out_request_last_stage(seq_group)
    llumlet.clear_migration_states(is_migrate_in=False)
    assert llumlet.backend_engine.get_last_running_request() is not None
