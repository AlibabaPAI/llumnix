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

from typing import List
import asyncio
import math
import pytest
import ray

from vllm import EngineArgs, SamplingParams
from vllm.utils import random_uuid

from llumnix.backends.vllm.llm_engine import BackendVLLM
from llumnix.llumlet.llumlet import Llumlet
from llumnix.backends.utils import BackendType
from llumnix.internal_config import MigrationConfig
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType
from llumnix.queue.queue_type import QueueType

from tests.unit_test.queue.utils import request_output_queue_server
# pylint: disable=unused-import
from tests.conftest import setup_ray_env

from .test_llm_engine import MockEngine
from .utils import create_dummy_prompt

TEST_PROMPTS = [
    "hello world, ",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.\n",
    "Write a short story about a robot that dreams for the first time.\n",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.\n",
    "Swahili: 'The early bird catches the worm.'\n"
]

class MockBackendVLLM(BackendVLLM):
    def __init__(self):
        self.engine = MockEngine()

class MockLlumlet(Llumlet):
    def __init__(self):
        self.instance_id = "0"
        self.backend_engine = MockBackendVLLM()

@pytest.mark.parametrize("migration_backend", ['rpc', 'gloo', 'nccl'])
@pytest.mark.asyncio
async def test_migration_correctness(setup_ray_env, migration_backend):
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    id_rank_map = {"0":0, "1":1}
    migration_config = MigrationConfig("LCFS", migration_backend, 16, 1, 4, 5, 20)

    output_queue_type = QueueType.RAYQUEUE
    que, server_info = request_output_queue_server(output_queue_type)
    asyncio.create_task(que.run_server_loop())

    llumlet_0: Llumlet = Llumlet.from_args(
                            output_queue_type,
                            False,
                            True,
                            ray.get_runtime_context().get_node_id(),
                            "0",
                            BackendType.VLLM,
                            1,
                            migration_config,
                            engine_args)

    llumlet_1: Llumlet = Llumlet.from_args(
                            output_queue_type,
                            False,
                            True,
                            ray.get_runtime_context().get_node_id(),
                            "1",
                            BackendType.VLLM,
                            1,
                            migration_config,
                            engine_args)

    while True:
        res = ray.get([llumlet_0.is_ready.remote(),llumlet_1.is_ready.remote()])
        if all(res):
            break

    ray.get([llumlet_0.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix"),
            llumlet_1.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix")])

    # empty instance migrate out
    res = ray.get(llumlet_0.migrate_out.remote("instance_1", num_requests=math.inf))
    assert not res

    # running without migration
    async def test_correctness(prompt):
        sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
        request_id0 = random_uuid()
        llumlet_0.generate.remote(request_id0, server_info, math.inf, prompt, sampling_params)
        request_output_queue = que
        origin_output = None
        finished = False
        while not finished:
            request_outputs = await request_output_queue.get()
            for request_output in request_outputs:
                origin_output = request_output.outputs[0]
                finished = request_output.finished

        request_id1 = random_uuid()
        ray.get(llumlet_0.generate.remote(request_id1, server_info, math.inf, prompt, sampling_params))
        # wait prefill done
        while True:
            running_queue: List[LlumnixRequest] = ray.get(llumlet_0.execute_engine_method.remote("get_running_queue"))
            if len(running_queue) > 0 and running_queue[0].inference_type == RequestInferenceType.DECODE:
                break
        # migrate request
        res = ray.get(llumlet_0.migrate_out.remote("instance_1", num_requests=math.inf))
        assert len(res) == 1

        request_output_queue = que
        output = None
        finished = False
        while not finished:
            request_outputs = await request_output_queue.get()
            for request_output in request_outputs:
                origin_output = request_output.outputs[0]
                finished = request_output.finished
                if request_output.request_id != request_id1:
                    continue
                output = request_output.outputs[0]
                finished = request_output.finished

        assert output.text == origin_output.text
        assert output.cumulative_logprob == origin_output.cumulative_logprob

    for prompt in TEST_PROMPTS:
        await test_correctness(prompt)
    que.cleanup()

@pytest.mark.parametrize("migration_backend", ['rpc', 'gloo', 'nccl'])
@pytest.mark.asyncio
async def test_pd_diaggregation_correctness(setup_ray_env, migration_backend):
    engine_args = EngineArgs(model="facebook/opt-125m",worker_use_ray=True)
    id_rank_map = {"0":0,"1":1}
    migration_config = MigrationConfig("LCFS", migration_backend, 16, 1, 4, 5, 20)

    output_queue_type = QueueType.RAYQUEUE
    que, server_info = request_output_queue_server(output_queue_type)
    asyncio.create_task(que.run_server_loop())

    llumlet_0:Llumlet = Llumlet.from_args(
                            output_queue_type,
                            False,
                            True,
                            ray.get_runtime_context().get_node_id(),
                            "0",
                            BackendType.VLLM,
                            1,
                            migration_config,
                            engine_args,)

    llumlet_1:Llumlet = Llumlet.from_args(
                            output_queue_type,
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
    res = ray.get(llumlet_0.migrate_out.remote("instance_1", num_requests=math.inf))
    assert not res

    # running without migration
    async def test_correctness(prompt):
        sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
        request_id0 = random_uuid()
        request_expected_steps_id0 = math.inf
        llumlet_0.generate.remote(request_id0, server_info, request_expected_steps_id0, prompt, sampling_params)
        request_output_queue = que
        origin_output = None
        finished = False
        while not finished:
            request_outputs = await request_output_queue.get()
            for request_output in request_outputs:
                origin_output = request_output.outputs[0]
                finished = request_output.finished

        request_id1 = random_uuid()
        request_expected_steps_id1 = 1
        ray.get(llumlet_0.generate.remote(request_id1, server_info, request_expected_steps_id1, prompt, sampling_params))
        # migrate request for decoding
        while True:
            res = ray.get(llumlet_0.migrate_out.remote("instance_1", num_requests = math.inf))
            if len(res) == 1:
                break
        request_output_queue = que
        output = None
        finished = False
        while not finished:
            request_outputs = await request_output_queue.get()
            for request_output in request_outputs:
                origin_output = request_output.outputs[0]
                finished = request_output.finished
                if request_output.request_id != request_id1:
                    continue
            output = request_output.outputs[0]
            finished = request_output.finished

        assert output.text == origin_output.text
        assert output.cumulative_logprob == origin_output.cumulative_logprob
    for prompt in TEST_PROMPTS:
        await test_correctness(prompt)
    que.cleanup()

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
    assert len(llumlet.backend_engine.get_running_queue()) > 0
