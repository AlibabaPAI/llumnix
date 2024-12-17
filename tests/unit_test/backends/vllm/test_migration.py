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
import math
from unittest.mock import MagicMock
import pytest
import ray

from vllm import EngineArgs, SamplingParams
from vllm.utils import random_uuid
from vllm.sequence import SequenceStatus

from llumnix.backends.vllm.llm_engine import BackendVLLM
from llumnix.llumlet.llumlet import Llumlet
from llumnix.backends.utils import BackendType
from llumnix.llumlet.request import RequestInferenceType, RequestStatus
from llumnix.queue.queue_type import QueueType
from llumnix.arg_utils import InstanceArgs
from llumnix.utils import initialize_placement_group, get_placement_group_name

from tests.unit_test.queue.utils import request_output_queue_server
# pylint: disable=unused-import
from tests.conftest import ray_env

from .test_llm_engine import MockEngine
from .utils import create_dummy_prompt

TEST_PROMPTS = [
    "hello world, ",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.\n",
    "Write a short story about a robot that dreams for the first time.\n",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.\n",
    "Swahili: 'The early bird catches the worm.'\n"
]

def init_llumlet(request_output_queue_type, instance_id, instance_args, engine_args):
    placement_group = initialize_placement_group(get_placement_group_name(instance_id), num_cpus=3, num_gpus=1, detached=True)
    llumlet = Llumlet.from_args(
                instance_id=instance_id,
                instance_args=instance_args,
                placement_group=placement_group,
                request_output_queue_type=request_output_queue_type,
                backend_type=BackendType.VLLM,
                engine_args=engine_args)
    return llumlet

class MockBackendVLLM(BackendVLLM):
    def __init__(self):
        self.engine = MockEngine()

class MockLlumlet(Llumlet):
    def __init__(self):
        self.instance_id = "0"
        self.backend_engine = MockBackendVLLM()

@ray.remote(num_cpus=1, max_concurrency=4)
class MockLlumletDoNotSchedule(Llumlet):
    def __init__(self, *args, **kwargs):
        instance_id = kwargs["instance_id"]
        placement_group = initialize_placement_group(get_placement_group_name(instance_id), num_cpus=3, num_gpus=1, detached=True)
        kwargs["placement_group"] = placement_group
        super().__init__(*args, **kwargs)
        # stop the schedule in engine step loop
        self.backend_engine.engine.scheduler[0].schedule = MagicMock()

        # For some reason, if MockScheduelrOutputs is defined outside, the constructor would raise error.
        class MockScheduelrOutputs:
            def __init__(self):
                self.scheduled_seq_groups = []
                self.ignored_seq_groups = []
                self.num_batched_tokens = 0
                self.preempted = False

            def is_empty(self) -> bool:
                return not self.scheduled_seq_groups

        scheduler_outputs = MockScheduelrOutputs()
        self.backend_engine.engine.scheduler[0].schedule.return_value = ([], scheduler_outputs, False)

        self.step_async = self.backend_engine.engine.step_async

        async def step_async_try_schedule():
            request_outputs, server_infos = await self.step_async()
            for seq_group in self.backend_engine.engine.scheduler[0].waiting:
                seq_group.try_schedule_times += 1
            return request_outputs, server_infos

        self.backend_engine.engine.step_async = step_async_try_schedule

@pytest.mark.parametrize("migration_backend", ['rayrpc', 'gloo', 'nccl'])
@pytest.mark.parametrize("migration_request_status", ['waiting', 'running'])
@pytest.mark.asyncio
async def test_migration_correctness(ray_env, migration_backend, migration_request_status):
    instance_args = InstanceArgs(instance_type="no_constraints")
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    id_rank_map = {"0": 0, "1": 1, "2": 2}
    if migration_request_status == 'running':
        request_migration_policy = "SR"
    elif migration_request_status == 'waiting':
        request_migration_policy = "FCW"

    instance_args = InstanceArgs()
    instance_args.request_migration_policy = request_migration_policy
    instance_args.migration_backend = migration_backend

    request_output_queue_type = QueueType.RAYQUEUE
    que, server_info = request_output_queue_server(request_output_queue_type)
    asyncio.create_task(que.run_server_loop())

    llumlet_0: Llumlet = init_llumlet(request_output_queue_type, "0", instance_args, engine_args)
    llumlet_1: Llumlet = init_llumlet(request_output_queue_type, "1", instance_args, engine_args)

    llumlet_2: Llumlet = MockLlumletDoNotSchedule.options(
        name='instance_2',
        namespace='llumnix').remote(
            instance_id="2",
            instance_args=instance_args,
            request_output_queue_type=request_output_queue_type,
            backend_type=BackendType.VLLM,
            engine_args=engine_args,
        )

    while True:
        res = ray.get([llumlet_0.is_ready.remote(), llumlet_1.is_ready.remote(), llumlet_2.is_ready.remote()])
        if all(res):
            break

    ray.get([llumlet_0.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix"),
             llumlet_1.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix"),
             llumlet_2.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix")])

    # empty instance migrate out
    res = ray.get(llumlet_0.migrate_out.remote("instance_1"))
    assert not res
    res = ray.get(llumlet_2.migrate_out.remote("instance_1"))
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

        if migration_request_status == 'running':
            request_id1 = random_uuid()
            ray.get(llumlet_0.generate.remote(request_id1, server_info, math.inf, prompt, sampling_params))
            # wait prefill done
            while True:
                running_queue = ray.get(llumlet_0.execute_engine_method.remote("get_running_queue"))
                if len(running_queue) > 0 and running_queue[0].inference_type == RequestInferenceType.DECODE:
                    break
            # migrate request
            res = ray.get(llumlet_0.migrate_out.remote("instance_1"))
            assert len(res) == 1
        elif migration_request_status == 'waiting':
            request_id1 = random_uuid()
            ray.get(llumlet_2.generate.remote(request_id1, server_info, math.inf, prompt, sampling_params))
            # wait try schedule done
            while True:
                waiting_queue = ray.get(llumlet_2.execute_engine_method.remote("get_waiting_queue"))
                if len(waiting_queue) > 0 and waiting_queue[0].try_schedule_times >= 1:
                    break
            # migrate request
            res = ray.get(llumlet_2.migrate_out.remote("instance_1"))
            assert len(res) == 1

        request_output_queue = que
        output = None
        finished = False
        while not finished:
            request_outputs = await request_output_queue.get()
            for request_output in request_outputs:
                output = request_output.outputs[0]
                finished = request_output.finished

        assert output.text == origin_output.text
        assert output.cumulative_logprob == origin_output.cumulative_logprob

    for prompt in TEST_PROMPTS:
        await test_correctness(prompt)
    que.cleanup()

@pytest.mark.parametrize("migration_backend", ['rayrpc', 'gloo', 'nccl'])
@pytest.mark.asyncio
async def test_pd_diaggregation_correctness(ray_env, migration_backend):
    instance_args = InstanceArgs(instance_type="no_constraints")
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    id_rank_map = {"0":0, "1":1}

    instance_args = InstanceArgs()
    instance_args.request_migration_policy = "SR"
    instance_args.migration_backend = migration_backend

    request_output_queue_type = QueueType.RAYQUEUE
    que, server_info = request_output_queue_server(request_output_queue_type)
    asyncio.create_task(que.run_server_loop())

    llumlet_0 = init_llumlet(request_output_queue_type, "0", instance_args, engine_args)
    llumlet_1 = init_llumlet(request_output_queue_type, "1", instance_args, engine_args)

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
            res = ray.get(llumlet_0.migrate_out.remote("instance_1"))
            if len(res) == 1:
                break
        request_output_queue = que
        output = None
        finished = False
        while not finished:
            request_outputs = await request_output_queue.get()
            for request_output in request_outputs:
                output = request_output.outputs[0]
                finished = request_output.finished

        assert output.text == origin_output.text
        assert output.cumulative_logprob == origin_output.cumulative_logprob

    for prompt in TEST_PROMPTS:
        await test_correctness(prompt)

    que.cleanup()

def test_clear_migration_states():
    num_gpu_blocks = 8
    block_size = 4
    llumlet = MockLlumlet()
    llumlet.backend_engine.pre_alloc("0", RequestStatus.RUNNING, 0.0, 1, range(4))

    llumlet.clear_migration_states(is_migrate_in=True)
    assert len(llumlet.backend_engine.pre_alloc("0", RequestStatus.RUNNING, 0.0, num_gpu_blocks, range(4*num_gpu_blocks))) == num_gpu_blocks
    _, seq_group = create_dummy_prompt("0",7,block_size,SequenceStatus.RUNNING)
    seq_group.set_status(RequestStatus.RUNNING_MIGRATING)
    llumlet.backend_engine.add_migrating_out_request_last_stage(seq_group)
    llumlet.clear_migration_states(is_migrate_in=False)
    assert len(llumlet.backend_engine.get_running_queue()) == 1
    _, seq_group = create_dummy_prompt("0",7,block_size,SequenceStatus.WAITING)
    seq_group.set_status(RequestStatus.WAITING_MIGRATING)
    llumlet.backend_engine.add_migrating_out_request_last_stage(seq_group)
    llumlet.clear_migration_states(is_migrate_in=False)
    assert len(llumlet.backend_engine.get_waiting_queue()) == 1
