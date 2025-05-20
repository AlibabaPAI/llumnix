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
import os
from unittest.mock import MagicMock
from typing import List

import pytest
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import EngineArgs, SamplingParams
from vllm.utils import random_uuid
from vllm.sequence import SequenceStatus
from vllm.outputs import RequestOutput

from llumnix.backends.vllm.llm_engine import BackendVLLM
from llumnix.llumlet.llumlet import Llumlet
from llumnix.llumlet.request import RequestInferenceType, RequestStatus
from llumnix.queue.queue_type import QueueType
from llumnix.arg_utils import InstanceArgs
from llumnix.utils import get_llumnix_env_vars
from llumnix.ray_utils import (initialize_placement_group, get_placement_group_name,
                               remove_placement_group, kill_instance)
from llumnix.entrypoints.vllm.arg_utils import VllmEngineArgs
from llumnix.request_output import LlumnixRequestOuput as LlumnixRequestOuputVLLM

from tests.unit_test.queue.utils import request_output_queue_server
# pylint: disable=unused-import
from tests.conftest import ray_env, cleanup_ray_env_func
from tests.utils import try_convert_to_local_path

from .test_llm_engine import MockEngine
from .utils import create_dummy_prompt


TEST_PROMPTS = [
    "hello world, ",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.\n",
    "Write a short story about a robot that dreams for the first time.\n",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.\n",
    "Swahili: 'The early bird catches the worm.'\n"
]

# When directly inheriting llumlet without @ray.remote in subclass, ray will raise async actor error during initialization.
# And class with @ray.remote cannot be inherited, so the definition of class MockLlumletTestMigration and MockLlumletTestMigrationDoNotSchedule
# seems duplicated.
@ray.remote
class MockLlumletTestMigration(Llumlet):
    def get_pre_alloc_cache_dict(self):
        return self.backend_engine.engine.scheduler[0].pre_alloc_cache_dict

    def get_migrating_out_request_last_stage(self):
        return self.backend_engine.engine.scheduler[0].migrating_out_request_last_stage

    def get_running_queue(self):
        return self.backend_engine.get_running_queue()

    def get_waiting_queue(self):
        return self.backend_engine.get_waiting_queue()

    def get_num_free_blocks(self):
        return self.backend_engine.engine.scheduler[0].block_manager.get_num_free_gpu_blocks()

    async def commit_seq_group_metadata_worker(self, request_id):
        # if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
        return await self.backend_engine._run_workers_async("commit_seq_group_metadata", request_id)

    async def pop_migrating_out_seq_group_metadata_worker(self, request_id):
        # if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
        return await self.backend_engine._run_workers_async("pop_migrating_out_seq_group_metadata", request_id)


def init_llumlet(request_output_queue_type, instance_id, instance_args, engine_args, num_gpus=1):
    placement_group = initialize_placement_group(get_placement_group_name(instance_id), num_cpus=3, num_gpus=num_gpus, detached=True)
    llumlet = MockLlumletTestMigration.options(
                num_cpus=1,
                num_gpus=0.5,
                name=f"instance_{instance_id}",
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True
                ),
                namespace='llumnix').remote(
                    instance_id=instance_id,
                    instance_args=instance_args,
                    placement_group=placement_group,
                    request_output_queue_type=request_output_queue_type,
                    engine_args=engine_args
                )
    return llumlet


class MockBackendVLLM(BackendVLLM):
    def __init__(self):
        self.engine = MockEngine()
        self.use_ray_spmd_worker = True


class MockLlumlet(Llumlet):
    def __init__(self):
        self.instance_id = "0"
        self.backend_engine = MockBackendVLLM()


@ray.remote
class MockLlumletTestMigrationDoNotSchedule(Llumlet):
    def __init__(self, placement_group, *args, **kwargs):
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

    def get_pre_alloc_cache_dict(self):
        return self.backend_engine.engine.scheduler[0].pre_alloc_cache_dict

    def get_migrating_out_request_last_stage(self):
        return self.backend_engine.engine.scheduler[0].migrating_out_request_last_stage

    def get_running_queue(self):
        return self.backend_engine.get_running_queue()

    def get_waiting_queue(self):
        return self.backend_engine.get_waiting_queue()

    def get_num_free_blocks(self):
        return self.backend_engine.engine.scheduler[0].block_manager.get_num_free_gpu_blocks()

    async def commit_seq_group_metadata_worker(self, request_id):
        # if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
        return await self.backend_engine._run_workers_async("commit_seq_group_metadata", request_id)

    async def pop_migrating_out_seq_group_metadata_worker(self, request_id):
        # if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
        return await self.backend_engine._run_workers_async("pop_migrating_out_seq_group_metadata", request_id)

@pytest.mark.asyncio
@pytest.mark.parametrize("migration_backend", ['rayrpc', 'gloo', 'nccl'])
@pytest.mark.parametrize("migration_request_status", ['running', 'waiting'])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("disable_async_output_proc", [False, True])
@pytest.mark.parametrize("use_ray_spmd_worker", [True, False])
async def test_migration_correctness(migration_backend, migration_request_status, tensor_parallel_size,
                                     disable_async_output_proc, use_ray_spmd_worker):
    if migration_backend == 'nccl' and tensor_parallel_size == 2:
        pytest.skip("When the migration backend is nccl, Llumnix does not support tensor parallelism.")
    if disable_async_output_proc and migration_backend != "gloo":
        pytest.skip("Only test the gloo migration backend when disable async output processing.")
    if use_ray_spmd_worker and tensor_parallel_size == 2:
        pytest.skip("When using ray spmd worker, ray will raise RayCgraphCapacityExceeded exeception when tensor parallelism is enabled.")
    if use_ray_spmd_worker and not disable_async_output_proc:
        pytest.skip("When using ray spmd worker, async output processing is not supported.")
    if use_ray_spmd_worker and migration_backend != "gloo":
        pytest.skip("Only test the gloo migration backend when using ray spmd worker.")

    if use_ray_spmd_worker:
        os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
        os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "1"
    else:
        os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "0"
        os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "0"

    ray.init(namespace="llumnix", ignore_reinit_error=True, runtime_env={"env_vars": get_llumnix_env_vars()})

    engine_args = VllmEngineArgs(
        engine_args=EngineArgs(
            model=try_convert_to_local_path("facebook/opt-125m"),
            download_dir="/mnt/model",
            worker_use_ray=True,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            disable_async_output_proc=disable_async_output_proc,
        )
    )
    id_rank_map = {"0": 0, "1": 1}
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

    llumlet_0: MockLlumletTestMigration = init_llumlet(request_output_queue_type, "0", instance_args, engine_args, num_gpus=tensor_parallel_size)
    llumlet_1: MockLlumletTestMigration = init_llumlet(request_output_queue_type, "1", instance_args, engine_args, num_gpus=tensor_parallel_size)

    while True:
        res = ray.get([llumlet_0.is_ready.remote(), llumlet_1.is_ready.remote()])
        if all(res):
            break

    num_free_blocks_ori = ray.get(llumlet_0.get_num_free_blocks.remote())

    id_rank_map = {"0": 0, "1": 1}
    ray.get([llumlet_0.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix"),
             llumlet_1.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix")])

    # empty instance migrate out
    res = ray.get(llumlet_0.migrate_out.remote("1", llumlet_1))
    assert not res

    sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
    origin_outputs = []
    # running without migration
    async def gen_origin_outputs(prompt):
        request_id0 = random_uuid()
        llumlet_0.generate.remote(request_id0, server_info, math.inf, prompt, sampling_params)
        request_output_queue = que
        origin_output = None
        finished = False
        while not finished:
            llumnix_responses: List[LlumnixRequestOuputVLLM] = await request_output_queue.get()
            request_outputs: List[RequestOutput] = [llumnix_response.get_engine_output()
                                                    for llumnix_response in llumnix_responses]
            for request_output in request_outputs:
                origin_output = request_output.outputs[0]
                finished = request_output.finished
        origin_outputs.append(origin_output)

    async def test_correctness(prompt, origin_output):
        if migration_request_status == 'running':
            request_id1 = random_uuid()
            ray.get(llumlet_0.generate.remote(request_id1, server_info, math.inf, prompt, sampling_params))
            # wait prefill done
            while True:
                running_queue = ray.get(llumlet_0.execute_engine_method.remote("get_running_queue"))
                if len(running_queue) > 0 and running_queue[0].inference_type == RequestInferenceType.DECODE:
                    break
            # migrate request
            res = ray.get(llumlet_0.migrate_out.remote("1", llumlet_1))
            assert len(res) == 1
        else: # migration_request_status == 'waiting'
            request_id1 = random_uuid()
            ray.get(llumlet_2.generate.remote(request_id1, server_info, math.inf, prompt, sampling_params))
            # wait try schedule done
            while True:
                waiting_queue = ray.get(llumlet_2.execute_engine_method.remote("get_waiting_queue"))
                if len(waiting_queue) > 0 and waiting_queue[0].try_schedule_times >= 1:
                    break
            # migrate request
            res = ray.get(llumlet_2.migrate_out.remote("1", llumlet_1))
            assert len(res) == 1

        request_output_queue = que
        output = None
        finished = False
        while not finished:
            llumnix_responses: List[LlumnixRequestOuputVLLM] = await request_output_queue.get()
            request_outputs: List[RequestOutput] = [llumnix_response.get_engine_output()
                                                    for llumnix_response in llumnix_responses]
            for request_output in request_outputs:
                output = request_output.outputs[0]
                finished = request_output.finished

        # test request output
        assert output.text == origin_output.text
        assert output.cumulative_logprob == origin_output.cumulative_logprob

        # test engine migration states
        if migration_request_status == 'running':
            llumlets = [llumlet_0, llumlet_1]
        else:
            llumlets = [llumlet_1, llumlet_2]
        for llumlet in llumlets:
            running_queue = ray.get(llumlet.get_running_queue.remote())
            waiting_queue = ray.get(llumlet.get_waiting_queue.remote())
            pre_alloc_cache_dict = ray.get(llumlet.get_pre_alloc_cache_dict.remote())
            migrating_out_request_last_stage = ray.get(llumlet.get_migrating_out_request_last_stage.remote())
            num_free_blocks_cur = ray.get(llumlet.get_num_free_blocks.remote())
            assert len(running_queue) == 0 and len(waiting_queue) == 0 and len(pre_alloc_cache_dict) == 0 \
                and len(migrating_out_request_last_stage) == 0 and num_free_blocks_cur == num_free_blocks_ori

        # test worker migration states
        if use_ray_spmd_worker and migration_request_status == 'running':
            llumlets = [llumlet_0, llumlet_1]
            for llumlet in llumlets:
                assert_commit = None
                try:
                    ray.get(llumlet.commit_seq_group_metadata_worker.remote(request_id1))
                    assert_commit = False
                except AssertionError:
                    assert_commit = True
                popped = ray.get(llumlet.pop_migrating_out_seq_group_metadata_worker.remote(request_id1))
                assert assert_commit and popped

    for prompt in TEST_PROMPTS:
        await gen_origin_outputs(prompt)

    if migration_request_status == 'waiting':
        await kill_instance("0")
        remove_placement_group("0")
        if use_ray_spmd_worker:
            num_gpus = 0.5
        else:
            num_gpus = 0
        placement_group = initialize_placement_group(get_placement_group_name("2"), num_cpus=3, num_gpus=tensor_parallel_size, detached=True)
        llumlet_2: MockLlumletTestMigration = MockLlumletTestMigrationDoNotSchedule.options(
            num_cpus=1,
            num_gpus=num_gpus,
            name='instance_2',
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True
            ),
            namespace='llumnix').remote(
                instance_id="2",
                instance_args=instance_args,
                request_output_queue_type=request_output_queue_type,
                engine_args=engine_args,
                placement_group=placement_group
            )
        while True:
            res = ray.get(llumlet_2.is_ready.remote())
            if res:
                break
        id_rank_map = {"2": 0, "1": 1}
        ray.get([llumlet_2.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix"),
                    llumlet_1.execute_engine_method.remote("_run_workers", "rebuild_migration_backend", id_rank_map, "llumnix")])
        res = ray.get(llumlet_2.migrate_out.remote("1", llumlet_1))
        assert not res

    for i, prompt in enumerate(TEST_PROMPTS):
        origin_output = origin_outputs[i]
        await test_correctness(prompt, origin_output)
    que.cleanup()

    cleanup_ray_env_func()

@pytest.mark.asyncio
@pytest.mark.parametrize("migration_backend", ['rayrpc', 'gloo', 'nccl'])
@pytest.mark.parametrize("disable_async_output_proc", [False, True])
async def test_pd_diaggregation_correctness(ray_env, migration_backend, disable_async_output_proc):
    engine_args = VllmEngineArgs(
        engine_args=EngineArgs(
            try_convert_to_local_path("facebook/opt-125m"),
            download_dir="/mnt/model",
            worker_use_ray=True,
            enforce_eager=True,
            disable_async_output_proc=disable_async_output_proc,
        )
    )
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
    res = ray.get(llumlet_0.migrate_out.remote("1", llumlet_1))
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
            llumnix_responses: List[LlumnixRequestOuputVLLM] = await request_output_queue.get()
            request_outputs: List[RequestOutput] = [llumnix_response.get_engine_output()
                                                    for llumnix_response in llumnix_responses]
            for request_output in request_outputs:
                origin_output = request_output.outputs[0]
                finished = request_output.finished

        request_id1 = random_uuid()
        request_expected_steps_id1 = 1
        ray.get(llumlet_0.generate.remote(request_id1, server_info, request_expected_steps_id1, prompt, sampling_params))
        # migrate request for decode
        while True:
            res = ray.get(llumlet_0.migrate_out.remote("1", llumlet_1))
            if len(res) == 1:
                break
        request_output_queue = que
        output = None
        finished = False
        while not finished:
            llumnix_responses: List[LlumnixRequestOuputVLLM] = await request_output_queue.get()
            request_outputs: List[RequestOutput] = [llumnix_response.get_engine_output()
                                                    for llumnix_response in llumnix_responses]
            for request_output in request_outputs:
                output = request_output.outputs[0]
                finished = request_output.finished

        assert output.text == origin_output.text
        assert output.cumulative_logprob == origin_output.cumulative_logprob

    for prompt in TEST_PROMPTS:
        await test_correctness(prompt)

    que.cleanup()

@pytest.mark.asyncio
async def test_clear_migration_states():
    num_gpu_blocks = 8
    block_size = 4
    llumlet = MockLlumlet()
    llumlet.backend_engine.pre_alloc("0", RequestStatus.RUNNING, 0.0, 1, range(4))

    await llumlet.clear_migration_states(is_migrate_in=True)
    assert len(llumlet.backend_engine.pre_alloc("0", RequestStatus.RUNNING, 0.0, num_gpu_blocks, range(4*num_gpu_blocks))) == num_gpu_blocks
    _, seq_group = create_dummy_prompt("0",7,block_size,SequenceStatus.RUNNING)
    seq_group.set_status(RequestStatus.RUNNING_MIGRATING)
    llumlet.backend_engine.add_migrating_out_request_last_stage(seq_group)
    await llumlet.clear_migration_states(is_migrate_in=False)
    assert len(llumlet.backend_engine.get_running_queue()) == 1
    _, seq_group = create_dummy_prompt("0",7,block_size,SequenceStatus.WAITING)
    seq_group.set_status(RequestStatus.WAITING_MIGRATING)
    llumlet.backend_engine.add_migrating_out_request_last_stage(seq_group)
    await llumlet.clear_migration_states(is_migrate_in=False)
    assert len(llumlet.backend_engine.get_waiting_queue()) == 1
