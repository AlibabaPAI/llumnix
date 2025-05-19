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

import math
from unittest.mock import MagicMock
import os

import torch
import pytest
import ray

from vllm import EngineArgs, SamplingParams
from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter
from vllm.outputs import RequestOutput, CompletionOutput

from llumnix.backends.vllm.llm_engine import LLMEngineLlumnix
from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor
from llumnix.backends.vllm.sim_executor import SimGPUExecutor
from llumnix.backends.profiling import LatencyMemData
from llumnix.backends.vllm.sequence import LlumnixRequest
from llumnix.queue.queue_type import QueueType
from llumnix.server_info import ServerInfo
from llumnix.ray_utils import initialize_placement_group, get_placement_group_name
from llumnix.utils import try_convert_to_local_path, random_uuid
from llumnix.backends.backend_interface import BackendType

# pylint: disable=unused-import
from tests.conftest import ray_env
from .utils import initialize_scheduler


class MockEngine(LLMEngineLlumnix):
    def __init__(self, *args, executor_class=None, **kwargs):
        self.scheduler = [initialize_scheduler()]
        detokenizer = MagicMock(spec=Detokenizer)
        stop_checker = MagicMock(spec=StopChecker)
        self.seq_counter = Counter()
        self.instance_info = None
        self.executor_class = executor_class
        self.scheduler[0].add_update_instance_info_callback(self.update_instance_info)
        self.output_processor = SingleStepOutputProcessor(self.scheduler[0].scheduler_config,detokenizer,
                                                          self.scheduler,
                                                          self.seq_counter,
                                                          stop_checker)

    def update_instance_info(self, instance_info):
        pass


@ray.remote(num_cpus=0)
class MockAsyncPutQueueActor:
    async def put_nowait_to_servers(self, server_request_outputs, server_info_dict):
        self.server_request_outputs = server_request_outputs
        self.server_info_dict = server_info_dict

    def get(self):
        return self.server_request_outputs, self.server_info_dict


def get_engine_args():
    # If not enabling use_ray_spmd_worker, gpu memory will not be released despite actor has been killed for unknown reason.
    os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
    os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "1"
    engine_args = EngineArgs(model=try_convert_to_local_path("facebook/opt-125m"), download_dir="/mnt/model",
                             worker_use_ray=True, enforce_eager=True)
    return engine_args

def init_llm_engine(instance_id):
    placement_group = initialize_placement_group(get_placement_group_name("0"), num_cpus=3, num_gpus=1, detached=True)
    llm_engine = LLMEngineLlumnix.from_engine_args(
        engine_args=get_engine_args(),
        request_output_queue_type=QueueType.RAYQUEUE,
        instance_id=instance_id,
        placement_group=placement_group,
        backend_type=BackendType.VLLM,
        latency_mem=None,
        migration_config=None
    )

    return llm_engine

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need at least 1 GPU to run the test.")
def test_from_engine_args(ray_env):
    engine_args = get_engine_args()
    placement_group = initialize_placement_group(get_placement_group_name("0"), num_cpus=3, num_gpus=1, detached=True)
    llm_engine = MockEngine.from_engine_args(engine_args=engine_args, request_output_queue_type=QueueType.RAYQUEUE,
                                             instance_id="0", migration_config=None, placement_group=placement_group,
                                             backend_type=BackendType.VLLM)
    assert llm_engine.executor_class == LlumnixRayGPUExecutor

def test_from_engine_args_sim(ray_env):
    latency_data = LatencyMemData({},{},{})
    engine_args = get_engine_args()
    placement_group = initialize_placement_group(get_placement_group_name("0"), num_cpus=2, num_gpus=0, detached=True)
    llm_engine = MockEngine.from_engine_args(engine_args=engine_args, request_output_queue_type=QueueType.RAYQUEUE,
                                             instance_id="0", migration_config=None, latency_mem=latency_data,
                                             placement_group=placement_group, backend_type=BackendType.VLLM)
    assert llm_engine.executor_class == SimGPUExecutor

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need at least 1 GPU to run the test.")
async def test_add_requset(ray_env):
    llm_engine = init_llm_engine("0")
    sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
    server_info = ServerInfo(None, None, None, None, None)
    await llm_engine.add_request("0", server_info, math.inf, "prompt", sampling_params)
    assert len(llm_engine.scheduler[0].waiting) == 1
    assert llm_engine.scheduler[0].waiting[-1].request_id == "0"
    assert llm_engine.scheduler[0].waiting[-1].expected_steps == math.inf
    assert isinstance(llm_engine.scheduler[0].waiting[-1], LlumnixRequest)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need at least 1 GPU to run the test.")
async def test_put_request_outputs_to_server(ray_env):
    instance_id = random_uuid()
    llm_engine = init_llm_engine(instance_id)
    async_put_queue_actor = MockAsyncPutQueueActor.remote()
    llm_engine.async_put_queue_actor = async_put_queue_actor
    request_id = random_uuid()
    completion_output = CompletionOutput(0, "", [], 0.0, None)
    request_outputs = [RequestOutput(request_id, "", [], None, [completion_output], finished=True)]
    server_id = random_uuid()
    server_infos = [ServerInfo(server_id, None, None, None, None)]
    llm_engine._put_request_outputs_to_server(request_outputs, server_infos)
    server_request_outputs, _ = ray.get(async_put_queue_actor.get.remote())
    request_outputs_engine = server_request_outputs[server_id]
    request_output, request_output_info = request_outputs_engine[0]
    assert request_output.request_id == request_id
    assert request_output_info.instance_id == instance_id
