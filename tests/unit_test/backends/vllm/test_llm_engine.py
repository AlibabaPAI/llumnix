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

import torch
import pytest

from vllm import EngineArgs, SamplingParams
from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

from llumnix.backends.vllm.llm_engine import LLMEngineLlumnix
from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor
from llumnix.backends.vllm.sim_executor import SimGPUExecutor
from llumnix.backends.profiling import LatencyMemData
from llumnix.backends.vllm.sequence import LlumnixRequest
from llumnix.queue.queue_type import QueueType
from llumnix.server_info import ServerInfo
from llumnix.utils import initialize_placement_group, get_placement_group_name

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

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need at least 1 GPU to run the test.")
def test_llm_engine_from_engine_args(ray_env):
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    placement_group = initialize_placement_group(get_placement_group_name("0"), num_cpus=3, num_gpus=1, detached=True)
    llm_engine = MockEngine.from_engine_args(engine_args=engine_args, request_output_queue_type=QueueType.RAYQUEUE,
                                             instance_id="0", migration_config=None, placement_group=placement_group)
    assert llm_engine.executor_class == LlumnixRayGPUExecutor

def test_llm_engine_from_engine_args_sim(ray_env):
    latency_data = LatencyMemData({},{},{})
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    placement_group = initialize_placement_group(get_placement_group_name("0"), num_cpus=2, num_gpus=1, detached=True)
    llm_engine = MockEngine.from_engine_args(engine_args=engine_args, request_output_queue_type=QueueType.RAYQUEUE,
                                             instance_id="0", migration_config=None, latency_mem=latency_data,
                                             placement_group=placement_group)
    assert llm_engine.executor_class == SimGPUExecutor

def test_llm_engine_add_requset(ray_env):
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    latency_data = LatencyMemData({},{},{})
    placement_group = initialize_placement_group(get_placement_group_name("0"), num_cpus=1, num_gpus=0, detached=True)
    llm_engine = LLMEngineLlumnix.from_engine_args(engine_args=engine_args,
                                                   request_output_queue_type=QueueType.RAYQUEUE,
                                                   instance_id="0",
                                                   placement_group=placement_group,
                                                   latency_mem = latency_data,
                                                   migration_config=None)
    sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
    server_info = ServerInfo(None, None, None, None, None)
    llm_engine.add_request("0", server_info, math.inf, "prompt", sampling_params)
    assert len(llm_engine.scheduler[0].waiting) == 1
    assert llm_engine.scheduler[0].waiting[-1].request_id == "0"
    assert llm_engine.scheduler[0].waiting[-1].expected_steps == math.inf
    assert isinstance(llm_engine.scheduler[0].waiting[-1], LlumnixRequest)
