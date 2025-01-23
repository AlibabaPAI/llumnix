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

from vllm.sequence import (Logprob, SequenceGroupOutput, SequenceOutput,
                           SequenceStatus,SamplerOutput)
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

from tests.conftest import ray_env
from .utils import create_dummy_prompt, initialize_scheduler


class MockEngine(LLMEngineLlumnix):
    def __init__(self, *args, executor_class=None, **kwargs):
        self.scheduler = initialize_scheduler()
        detokenizer = MagicMock(spec=Detokenizer)
        stop_checker = MagicMock(spec=StopChecker)
        self.seq_counter = Counter()
        self.instance_info = None
        self.executor_class = executor_class
        self.scheduler.add_update_instance_info_callback(self.update_instance_info)
        self.output_processor = SingleStepOutputProcessor(self.scheduler.scheduler_config,detokenizer, self.scheduler, self.seq_counter, stop_checker)

    def update_instance_info(self, instance_info):
        pass


def test_llm_engine_process_model_outputs():
    llm_engine = MockEngine()
    _, seq_group_0 = create_dummy_prompt(
        "0", prompt_length=7, block_size=4
    )
    _, seq_group_1 = create_dummy_prompt(
        "1", prompt_length=7, block_size=4
    )
    llm_engine.scheduler.add_seq_group(seq_group_0)
    llm_engine.scheduler.add_seq_group(seq_group_1)
    metas, out = llm_engine.scheduler.schedule()

    seqs = [seq_group_0.get_seqs()[0], seq_group_1.get_seqs()[0]]

    outputs = [
        SequenceGroupOutput(
            samples=[
                SequenceOutput(
                    parent_seq_id=seq.seq_id,
                    output_token=1,
                    logprobs={1: Logprob(0.0)},
                )
            ],
            prompt_logprobs=None,
        ) for seq in seqs
    ]
    sampler_outputs = [SamplerOutput(outputs=outputs)]

    scheduled_seq_groups = out.scheduled_seq_groups
    # normal case, all requests be processed
    ret, _ = llm_engine._process_model_outputs(sampler_outputs, scheduled_seq_groups,[], metas)
    assert len(ret) == 2
    metas, out = llm_engine.scheduler.schedule()
    scheduled_seq_groups = out.scheduled_seq_groups
    seqs[0].status=SequenceStatus.WAITING
    # migration case , requests stopping during last stage migration, stop process
    ret, _ = llm_engine._process_model_outputs(sampler_outputs, scheduled_seq_groups,[], metas)
    assert len(ret) == 1

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
    placement_group = initialize_placement_group(get_placement_group_name("0"), num_cpus=3, num_gpus=1, detached=True)
    llm_engine = LLMEngineLlumnix.from_engine_args(engine_args=engine_args,
                                                   request_output_queue_type=QueueType.RAYQUEUE,
                                                   instance_id="0",
                                                   placement_group=placement_group,
                                                   migration_config=None,
                                                   latency_mem=MagicMock(sepc=LatencyMemData))
    sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
    server_info = ServerInfo(None, None, None, None, None)
    llm_engine.add_request("0", server_info, math.inf, "prompt", sampling_params)
    assert len(llm_engine.scheduler.waiting) == 1
    assert llm_engine.scheduler.waiting[-1].request_id == "0"
    assert llm_engine.scheduler.waiting[-1].expected_steps == math.inf
    assert isinstance(llm_engine.scheduler.waiting[-1], LlumnixRequest)
