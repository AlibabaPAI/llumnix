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
import asyncio
from unittest.mock import MagicMock
import pytest
import ray


from llumnix.backends.bladellm.llm_engine import LLMEngineLlumnix
# from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor, SimGPUExecutor
from llumnix.backends.profiling import LatencyMemData
# from llumnix.backends.vllm.sequence import LlumnixRequest
from llumnix.queue.queue_type import QueueType
from llumnix.server_info import ServerInfo


from vllm.sequence import (Logprob, SequenceGroupOutput, SequenceOutput,
                           SequenceStatus,SamplerOutput)
from vllm import EngineArgs, SamplingParams
from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.transformers_utils.detokenizer import Detokenizer

from blade_llm.service.proto import bladellm_pb2 as pb
from blade_llm.service.scheduler_types import SchedulerAsyncUpdateOutput
from blade_llm.protocol import GenerateStreamResponse


from .utils import create_dummy_request, initialize_scheduler
# from tests.unit_test.backends.bladellm.utils import initialize_scheduler, create_dummy_request



class MockEngine(LLMEngineLlumnix):
    def __init__(self, *args, executor_class=None, **kwargs):
        self._scheduler = initialize_scheduler()
        self.step_request_queue = asyncio.Queue()
#         detokenizer = MagicMock(spec=Detokenizer)
#         stop_checker = MagicMock(spec=StopChecker)
#         self.seq_counter = Counter()
#         self.instance_info = None
#         self.executor_class = executor_class
#         self.scheduler.add_update_instance_info_callback(self.update_instance_info)
#         self.output_processor = SingleStepOutputProcessor(self.scheduler.scheduler_config,detokenizer, self.scheduler, self.seq_counter, stop_checker)

#     def update_instance_info(self, instance_info):
#         pass

@pytest.mark.asyncio

async def test_llm_engine_process_model_outputs():
    # process model outputs
    llm_engine = MockEngine()
    prompt_length = 7
    server_request_0 = create_dummy_request(0, prompt="0" * prompt_length)
    server_request_1 = create_dummy_request(1, prompt="1" * prompt_length)

    llm_engine._scheduler.add_request(server_request_0)
    llm_engine._scheduler.add_request(server_request_1)

    _ = llm_engine._scheduler.step()

    update_output = SchedulerAsyncUpdateOutput(
                    response={
                        0: GenerateStreamResponse(is_finished=False),
                        1: GenerateStreamResponse(is_finished=False),
                    },
                )
    # normal case, all requests be processed
    llm_engine.process_model_outputs(update_output)
    request_outputs, _ = llm_engine.step_request_queue.get_nowait()
    assert len(request_outputs) == 2
    step_out = llm_engine._scheduler.step()
    llm_engine._scheduler.remove_running_request(0)
    # migration case , requests stopping during last stage migration, stop process
    llm_engine.process_model_outputs(update_output)
    request_outputs, _ = llm_engine.step_request_queue.get_nowait()
    assert len(request_outputs) == 1

def test_llm_engine_from_engine_args():
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
#     llm_engine = MockEngine.from_engine_args(engine_args, output_queue_type=QueueType.RAYQUEUE,
#                                              instance_id="0", migration_config=None)
#     assert llm_engine.executor_class == LlumnixRayGPUExecutor

#     latency_data = LatencyMemData({},{},{})
#     llm_engine = MockEngine.from_engine_args(engine_args, output_queue_type=QueueType.RAYQUEUE,
#                                              instance_id="0", migration_config=None, latency_mem=latency_data)
#     assert llm_engine.executor_class == SimGPUExecutor

# def test_llm_engine_add_requset():
#     engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
#     llm_engine = LLMEngineLlumnix.from_engine_args(engine_args,
#                                                    output_queue_type=QueueType.RAYQUEUE,
#                                                    instance_id="0",
#                                                    placement_group=None,
#                                                    node_id=ray.get_runtime_context().get_node_id(),
#                                                    migration_config=None,
#                                                    latency_mem=MagicMock(sepc=LatencyMemData))
#     sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
#     server_info = ServerInfo(None, None, None, None, None)
#     llm_engine.add_request("0", server_info, math.inf, "prompt", sampling_params)
#     assert len(llm_engine.scheduler.waiting) == 1
#     assert llm_engine.scheduler.waiting[-1].request_id == "0"
#     assert llm_engine.scheduler.waiting[-1].expected_steps == math.inf
#     assert isinstance(llm_engine.scheduler.waiting[-1], LlumnixRequest)
