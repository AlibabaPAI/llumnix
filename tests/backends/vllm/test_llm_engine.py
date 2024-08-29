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

from unittest.mock import MagicMock

from vllm.sequence import (Logprob, SequenceGroupOutput, SequenceOutput,
                           SequenceStatus,SamplerOutput)
from vllm import EngineArgs
from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

from llumnix.backends.vllm.llm_engine import LLMEngineLlumnix
from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor, SimGPUExecutor
from llumnix.backends.profiling import LatencyMemData

from .utils import create_dummy_prompt, initialize_scheduler


class MockEngine(LLMEngineLlumnix):
    def __init__(self, executor_class=None, *args, **kwargs):
        self.scheduler = initialize_scheduler()
        detokenizer = MagicMock(spec=Detokenizer)
        stop_checker = MagicMock(spec=StopChecker)
        seq_counter = Counter()
        self.executor_class = executor_class

        self.output_processor = SingleStepOutputProcessor(self.scheduler.scheduler_config,detokenizer, self.scheduler, seq_counter, stop_checker)


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
    ret = llm_engine._process_model_outputs(sampler_outputs, scheduled_seq_groups,[], metas)
    assert len(ret) == 2
    metas, out = llm_engine.scheduler.schedule()
    scheduled_seq_groups = out.scheduled_seq_groups
    seqs[0].status=SequenceStatus.WAITING
    # migration case , requests stopping during last stage migration, stop process
    ret = llm_engine._process_model_outputs(sampler_outputs, scheduled_seq_groups,[], metas)
    assert len(ret) == 1

def test_llm_engine_from_engine_args():
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    llm_engine = MockEngine.from_engine_args(engine_args, instance_id="0", migration_config=None)
    assert llm_engine.executor_class == LlumnixRayGPUExecutor

    latency_data = LatencyMemData({},{},{})
    llm_engine = MockEngine.from_engine_args(engine_args, instance_id="0", migration_config=None, latency_mem=latency_data)
    assert llm_engine.executor_class == SimGPUExecutor
