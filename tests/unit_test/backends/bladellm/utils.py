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
import time
from unittest.mock import MagicMock
from typing import Iterable, Optional, Tuple
from transformers import PreTrainedTokenizerBase

from blade_llm.protocol import DetokenParams
from blade_llm.service.schedulers import PagedScheduler
from blade_llm.service.args import ServingArgs, ServingLoraOptions
from blade_llm.service.scheduler_types import SchedulerInitInfo
from blade_llm.model.config_base import ConfigBase, ModelType
from blade_llm.model.config_utils import load_config
from blade_llm.utils.load_model_options import LoadModelOptions
from blade_llm.protocol import SamplingParams, ServerRequest, StoppingCriteria
from blade_llm.model.tokenizer_utils import load_tokenizer



from llumnix.backends.bladellm.scheduler import PagedSchedulerLlumnix
from llumnix.backends.bladellm.sequence import ServerRequestLlumnix
from llumnix.server_info import ServerInfo

def initialize_scheduler() -> PagedSchedulerLlumnix:
    block_size = 4
    max_processing_units = 8
    serving_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/dataset/opt-125m'))
    tokenizer = load_tokenizer(serving_args.load_model_options.tokenizer_dir, serving_args.load_model_options.special_token_dict)
    sched_init_info = SchedulerInitInfo(
        token_capacity=max_processing_units, block_size=block_size, model_max_len=2048, cpu_blocks=max_processing_units
    )
    model_conf=load_config(serving_args.load_model_options.model)
    scheduler = PagedSchedulerLlumnix(serving_args, tokenizer, sched_init_info, model_conf)
    scheduler.block_manager.reserved_blocks = 0
    scheduler.update_instance_info_callback = MagicMock()
    return scheduler

def create_dummy_request(
        id,
        prompt="",
        max_new_tokens=30,
        use_beam_search=False,
        beam_width=0,
        best_of=1,
        prompt_tokens=[],
        lora_path=None,):
    if use_beam_search:
        best_of = beam_width
    if prompt != "":
        prompt_tokens = list(range(len(prompt)))
    server_info = ServerInfo(None, None, None, None, None)
    server_req = ServerRequest(
        id=id,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        sampling_params=SamplingParams(
            temperature=-1,
            top_k=10,
            top_p=0.8,
            use_beam_search=use_beam_search,
            beam_width=beam_width,
            best_of=best_of,
        ),
        stopping_criterial=StoppingCriteria(max_new_tokens=max_new_tokens, ignore_eos=True),
        lora_path=lora_path,
        detoken_params = DetokenParams(cat_prompt=False))
    return ServerRequestLlumnix(id, server_info, -1, server_request=server_req)


# def create_seq_group(
#         seq_prompt_len: int = 1024,
#         seq_output_lens: Iterable[int] = (128, ),
#         request_id: str = '0',
#         seq_id_start: int = 0,
#         sampling_params: Optional[SamplingParams] = None) -> SequenceGroupLlumnix:

#     assert len(seq_output_lens) > 0

#     if sampling_params is None:
#         sampling_params = SamplingParams()

#     prompt_token_ids = [0] * seq_prompt_len

#     seqs = []
#     for seq_id_offset, output_len in enumerate(seq_output_lens):
#         seq = Sequence(
#             seq_id=seq_id_start + seq_id_offset,
#             prompt="",
#             prompt_token_ids=prompt_token_ids,
#             block_size=16,
#         )

#         for i in range(output_len):
#             seq.append_token_id(
#                 token_id=i,
#                 logprobs={i: Logprob(0.0)},
#             )
#         seqs.append(seq)

#     seq_group = SequenceGroupLlumnix(
#         request_id=request_id,
#         server_info=None,
#         expected_steps=math.inf,
#         seqs=seqs,
#         sampling_params=sampling_params,
#         arrival_time=time.time(),
#     )

#     return seq_group

# def create_token_budget(token_budget: int = 10000,
#                         max_num_seqs: int = 10000) -> SchedulingBudget:
#     return SchedulingBudget(
#         token_budget=token_budget,
#         max_num_seqs=max_num_seqs,
#     )

# def round_up_to_next_block(seq_len: int, block_size: int) -> int:
#     return (seq_len + block_size - 1) // block_size
