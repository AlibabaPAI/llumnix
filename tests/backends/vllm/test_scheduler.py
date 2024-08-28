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

import time

from vllm.sequence import Sequence
from vllm.sequence import Logprob
from llumnix.backends.vllm.scheduler import BlockManagerLlumnix
from llumnix.llumlet.request import RequestInferenceType
from .utils import create_dummy_prompt, initialize_scheduler


def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]

def schedule_and_update_computed_tokens(scheduler):
    metas, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    return metas, out

def append_new_token(out, token_id: int):
    seq_groups = get_sequence_groups(out)
    for seq_group in seq_groups:
        for seq in seq_group.get_seqs():
            seq.append_token_id(token_id, {token_id: Logprob(token_id)})

def test_manager_get_free_blocks():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockManagerLlumnix(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0)
    before_allocate = block_manager.get_num_free_gpu_blocks()
    block_table = block_manager.get_free_blocks(2)
    after_allocate = block_manager.get_num_free_gpu_blocks()
    assert after_allocate + 2 == before_allocate
    block_manager._free_block_table(block_table)
    after_free = block_manager.get_num_free_gpu_blocks()
    assert after_free == before_allocate

def test_manager_add_block_table():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockManagerLlumnix(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0)
    block_table = block_manager.get_free_blocks(2)
    seq = Sequence(1,"1",[0], block_size=block_size)
    block_manager.add_block_table(block_table, seq.seq_id)
    after_allocate = block_manager.get_num_free_gpu_blocks()
    assert after_allocate + 2 == num_gpu_blocks
    block_manager.free(seq)
    after_free = block_manager.get_num_free_gpu_blocks()
    assert after_free == num_gpu_blocks

def test_sequence_group_inference_type():
    scheduler = initialize_scheduler()
    num_seq_group = 4
    block_size = 4
    for idx in range(1, num_seq_group + 1):
        _, seq_group = create_dummy_prompt(str(idx), prompt_length=idx, block_size=block_size)
        scheduler.add_seq_group(seq_group)

    # all seq_group in waiting queue
    for req in scheduler.waiting:
        assert req.inference_type == RequestInferenceType.PREFILL
    # all seq_group in prefilling stage
    metas, out = scheduler.schedule()
    for req in scheduler.running:
        assert req.inference_type == RequestInferenceType.PREFILL
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    append_new_token(out, 1)
    schedule_and_update_computed_tokens(scheduler)
    # all in running queue
    for req in scheduler.running:
        assert req.inference_type == RequestInferenceType.DECODE

def test_scheduler_num_killed_request():
    scheduler = initialize_scheduler()
    # tot 8 blocks
    num_seq_group = 4
    block_size = 4
    for idx in range(1, num_seq_group + 1):
        _, seq_group = create_dummy_prompt(str(idx), prompt_length=8, block_size=block_size)
        scheduler.add_seq_group(seq_group)
    # remain 0 blocks
    _, out = schedule_and_update_computed_tokens(scheduler)
    append_new_token(out, 1)
    assert scheduler._get_num_killed_requests() == 0
    # preempt 2 requests
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert scheduler._get_num_killed_requests() == 2

def test_scheduler_running_request():
    scheduler = initialize_scheduler()
    num_seq_group = 4
    block_size = 4
    for idx in range(1, num_seq_group + 1):
        _, seq_group = create_dummy_prompt(str(idx), prompt_length=idx, block_size=block_size)
        scheduler.add_seq_group(seq_group)
    schedule_and_update_computed_tokens(scheduler)
    assert scheduler.get_num_unfinished_seq_groups() == 4
    scheduler.remove_running_request("1")
    assert scheduler.get_num_unfinished_seq_groups() == 3
    _, seq_group = create_dummy_prompt("5", prompt_length=idx, block_size=block_size)
    scheduler.add_running_request(seq_group)
    assert scheduler.get_num_unfinished_seq_groups() == 4

def test_scheduler_migrating_out_request_last_stage():
    scheduler = initialize_scheduler()
    block_size = 4
    _, seq_group = create_dummy_prompt("1", prompt_length=1, block_size=block_size)
    scheduler.add_migrating_out_request_last_stage(seq_group)
    assert len(scheduler.pop_migrating_out_requests_last_stage()) == 1
    scheduler.add_migrating_out_request_last_stage(seq_group)
    scheduler.remove_migrating_out_request_last_stage(seq_group)
    assert len(scheduler.pop_migrating_out_requests_last_stage()) == 0

def test_scheduler_pre_alloc():
    # total 8 blocks
    scheduler = initialize_scheduler()
    blocks = scheduler.pre_alloc("1", 2)
    assert len(blocks) == 2
    assert len(scheduler.pre_alloc_cache_dict["1"]) == 2
    blocks = scheduler.pre_alloc("1", 4)
    assert len(blocks) == 4
    assert len(scheduler.pre_alloc_cache_dict["1"]) == 6
    blocks = scheduler.pre_alloc("2,", 4)
    assert len(blocks) == 0

