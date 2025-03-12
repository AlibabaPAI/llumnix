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

from vllm.sequence import Logprob, Sequence
from vllm.inputs import token_inputs

from llumnix.backends.vllm.scheduler import BlockManagerLlumnix
from llumnix.llumlet.request import RequestInferenceType, RequestStatus
from .utils import create_dummy_prompt, initialize_scheduler, create_token_budget


def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]

def schedule_and_update_computed_tokens(scheduler):
    metas, out, _ = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    return metas, out

def append_new_token_seq_group(token_chunk_size, seq_group, token_id: int):
    seq_group.update_num_computed_tokens(token_chunk_size)
    for seq in seq_group.get_seqs():
        seq.append_token_id(token_id, {token_id: Logprob(token_id)})

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
    block_table = block_manager.get_free_blocks(2, range(2*block_size))
    after_allocate = block_manager.get_num_free_gpu_blocks()
    assert after_allocate + 2 == before_allocate
    block_table.free()
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
    block_table = block_manager.get_free_blocks(2, range(2*block_size))
    seq = Sequence(1,token_inputs([0]),block_size)
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
    metas, out, _ = scheduler.schedule()
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

def test_scheduler_waiting_request():
    scheduler = initialize_scheduler()
    num_seq_group = 4
    block_size = 4
    _, seq_group_0 = create_dummy_prompt("0", prompt_length=0, block_size=block_size)
    for idx in range(1, num_seq_group + 1):
        _, seq_group = create_dummy_prompt(str(idx), prompt_length=idx, block_size=block_size)
        scheduler.add_seq_group(seq_group)
    assert scheduler.get_num_unfinished_seq_groups() == 4
    scheduler.remove_waiting_request("1")
    assert scheduler.get_num_unfinished_seq_groups() == 3
    _, seq_group = create_dummy_prompt("6", prompt_length=idx, block_size=block_size)
    scheduler.add_waiting_request(seq_group)
    assert scheduler.get_num_unfinished_seq_groups() == 4
    # Test if sort the waiting queue by arrival time in add_waiting_request.
    scheduler.add_waiting_request(seq_group_0)
    waiting_queue = scheduler.get_waiting_queue()
    assert waiting_queue[0] == seq_group_0

def test_scheduler_migrating_out_request_last_stage():
    scheduler = initialize_scheduler()
    block_size = 4
    _, seq_group = create_dummy_prompt("1", prompt_length=1, block_size=block_size)
    scheduler.add_migrating_out_request_last_stage("0", seq_group)
    assert len(scheduler.pop_migrating_out_requests_last_stage("0")) == 1
    scheduler.add_migrating_out_request_last_stage("0", seq_group)
    scheduler.remove_migrating_out_request_last_stage("0", seq_group)
    assert len(scheduler.pop_migrating_out_requests_last_stage("0")) == 0

def test_scheduler_pre_alloc():
    # total 8 blocks
    scheduler = initialize_scheduler()

    blocks = scheduler.pre_alloc("0", "1", RequestStatus.RUNNING, 0.0, 2, range(2*4))
    assert len(blocks) == 2
    assert len(scheduler.pre_alloc_cache_dict["1"].physical_block_ids) == 2
    blocks = scheduler.pre_alloc("0", "1", RequestStatus.RUNNING, 0.0, 4, range(4*4))
    assert len(blocks) == 4
    assert len(scheduler.pre_alloc_cache_dict["1"].physical_block_ids) == 6
    blocks = scheduler.pre_alloc("0", "2,", RequestStatus.RUNNING, 0.0, 4, range(4*4))
    assert len(blocks) == 0

def test_schedule_running():
    scheduler = initialize_scheduler()
    budget = create_token_budget()
    curr_loras = None

    _, seq_group_0 = create_dummy_prompt("0", prompt_length=1, expected_steps=math.inf)
    scheduler._allocate_and_set_running(seq_group_0)
    append_new_token_seq_group(1, seq_group_0, 1)
    scheduler.running.append(seq_group_0)
    running_scheduled = scheduler._schedule_running(budget, curr_loras)

    assert len(running_scheduled.decode_seq_groups_list) == 1
    assert len(running_scheduled.prefill_seq_groups_list) == 0
    assert len(scheduler.running) == 0

    _, seq_group_1 = create_dummy_prompt("1", prompt_length=1, expected_steps=1)
    scheduler._allocate_and_set_running(seq_group_1)
    append_new_token_seq_group(1, seq_group_1, 1)
    scheduler.running.append(seq_group_1)
    running_scheduled = scheduler._schedule_running(
        scheduler.running, budget, curr_loras)
    assert len(running_scheduled.decode_seq_groups_list) == 0
    assert len(running_scheduled.prefill_seq_groups_list) == 0
    assert len(scheduler.running) == 1

    # test pre alloc waiting condition
    # total 8 blocks
    scheduler = initialize_scheduler()
    before_arrival = time.time()
    _, seq_group = create_dummy_prompt("1", prompt_length=1, block_size=2, expected_steps=math.inf)
    after_arrival = time.time()
    blocks = scheduler.pre_alloc("0", "2", RequestStatus.WAITING_MIGRATING, after_arrival, 2, range(2*4))
    assert len(blocks) == 2
    scheduler.add_waiting_request(seq_group)
    blocks = scheduler.pre_alloc("0", "3", RequestStatus.WAITING_MIGRATING, after_arrival, 2, range(2*4))
    assert len(blocks) == 0
    blocks = scheduler.pre_alloc("0", "4", RequestStatus.WAITING_MIGRATING, before_arrival, 2, range(2*4))
    assert len(blocks) == 2

def test_try_schedule_times():
    # total 8 blocks
    scheduler = initialize_scheduler()
    _, seq_group_1 = create_dummy_prompt("1", prompt_length=32, block_size=4)
    _, seq_group_2 = create_dummy_prompt("2", prompt_length=32, block_size=4)
    scheduler.add_seq_group(seq_group_1)
    scheduler.add_seq_group(seq_group_2)
    waiting_queue = scheduler.get_waiting_queue()
    assert len(waiting_queue) == 2
    assert seq_group_1.try_schedule_times == 0
    assert seq_group_2.try_schedule_times == 0
    scheduler.schedule()
    # seq_group_2 cannot be scheduled due to lack of blocks
    assert seq_group_1.try_schedule_times == 0
    assert seq_group_2.try_schedule_times == 1
    append_new_token_seq_group(1, seq_group_1, 1)
    scheduler.schedule()
    # seq_group_1 is preempted to waiting queue
    assert seq_group_1.try_schedule_times == 1
    assert seq_group_2.try_schedule_times == 2
