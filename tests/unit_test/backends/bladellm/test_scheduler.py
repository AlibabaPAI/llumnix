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

from blade_llm.service.paged_utils import PagedRequestState
from blade_llm.service.proto.bladellm_pb2 import (
    DetokenParams,
    StoppingCriteria,
    WorkerRequest,
)
from blade_llm.service.proto import bladellm_pb2 as pb


from llumnix.backends.bladellm.scheduler import BlockManagerLlumnix
from llumnix.llumlet.request import RequestInferenceType
from tests.unit_test.backends.bladellm.utils import initialize_scheduler, create_dummy_request

DUMMY_PROMPT = "This is a dummy prompt"

def get_num_unfinished_seq_groups(scheduler):
    return len(scheduler.waiting) + len(scheduler.running) + len(scheduler.swapped)

def update_computed_tokens(scheduler, step_output):
    request_groups_map = scheduler.get_request_groups_map()
    step_ids = list(step_output.step.decode) + [r.id for r in step_output.step.prefill]
    for r_id in step_ids:
        request_groups_map[r_id].update_num_computed_tokens(request_groups_map[r_id].token_chunk_size)

def step_and_update(request_ids, scheduler, is_finished=False, tokens_num=1, update_only=False):
    if not isinstance(request_ids, list):
        request_ids = [request_ids]
    if not update_only:
        step_out = scheduler.step()
        update_computed_tokens(scheduler, step_out)
    if hasattr(scheduler, "_decode_batch"):
        running_ids = scheduler._decode_batch
    elif hasattr(scheduler, "running"):
        running_ids = [request_group.request_group_id for request_group in scheduler.running]
    else:
        raise ValueError("scheduler has no running or decode_batch")
    if is_finished:
        generation_groups = [
            pb.GenerationGroup(
                request_group_id=request_id,
                generations=[pb.RequestMeta(request_id=request_id)],
            )
            for request_id in request_ids
            if request_id in running_ids
        ]
    else:
        generation_groups = [
            pb.GenerationGroup(
                request_group_id=request_id,
                generations=[pb.RequestMeta(request_id=request_id)],
                next_ids=pb.NextIds(ids=[(x + 1) for x in range(tokens_num)]),
            )
            for request_id in request_ids
            if request_id in running_ids
        ]
    resp = pb.WorkerStepResponse(
        is_ok=True,
        batch_id=-1,
        generation_groups=pb.GenerationGroupList(generation_group=generation_groups),
    )
    _ = scheduler.update(resp)
    
def test_manager_get_free_blocks():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockManagerLlumnix(block_size,
                                        num_gpu_blocks,
                                        num_cpu_blocks,
                                        block_reserved_percent=0)
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
                                        num_gpu_blocks,
                                        num_cpu_blocks,
                                        block_reserved_percent=0)
    block_table = block_manager.get_free_blocks(2)
    req_proto = WorkerRequest(
        id=1,
        prompt=DUMMY_PROMPT,
        stopping_criterial=StoppingCriteria(max_new_tokens=30),
        detoken_params=DetokenParams(cat_prompt=True),
    )
    req_state = PagedRequestState(req_proto, block_size=block_size)
    block_manager.add_block_table(block_table, req_state.block_table_id)
    after_allocate = block_manager.get_num_free_gpu_blocks()
    assert after_allocate + 2 == num_gpu_blocks
    block_manager.free(req_state)
    after_free = block_manager.get_num_free_gpu_blocks()
    assert after_free == num_gpu_blocks

def test_sequence_group_inference_type():
    scheduler = initialize_scheduler()
    num_gen_group = 4
    request_ids = [idx for idx in range(1, num_gen_group + 1)]
    for idx in request_ids:
        server_request = create_dummy_request(idx, prompt=str(idx))
        scheduler.add_request(server_request)

    # all seq_group in waiting queue
    for req in scheduler.waiting:
        assert req.inference_type == RequestInferenceType.PREFILL
    # all seq_group in prefilling stage
    step_out = scheduler.step()
    for req in scheduler.running:
        assert req.inference_type == RequestInferenceType.PREFILL
    update_computed_tokens(scheduler, step_out)
    step_and_update(request_ids, scheduler)
    # all in running queue
    for req in scheduler.running:
        assert req.inference_type == RequestInferenceType.DECODE

def test_scheduler_num_killed_request():
    scheduler = initialize_scheduler()
    # tot 8 blocks
    num_gen_group = 4
    request_ids = [idx for idx in range(1, num_gen_group + 1)]
    for idx in request_ids:
        # BladeLLM allocate blocks for next step in advance.
        server_request = create_dummy_request(idx, prompt=str(idx) * 7)
        scheduler.add_request(server_request)
    # remain 0 blocks
    step_and_update(request_ids, scheduler)
    assert scheduler._get_num_killed_requests() == 0
    # preempt 2 requests
    step_and_update(request_ids, scheduler)
    assert scheduler._get_num_killed_requests() == 2

def test_scheduler_running_request():
    scheduler = initialize_scheduler()
    num_gen_group = 4
    request_ids = [idx for idx in range(1, num_gen_group + 1)]
    for idx in request_ids:
        server_request = create_dummy_request(idx, prompt=str(idx))
        scheduler.add_request(server_request)
    step_and_update(request_ids, scheduler)
    assert get_num_unfinished_seq_groups(scheduler) == 4
    scheduler.remove_running_request(1)
    assert get_num_unfinished_seq_groups(scheduler) == 3
    server_request = create_dummy_request(5, prompt=DUMMY_PROMPT)
    scheduler.add_running_request(server_request)
    assert get_num_unfinished_seq_groups(scheduler) == 4

def test_scheduler_migrating_out_request_last_stage():
    scheduler = initialize_scheduler()
    server_request = create_dummy_request(1, prompt=DUMMY_PROMPT)
    scheduler.add_migrating_out_request_last_stage(server_request)
    assert len(scheduler.pop_migrating_out_requests_last_stage()) == 1
    scheduler.add_migrating_out_request_last_stage(server_request)
    scheduler.remove_migrating_out_request_last_stage(server_request)
    assert len(scheduler.pop_migrating_out_requests_last_stage()) == 0

def test_scheduler_pre_alloc():
    # total 8 blocks
    scheduler = initialize_scheduler()
    blocks = scheduler.pre_alloc(1, 2)
    assert len(blocks) == 2
    assert len(scheduler.pre_alloc_cache_dict[1]) == 2
    blocks = scheduler.pre_alloc(1, 4)
    assert len(blocks) == 4
    assert len(scheduler.pre_alloc_cache_dict[1]) == 6
    blocks = scheduler.pre_alloc(2, 4)
    assert len(blocks) == 0
