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

# pylint: disable=protected-access

import time
from typing import Dict, List, Set, Union
import heapq

from loguru import logger

from blade_llm.service.paged_utils import PreemptionMode
from blade_llm.service.block_space_manager import BlockTable
from blade_llm.service.proto.bladellm_pb2 import FinishedInfo
from blade_llm.service.schedulers import PagedScheduler
from blade_llm.service.scheduler_types import SchedulerStepOutput, GenerationGroupState
from blade_llm.service.args import ServingArgs

from llumnix.backends.bladellm.metrics import BladeLLMMetrics
from llumnix.backends.bladellm.sequence import GenerationGroupStateLlumnix
from llumnix.llumlet.request import RequestStatus
from llumnix.backends.bladellm.llm_engine import AsyncBackQueueWrapper
from llumnix.server_info import ServerInfo


class PagedSchedulerLlumnix(PagedScheduler):
    def __init__(self, serving_args: ServingArgs, *args, **kwargs) -> None:
        PagedScheduler.__init__(self, serving_args, *args, **kwargs)
        self.llumnix_metrics = BladeLLMMetrics()
        self.id2group: Dict[int, GenerationGroupStateLlumnix] = {}
        self.pre_alloc_cache_dict: Dict[int, BlockTable] = {}
        self.migrating_out_request_last_stage: Dict[int, GenerationGroupStateLlumnix] = {}
        self.llumnix_metrics.block_manager_init_metrics(self.block_manager)
        self.llumnix_metrics.scheduler_init_metrics(self)

        self.running_filter_request_ids: Set[int] = set()
        # init in engine.start
        self.trans_wrapper: AsyncBackQueueWrapper = None
        self.step_counter: int = 0

    def pipeline_running_filter(self, batches: Union[List[int], List[GenerationGroupState]]):
        batches = super().pipeline_running_filter(batches)
        if not batches:
            return batches

        if isinstance(batches[0], GenerationGroupState):
            after =  [b for b in batches if b.request_group_id not in self.running_filter_request_ids]
        else:
            after =  [b for b in batches if b not in self.running_filter_request_ids]

        return after

    def is_hunger(self):
        def all_running_migrated_condition():
            running_request_ids = [req.request_group_id for req in self.running]
            return len(set(running_request_ids) - set(self.running_filter_request_ids)) == 0

        general_hunger = super().is_hunger()
        if not general_hunger:
            hunger_in_migrate = not self.waiting and not self.swapped and not self.hanging \
                and self.pipeline_running_is_empty() and not self.offloading_waiting \
                and not self.handling_elastic_prefill and all_running_migrated_condition()
            general_hunger = general_hunger or hunger_in_migrate

        return general_hunger

    def step(self) -> SchedulerStepOutput:
        self.step_counter += 1
        step_out = super().step()
        self.llumnix_metrics.scheduler_step_metrics(self)
        return step_out

    # migration related method

    # happends when add_request
    def add_gen_group(self, gen_group: GenerationGroupState, *args, **kwargs):
        server_info: ServerInfo = None # just set to None for warm-up requests
        if gen_group.request_group_id in self.trans_wrapper.request_server_map:
            server_info = self.trans_wrapper.request_server_map[gen_group.request_group_id]
        gen_group_llumnix = GenerationGroupStateLlumnix(
            gen_group, gen_group.request_group_id,
            server_info)
        gen_group_llumnix._status = RequestStatus.WAITING
        self.id2group[gen_group.request_group_id] = gen_group_llumnix
        super().add_gen_group(gen_group_llumnix, *args, **kwargs)

    def drop_request(self, req_id: int):
        self.id2group[req_id]._status = RequestStatus.FINISHED
        self.trans_wrapper.remove_request_server_info(req_id, self.step_counter + 1)
        super().drop_request(req_id)

    # happends when moving request from waiting to running
    def _add_prefill_to_queue(self, group_to_add: GenerationGroupStateLlumnix):
        super()._add_prefill_to_queue(group_to_add)
        group_to_add._status = RequestStatus.RUNNING

    def _preempt(self, gen_group: GenerationGroupStateLlumnix, insert=True) -> PreemptionMode:
        gen_group.last_preemption_time = time.time()
        return super()._preempt(gen_group, insert)

    def safe_remove_requests(self, request_ids: Set[int]):
        super().safe_remove_requests(request_ids)
        for request_id in request_ids:
            if request_id in self.id2group:
                self.id2group[request_id]._status = RequestStatus.FINISHED
                self.id2group.pop(request_id, None)

    def get_request_incremental_blocks(self, backend_request: GenerationGroupStateLlumnix, pre_stage_num_blocks: int) -> List[int]:
        assert len(backend_request.paged_reqs) == 1, "currently llumnix doesn't support multi-paged-req migration."
        paged_req = backend_request.paged_reqs[0]
        target_blocks = []
        if paged_req.block_table_id in self.block_manager.block_tables:
            blocks = self.block_manager.get_block_table(paged_req)
            target_blocks = blocks[pre_stage_num_blocks:]
            logger.debug(f"request {backend_request.request_id} get incremental blocks: {len(target_blocks)}")
        else:
            logger.warning(f"request {backend_request.request_id} not found in block manager, maybe finished.")
        return target_blocks

    def get_running_queue(self):
        return self.running

    def get_waiting_queue(self):
        return self.waiting

    def remove_running_request(self, request_id: int) -> None:
        for index, gen_group in enumerate(self.running):
            assert isinstance(gen_group, GenerationGroupStateLlumnix)
            if gen_group.request_group_id == request_id:
                self.running.pop(index)
                self._detokenizer.remove_state(request_id)
                self.id2group.pop(request_id, None)
                gen_group.set_status(RequestStatus.RUNNING_MIGRATING)
                return True
        return False

    def remove_waiting_request(self, request_id: int) -> bool:
        for index, seq_group in enumerate(self.waiting):
            if seq_group.request_group_id == request_id:
                self.running.pop(index)
                self._detokenizer.remove_state(request_id)
                return True
        return False

    def add_migrating_out_request_last_stage(self, backend_request: GenerationGroupStateLlumnix) -> None:
        self.migrating_out_request_last_stage[backend_request.request_group_id] = backend_request

    def add_running_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        backend_request.set_status(RequestStatus.RUNNING)
        self.id2group[backend_request.request_id] = backend_request
        self._detokenizer.add_new_request(
            backend_request.paged_reqs[0].req_proto,
            backend_request.detokenizer_state)

        self.running.append(backend_request)
        self.running.sort(key=lambda x: x.receive_time)

    def add_waiting_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        heapq.push(self.waiting, backend_request)

    def pop_migrating_out_request_last_stage(self, backend_request: GenerationGroupStateLlumnix) -> None:
        assert backend_request.request_id in self.migrating_out_request_last_stage, \
            "the request id of migrating out request in last stage should exist in migrating out request last stage"
        self.migrating_out_request_last_stage.pop(backend_request.request_id)

    # pylint: disable=unused-argument
    def pre_alloc(self, request_id: int, request_status: RequestStatus, request_arrival_time: float,
                  block_num: int, token_ids: List[int]) -> List[int]:
        if request_status == RequestStatus.WAITING_MIGRATING:
            if (self.waiting and request_arrival_time > self.waiting[0].arrival_time):
                return []

        blocks = []
        if not self.block_manager.can_allocate_num_blocks(block_num):
            return blocks
        for _ in range(block_num):
            block = self.block_manager.gpu_allocator.allocate()
            block.ref_count = 1
            blocks.append(block)
        pre_blocks = self.pre_alloc_cache_dict.get(request_id, [])
        pre_blocks.extend(blocks)
        self.pre_alloc_cache_dict[request_id] = pre_blocks
        blocks = [block.block_number for block in blocks]
        return blocks

    def free_src_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        assert backend_request.paged_reqs[0].block_table_id in self.block_manager.block_tables, "block table not found"
        self._free_req(backend_request)
        self._finished_req_to_remove.append(FinishedInfo(request_id=backend_request.request_id, pos=0))

    def free_dst_pre_alloc_cache(self, request_id: int = None) -> None:
        if request_id:
            blocks = self.pre_alloc_cache_dict.pop(request_id, [])
            # pylint: disable=protected-access
            self.block_manager._free_block_table(blocks)
        else:
            # Clear all pre-allocated cache of dst instance when src instance encounters exception.
            request_ids = list(self.pre_alloc_cache_dict.keys())
            for req_id in request_ids:
                blocks = self.pre_alloc_cache_dict.pop(req_id, [])
                # pylint: disable=protected-access
                self.block_manager._free_block_table(blocks)

    def free_migrating_out_requests_last_stage(self) -> List[GenerationGroupStateLlumnix]:
        migrating_out_requests_last_stage = list(self.migrating_out_request_last_stage.values())
        self.migrating_out_request_last_stage.clear()
        return migrating_out_requests_last_stage

    def add_block_table(self, block_table: BlockTable, block_table_id: int) -> None:
        self.block_manager.block_tables[block_table_id] = block_table

    def get_num_killed_requests(self) -> int:
        cnt = len(self.swapped)
        for seq_group in self.waiting:
            if seq_group.last_preemption_time:
                cnt += 1
        return cnt

    def get_num_blocks_first_waiting_request(self) -> int:
        return self.waiting[0].paged_reqs[0].required_blocks if len(self.waiting) > 0 else 0

    def get_num_blocks_last_running_request(self) -> int:
        return self.running[-1].paged_reqs[0].required_blocks if len(self.running) > 0 else 0

    def get_num_cached_request_ids(self) -> int:
        return len(self.id2group)

    def get_num_blocks_all_waiting_requests(self) -> int:
        num_blocks_all_waiting_requests = 0
        for gen_group_state in self.waiting:
            num_blocks_all_waiting_requests += sum([page_req.required_blocks for page_req in gen_group_state.paged_reqs])
        return num_blocks_all_waiting_requests
