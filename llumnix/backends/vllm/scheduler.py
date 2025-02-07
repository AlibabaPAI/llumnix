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
import bisect
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque

from vllm.utils import Device
from vllm.core.block_manager import SelfAttnBlockSpaceManager, BlockTable
from vllm.core.scheduler import (Scheduler, PreemptionMode, SequenceStatus, SequenceGroupMetadata, SchedulerOutputs)
from vllm.sequence import SequenceGroup
from vllm.core.interfaces import AllocStatus

from llumnix.instance_info import InstanceInfo
from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus
from llumnix.backends.vllm.sequence import SequenceGroupLlumnix


logger = init_logger(__name__)


# TODO(ZeldaHuang): adapt prefix cache and sliding window, now use v1 manager
class BlockManagerLlumnix(SelfAttnBlockSpaceManager):
    def get_free_blocks(self, num_required_blocks: int, token_ids: List[int]) -> BlockTable:
        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(device=Device.GPU)
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        if (num_free_gpu_blocks - num_required_blocks >=
                self.watermark_blocks):
            block_table.allocate(token_ids)

        return block_table

    def add_block_table(self, block_table: BlockTable, seq_id: int) -> None:
        self.block_tables[seq_id] = block_table
        self._computed_blocks_tracker.add_seq(seq_id)
        self._last_access_blocks_tracker.add_seq(seq_id)

class SchedulerLlumnix(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_manager = BlockManagerLlumnix(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)
        self.pre_alloc_cache_dict: Dict[str, BlockTable] = {}
        self.migrating_out_request_last_stage: List[SequenceGroupLlumnix] = []

    def add_update_instance_info_callback(self, update_instance_info_callback):
        self.update_instance_info_callback = update_instance_info_callback
        self.update_instance_info_callback(self._get_instance_info([]))

    def _preempt(
        self,
        seq_group: SequenceGroupLlumnix,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        seq_group.last_preemption_time = time.time()
        return super()._preempt(seq_group, blocks_to_swap_out, preemption_mode)

    def _get_num_killed_requests(self) -> int:
        cnt = len(self.swapped)
        for seq_group in self.waiting:
            if seq_group.last_preemption_time:
                cnt += 1
        return cnt

    def get_running_queue(self) -> Deque[SequenceGroupLlumnix]:
        return self.running

    def get_waiting_queue(self) -> Deque[SequenceGroupLlumnix]:
        return self.waiting

    def get_all_request_ids(self) -> List[str]:
        request_ids : List[str] = []
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                request_ids.append(seq_group.request_id)
        return request_ids

    def get_request_incremental_blocks(self, backend_request: LlumnixRequest, pre_stage_num_blocks: int) -> Tuple[List[int], List[int]]:
        seq = backend_request.get_seqs()[0]
        if seq.seq_id not in self.block_manager.block_tables:
            return []
        blocks = self.block_manager.get_block_table(seq)
        block_table = self.block_manager.block_tables[seq.seq_id]
        token_ids = backend_request.token_ids
        return blocks[pre_stage_num_blocks:], token_ids[pre_stage_num_blocks * self.block_manager.block_size:block_table.num_full_slots]

    def remove_running_request(self, request_id: str) -> bool:
        for seq_group in reversed(self.running):
            if seq_group.request_id == request_id:
                self.running.remove(seq_group)
                seq_group.set_status(RequestStatus.RUNNING_MIGRATING)
                return True
        return False

    def remove_waiting_request(self, request_id: str) -> bool:
        for seq_group in self.waiting:
            if seq_group.request_id == request_id:
                self.waiting.remove(seq_group)
                seq_group.set_status(RequestStatus.WAITING_MIGRATING)
                return True
        return False

    def add_migrating_out_request_last_stage(self, backend_request: SequenceGroupLlumnix) -> None:
        self.migrating_out_request_last_stage.append(backend_request)

    def remove_migrating_out_request_last_stage(self, backend_request: SequenceGroupLlumnix) -> None:
        self.migrating_out_request_last_stage.remove(backend_request)

    def pop_migrating_out_requests_last_stage(self) -> List[SequenceGroupLlumnix]:
        migrating_out_request_last_stage = self.migrating_out_request_last_stage.copy()
        self.migrating_out_request_last_stage.clear()
        return migrating_out_request_last_stage

    def pre_alloc(self,
                  request_id: str,
                  request_status: RequestStatus,
                  request_arrival_time: float,
                  block_num: int,
                  token_ids: List[int]) -> List[int]:
        # Only migrate waiting request when the waiting request is the earliest arrival one
        # among the requests of dst instance's waiting queue.
        if request_status == RequestStatus.WAITING_MIGRATING:
            if self.waiting and request_arrival_time > self.waiting[0].arrival_time:
                return []
        block_table = self.pre_alloc_cache_dict.get(request_id, None)
        if not block_table:
            block_table = self.block_manager.get_free_blocks(block_num, token_ids)
            self.pre_alloc_cache_dict[request_id] = block_table
        elif self.block_manager.get_num_free_gpu_blocks() >= block_num:
            block_table.append_token_ids(token_ids)

        if len(block_table.blocks) == self.block_manager.max_block_sliding_window:
            # abort migration due to sliding window
            return []

        return block_table.physical_block_ids[-block_num:]

    def add_running_request(self, backend_request: LlumnixRequest) -> None:
        self._set_status(backend_request, status_to=SequenceStatus.RUNNING)
        self.running.append(backend_request)

    def add_waiting_request(self, backend_request: LlumnixRequest) -> None:
        self._set_status(backend_request, status_to=SequenceStatus.WAITING)
        # pylint: disable=E0203
        arrival_time_list = [request.arrival_time for request in self.waiting]
        idx = bisect.bisect_right(arrival_time_list, backend_request.arrival_time)
        if idx < len(self.waiting):
            self.waiting.insert(idx, backend_request)
        else:
            self.waiting.append(backend_request)

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        if seq_group.status == RequestStatus.WAITING_MIGRATING:
            return AllocStatus.OK
        return super().can_allocate(seq_group)

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        # Change seq status to running, but request status is still waiting_migrating.
        if seq_group.status == RequestStatus.WAITING_MIGRATING:
            # For the waiting request migrated in, blocks have already been allocated when pre alloc.
            self._set_status(seq_group, status_to=SequenceStatus.RUNNING)
            seq_group.reset_status()
        else:
            super()._allocate_and_set_running(seq_group)

    def _set_status(self,
                    seq_group: SequenceGroup,
                    status_to: SequenceStatus,
                    status_from: SequenceStatus = None):
        for seq in seq_group.get_seqs(status=status_from):
            seq.status = status_to

    def free_dst_pre_alloc_cache(self, request_id: str = None) -> None:
        if request_id:
            block_table = self.pre_alloc_cache_dict.pop(request_id, None)
            if block_table:
                block_table.free()
        else:
            # TODO(s5u13b): Only effective with one-to-one migration restriction.
            # Clear all pre-allocated cache of dst instance when src instance encounters exception.
            request_ids = list(self.pre_alloc_cache_dict.keys())
            for req_id in request_ids:
                block_table = self.pre_alloc_cache_dict.pop(req_id, None)
                if block_table:
                    block_table.free()

    def free_src_request(self, backend_request: SequenceGroupLlumnix) -> None:
        seq = backend_request.get_seqs()[0]
        logger.info("free request: {} (seq: {})".format(backend_request.request_id, seq.seq_id))
        self.free_seq(seq)

    def _get_instance_info(self, scheduled_seq_groups: List[SequenceGroupLlumnix]) -> InstanceInfo:
        num_total_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
        num_used_gpu_blocks = num_total_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / num_total_gpu_blocks
        if self.waiting:
            num_blocks_waiting_requests = []
            waiting_time_waiting_requests = []
            for seq_group in self.waiting:
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                num_blocks = num_prompt_tokens / self.cache_config.block_size
                waiting_time = time.time() - seq_group.metrics.arrival_time
                num_blocks_waiting_requests.append(num_blocks)
                waiting_time_waiting_requests.append(waiting_time)
            num_blocks_first_waiting_request = num_blocks_waiting_requests[0]
            waiting_time_first_waiting_request = waiting_time_waiting_requests[0]
            num_blocks_all_waiting_requests = sum(num_blocks_waiting_requests)
        else:
            num_blocks_first_waiting_request = 0
            waiting_time_first_waiting_request = 0
            num_blocks_all_waiting_requests = 0
        instance_info = InstanceInfo(
            num_total_gpu_blocks=num_total_gpu_blocks,
            num_watermark_blocks=self.block_manager.watermark_blocks,
            num_used_gpu_blocks=num_used_gpu_blocks,
            num_free_gpu_blocks=num_free_gpu_blocks,
            gpu_cache_usage=gpu_cache_usage,
            num_running_requests=len(self.running),
            num_waiting_requests=len(self.waiting),
            num_killed_requests=self._get_num_killed_requests(),
            num_blocks_first_waiting_request=num_blocks_first_waiting_request,
            waiting_time_first_waiting_request=waiting_time_first_waiting_request,
            num_blocks_all_waiting_requests=num_blocks_all_waiting_requests,
        )
        for seq_group in scheduled_seq_groups:
            instance_info.running_seq_lens.extend([seq.get_len() for seq in seq_group.get_seqs()])
            instance_info.num_seqs = len(instance_info.running_seq_lens)
        if scheduled_seq_groups:
            instance_info.inference_type = scheduled_seq_groups[-1].inference_type
        # TODO(ZeldaHuang) adapt chunked-prefill
        instance_info.num_batched_tokens = sum([seq_group.request_len for seq_group in scheduled_seq_groups]) \
                                                if instance_info.inference_type == RequestInferenceType.PREFILL \
                                                else len(instance_info.running_seq_lens)
        instance_info.finished_request_ids = [seq_group.request_id for seq_group in self.running if seq_group.finished]
        return instance_info

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        seq_group_metadata_list, scheduler_outputs, allow_async_output_proc = super().schedule()
        self.update_instance_info_callback(self._get_instance_info([scheduled_seq_group.seq_group \
                                            for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups]))
        for seq_group in self.waiting:
            seq_group.try_schedule_times += 1
        return seq_group_metadata_list, scheduler_outputs, allow_async_output_proc

    def _schedule_running(self, *args, **kwargs):
        # pylint: disable=access-member-before-definition
        running_queue = self.running
        filtered_running_queue = deque()
        remove_running = deque()
        for seq_group in running_queue:
            if seq_group.output_len >= seq_group.expected_steps:
                remove_running.extend([seq_group])
            else:
                filtered_running_queue.extend([seq_group])

        self.running = filtered_running_queue
        ret = super()._schedule_running(*args, **kwargs)
        for seq_group in remove_running:
            self.running.extend([seq_group])
        return ret
