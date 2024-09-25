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

from asyncio.log import logger
import time
import threading
from typing import Dict, List, Optional, Tuple

from vllm.core.block_manager_v1 import BlockSpaceManagerV1, BlockTable
from vllm.core.scheduler import (Scheduler, PreemptionMode, SequenceStatus, SequenceGroupMetadata, SchedulerOutputs)

from llumnix.instance_info import InstanceInfo
from llumnix.logger import init_logger
from llumnix.llumlet.request import RequestInferenceType
from llumnix.backends.vllm.utils import scheduler_lock
from llumnix.backends.vllm.sequence import SequenceGroupLlumnix

logger = init_logger(__name__)

# TODO(ZeldaHuang): adapt prefix cache and sliding window, now use v1 manager
class BlockManagerLlumnix(BlockSpaceManagerV1):
    def get_free_blocks(self, num_required_blocks: int) -> BlockTable:
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        if (num_free_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return []
        blocks = []
        for _ in range(num_required_blocks):
            block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = 1
            blocks.append(block)
        return blocks

    def add_block_table(self, block_table: BlockTable, seq_id: int) -> None:
        self.block_tables[seq_id] = block_table.copy()

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
        self.scheduler_lock = threading.Lock()
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

    @scheduler_lock
    def get_running_queue(self):
        return self.running

    @scheduler_lock
    def get_all_request_ids(self) -> List[str]:
        request_ids : List[str] = []
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                request_ids.append(seq_group.request_id)
        return request_ids

    @scheduler_lock
    def get_request_incremental_blocks(self, backend_request: SequenceGroupLlumnix, pre_stage_num_blocks: int) -> List[int]:
        seq = backend_request.get_seqs()[0]
        blocks = self.block_manager.get_block_table(seq)
        return blocks[pre_stage_num_blocks:]

    @scheduler_lock
    def remove_running_request(self, request_id: str) -> None:
        for seq_group in self.running:
            if seq_group.request_id == request_id:
                seq = seq_group.get_seqs()[0]
                self.running.remove(seq_group)
                seq.status = SequenceStatus.WAITING
                break

    def add_migrating_out_request_last_stage(self, backend_request: SequenceGroupLlumnix) -> None:
        self.migrating_out_request_last_stage.append(backend_request)

    def remove_migrating_out_request_last_stage(self, backend_request: SequenceGroupLlumnix) -> None:
        self.migrating_out_request_last_stage.remove(backend_request)

    def pop_migrating_out_requests_last_stage(self) -> List[SequenceGroupLlumnix]:
        migrating_out_request_last_stage = self.migrating_out_request_last_stage.copy()
        self.migrating_out_request_last_stage.clear()
        return migrating_out_request_last_stage

    @scheduler_lock
    def pre_alloc(self, request_id: str, block_num: int) -> List[int]:
        blocks = self.block_manager.get_free_blocks(block_num)
        pre_blocks = self.pre_alloc_cache_dict.get(request_id, [])
        pre_blocks.extend(blocks)
        self.pre_alloc_cache_dict[request_id] = pre_blocks
        blocks = [block.block_number for block in blocks]
        return blocks

    @scheduler_lock
    def add_running_request(self, backend_request: SequenceGroupLlumnix) -> None:
        seq = backend_request.get_seqs()[0]
        seq.status = SequenceStatus.RUNNING
        self.running.append(backend_request)

    @scheduler_lock
    def is_request_running(self, backend_request: SequenceGroupLlumnix) -> bool:
        return backend_request in self.running

    @scheduler_lock
    def free_dst_pre_alloc_cache(self, request_id: str = None) -> None:
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

    @scheduler_lock
    def free_src_request(self, backend_request: SequenceGroupLlumnix) -> None:
        seq = backend_request.get_seqs()[0]
        logger.info("free seq {}".format(seq.seq_id))
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
        instance_info.num_batched_tokens = sum([seq_group.request_len for seq_group in scheduled_seq_groups])\
                                            if instance_info.inference_type == RequestInferenceType.PREFILL else len(instance_info.running_seq_lens)
        instance_info.finished_request_ids = [seq_group.request_id for seq_group in self.running if seq_group.is_finished()]
        return instance_info

    @scheduler_lock
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        seq_group_metadata_list, scheduler_outputs = super().schedule()
        self.update_instance_info_callback(self._get_instance_info([scheduled_seq_group.seq_group \
                                            for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups]))
        return seq_group_metadata_list, scheduler_outputs

    def add_seq_group(self, *args, **kwargs):
        # The scheduler lock is mannually released in the end of LLMEngineLlumnix.add_request function.
        # pylint: disable=R1732
        self.scheduler_lock.acquire()
        return super().add_seq_group(*args, **kwargs)

    @scheduler_lock
    def abort_seq_group(self, *args, **kwargs):
        return super().abort_seq_group(*args, **kwargs)
