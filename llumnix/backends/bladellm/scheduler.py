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

from blade_llm.service.paged_utils import BlockSpaceManager, PreemptionMode
from blade_llm.service.proto.bladellm_pb2 import BlockTable
from blade_llm.service.scheduler import PagedScheduler
from blade_llm.service.scheduler_types import SchedulerStepOutput

from llumnix.instance_info import InstanceInfo
from llumnix.logger import init_logger
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType
from llumnix.backends.bladellm.utils import scheduler_lock

logger = init_logger(__name__)

class BlockManagerLlumnix(BlockSpaceManager):
    def get_free_blocks(self, num_required_blocks: int) -> BlockTable:
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        if num_free_gpu_blocks - num_required_blocks < self.reserved_blocks:
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

class SchedulerLlumnix(PagedScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_manager = BlockManagerLlumnix(
            block_size=self.block_size,
            num_gpu_blocks=self._max_processing_units,
            num_cpu_blocks=self.cpu_blocks,
            block_reserved_percent=self.block_reserved_percent,
            gamma=self.gamma_blank, # gamma step for speculative decoding, lookahead etc
            disable_prompt_cache=args.disable_prompt_cache,
            prompt_cache_enable_swap=args.prompt_cache_enable_swap,
        )
        self.pre_alloc_cache_dict: Dict[str, BlockTable] = {}
        self.scheduler_lock = threading.Lock()
        self.migrating_out_request_last_stage: List[LlumnixRequest] = []
    
    def add_update_instance_info_callback(self, update_instance_info_callback):
        self.update_instance_info_callback = update_instance_info_callback
        self.update_instance_info_callback(self._get_instance_info())

    def _preempt(
        self,
        seq_group: LlumnixRequest,
        insert=True,
    ) -> PreemptionMode:
        seq_group.last_preemption_time = time.time()
        return super()._preempt(seq_group, insert)
    
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
        for state_queue in [self.waiting, self.running, self.swapped, self.prefill]:
            for seq_group in state_queue:
                request_ids.append(seq_group.request_group_id)
        return request_ids

    @scheduler_lock
    def get_request_incremental_blocks(self, backend_request: LlumnixRequest, pre_stage_num_blocks: int) -> List[int]:
        seq = backend_request.paged_reqs[0]
        blocks = self.block_manager.get_block_table(seq)
        return blocks[pre_stage_num_blocks:]

    @scheduler_lock
    def remove_running_request(self, request_id: str) -> None:
        for seq_group in self.running:
            if seq_group.request_group_id == request_id:
                self.running.remove(seq_group)
                break

    def add_migrating_out_request_last_stage(self, backend_request: LlumnixRequest) -> None:
        self.migrating_out_request_last_stage.append(backend_request)

    def remove_migrating_out_request_last_stage(self, backend_request: LlumnixRequest) -> None:
        self.migrating_out_request_last_stage.remove(backend_request)

    def pop_migrating_out_requests_last_stage(self) -> List[LlumnixRequest]:
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
    def add_running_request(self, backend_request: LlumnixRequest) -> None:
        self.running.append(backend_request)

    @scheduler_lock
    def is_request_running(self, backend_request: LlumnixRequest) -> bool:
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
    def free_src_request(self, backend_request: LlumnixRequest) -> None:
        self._free_req(backend_request)

    def _get_instance_info(self) -> InstanceInfo:
        num_total_gpu_blocks = self._max_processing_units
        num_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
        num_used_gpu_blocks = num_total_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / num_total_gpu_blocks
        if self.waiting:
            num_blocks_waiting_requests = []
            waiting_time_waiting_requests = []
            for seq_group in self.waiting:
                num_prompt_tokens = len(seq_group.paged_reqs[0].token_ids)
                num_blocks = num_prompt_tokens / self.cache_config.block_size
                waiting_time = time.monotonic() - seq_group.receive_time
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
            num_watermark_blocks=self.block_manager.reserved_blocks,
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
        for seq_group in self.running:
            instance_info.running_seq_lens.extend([len(seq.token_ids) for seq in seq_group.paged_reqs])
            instance_info.num_seqs = len(instance_info.running_seq_lens)
        if self.running:
            # TODO(xinyi): inference_type
            instance_info.inference_type = self.running[-1].inference_type
        # TODO(ZeldaHuang) adapt chunked-prefill
        instance_info.num_batched_tokens = sum([seq_group.request_len for seq_group in self.running])\
                                            if instance_info.inference_type == RequestInferenceType.PREFILL else len(instance_info.running_seq_lens)
        # TODO(xinyi): add in llm engine
        # instance_info.finished_request_ids = [seq_group.request_id for seq_group in self.running if seq_group.is_finished()]
        return instance_info

    @scheduler_lock
    def step(self) -> SchedulerStepOutput:
        scheduler_outputs = super().step()
        self.update_instance_info_callback(self._get_instance_info())
        return scheduler_outputs

    def add_request(self, *args, **kwargs):
        # The scheduler lock is mannually released in the end of LLMEngineLlumnix.add_request function.
        # pylint: disable=R1732
        self.scheduler_lock.acquire()
        return super().add_request(*args, **kwargs)

    @scheduler_lock
    def abort_seq_group(self, *args, **kwargs):
        return super().drop_request(*args, **kwargs)
