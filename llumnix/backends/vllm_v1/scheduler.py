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

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from llumnix.instance_info import InstanceInfo
from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus
from llumnix.backends.vllm_v1.request import LlumnixRequestVLLMV1
from llumnix.utils import MigrationResponse

logger = init_logger(__name__)


class SchedulerLlumnix(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.migrating_out_request_last_stage: Dict[str, Request] = {}

    def add_update_instance_info_callback(self, update_instance_info_callback):
        self.update_instance_info_callback = update_instance_info_callback
        self.update_instance_info_callback(self._get_instance_info())
    
    def get_running_queue(self) -> List[Request]:
        return self.running

    def get_waiting_queue(self) -> Deque[Request]:
        return self.waiting

    # TODO(zhaozhiyu): adapt to vllm v1
    def get_request_incremental_blocks(self, backend_request: LlumnixRequest, pre_stage_num_blocks: int) -> Tuple[List[int], List[int]]:
        raise NotImplementedError("get_request_incremental_blocks is not implemented in vllm v1")

    # TODO(zhaozhiyu): adapt to vllm v1
    def remove_running_request(self, request_id: str) -> bool:
        raise NotImplementedError("remove_running_request is not implemented in vllm v1")

    # TODO(zhaozhiyu): adapt to vllm v1
    def remove_waiting_request(self, request_id: str) -> bool:
        raise NotImplementedError("remove_waiting_request is not implemented in vllm v1")

    def add_migrating_out_request_last_stage(self, backend_request: LlumnixRequestVLLMV1) -> None:
        self.migrating_out_request_last_stage[backend_request.request_id] = backend_request

    def pop_migrating_out_request_last_stage(self, request_id: str) -> None:
        raise NotImplementedError("pop_migrating_out_request_last_stage is not implemented in vllm v1")

    def pre_alloc_cache(self,
                        request_id: str,
                        request_status: RequestStatus,
                        request_arrival_time: float,
                        block_num: int,
                        token_ids: List[int]) -> MigrationResponse:
        raise NotImplementedError("pre_alloc_cache is not implemented in vllm v1")

    def add_running_request(self, backend_request: LlumnixRequest) -> None:
        raise NotImplementedError("add_running_request is not implemented in vllm v1")
        # self._set_status(backend_request, status_to=SequenceStatus.RUNNING)
        # self.running.append(backend_request)

    def add_waiting_request(self, backend_request: LlumnixRequest) -> None:
        raise NotImplementedError("add_waiting_request is not implemented in vllm v1")

    def _allocate_and_set_running(self, req: Request) -> None:
        raise NotImplementedError("_allocate_and_set_running is not implemented in vllm v1.")

    def _set_status(self,
                    req: Request,
                    status_to: RequestStatus,
                    status_from: Optional[RequestStatus] = None):
        raise NotImplementedError("_set_status is not implemented in vllm v1")

    def free_pre_alloc_cache(self, request_id: str) -> None:
        raise NotImplementedError("free_pre_alloc_cache is not implemented in vllm v1")

    def free_src_request(self, backend_request: LlumnixRequestVLLMV1) -> None:
        raise NotImplementedError("free_src_request is not implemented in vllm v1")

    # TODO(zhaozhiyu): update waiting_time, num_watermark_blocks, inference_type
    def _get_instance_info(self, scheduler_output: Optional[SchedulerOutput] = None) -> InstanceInfo:
        num_total_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        num_used_gpu_blocks = num_total_gpu_blocks - num_free_gpu_blocks # type: ignore
        gpu_cache_usage = num_used_gpu_blocks / num_total_gpu_blocks # type: ignore
        if self.waiting:
            num_blocks_waiting_requests = []
            waiting_time_waiting_requests = []
            for req in self.waiting:
                num_prompt_tokens = req.num_prompt_tokens
                num_blocks = num_prompt_tokens / self.cache_config.block_size
                # FIXME(zhaozhiyu): arrival_time is not accessible through req, should be acessed in output_processor
                waiting_time = 0
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
            num_total_gpu_blocks=num_total_gpu_blocks, # type: ignore
            num_watermark_blocks=0, # NOTE(zhaozhiyu): there is no watermark_blocks in vllm v1
            num_used_gpu_blocks=num_used_gpu_blocks,
            num_free_gpu_blocks=num_free_gpu_blocks,
            gpu_cache_usage=gpu_cache_usage,
            num_running_requests=len(self.running),
            num_waiting_requests=len(self.waiting),
            num_killed_requests=0, # num_killed_requests should be obtained from IterationStats
            num_blocks_first_waiting_request=num_blocks_first_waiting_request,
            waiting_time_first_waiting_request=waiting_time_first_waiting_request,
            num_blocks_all_waiting_requests=num_blocks_all_waiting_requests,
        )
        
        if scheduler_output is not None:
            for new_req in scheduler_output.scheduled_new_reqs:
                instance_info.running_seq_lens.append(new_req.num_computed_tokens)
                instance_info.num_seqs = len(instance_info.running_seq_lens)
            # FIXME(zhaozhiyu) figure out how vllm v1 determine the inference type
            instance_info.inference_type = RequestInferenceType.UNKNOWN
            instance_info.num_batched_tokens = scheduler_output.total_num_scheduled_tokens # type: ignore
        return instance_info

    # TODO(zhaozhiyu): adapt vllm v1, remove sequence group
    def schedule(self) -> SchedulerOutput:
        scheduler_output = super().schedule()
        self.update_instance_info_callback(self._get_instance_info(scheduler_output))
        return scheduler_output
