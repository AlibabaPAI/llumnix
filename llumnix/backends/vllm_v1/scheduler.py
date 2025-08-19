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

from typing import Dict, List, Optional, Tuple, Deque
import time
from collections import defaultdict

import ray

from vllm.utils import Counter
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from llumnix.instance_info import InstanceInfo
from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus
from llumnix.backends.vllm_v1.request import LlumnixRequestVLLMV1
from llumnix.utils import MigrationResponse
from llumnix.ray_utils import get_llumnix_actor_id, LlumnixActor

logger = init_logger(__name__)


class SchedulerLlumnix(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        actor_name = ray.get_runtime_context().get_actor_name()
        self.instance_id = get_llumnix_actor_id(LlumnixActor.INSTANCE, actor_name)
        self.step_counter = Counter()
        self.migrating_out_request_last_stage: Dict[str, Request] = {}

    def add_update_instance_info_callback(self, update_instance_info_callback):
        self.update_instance_info_callback = update_instance_info_callback
        self.update_instance_info_callback(self._get_instance_info())

    def get_running_queue(self) -> List[Request]:
        return self.running

    def get_waiting_queue(self) -> Deque[Request]:
        return self.waiting

    def get_request_incremental_blocks(self, backend_request: LlumnixRequest, pre_stage_num_blocks: int) -> Tuple[List[int], List[int]]:
        raise NotImplementedError("get_request_incremental_blocks is not implemented in vllm v1")

    def remove_running_request(self, request_id: str) -> bool:
        raise NotImplementedError("remove_running_request is not implemented in vllm v1")

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
                # TODO(zhaozhiyu): arrival_time should be accessed in EngineCoreRequest
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
            instance_id=self.instance_id,
            step_id=next(self.step_counter),
            num_total_gpu_blocks=num_total_gpu_blocks, # type: ignore
            num_watermark_blocks=0, # NOTE(zhaozhiyu): there is no watermark_blocks in vllm v1
            num_used_gpu_blocks=num_used_gpu_blocks,
            num_free_gpu_blocks=num_free_gpu_blocks,
            gpu_cache_usage=gpu_cache_usage,
            num_running_requests=len(self.running),
            num_waiting_requests=len(self.waiting),
            num_killed_requests=0, # TODO(zhaozhiyu): num_killed_requests should be obtained from IterationStats
            num_blocks_first_waiting_request=num_blocks_first_waiting_request,
            waiting_time_first_waiting_request=waiting_time_first_waiting_request,
            num_blocks_all_waiting_requests=num_blocks_all_waiting_requests,
            decode_batch_size=sum([req.num_computed_tokens == 1 for req in self.running]),
        )

        if scheduler_output is not None:
            for new_req in scheduler_output.scheduled_new_reqs:
                instance_info.running_seq_lens.append(new_req.num_computed_tokens)
                instance_info.num_seqs = len(instance_info.running_seq_lens)
            # TODO(zhaozhiyu): inference_type is used in simulator, not essential for now
            if scheduler_output.num_scheduled_tokens == {}:
                instance_info.inference_type = RequestInferenceType.UNKNOWN
            elif all(v > 1 for v in scheduler_output.num_scheduled_tokens.values()):
                # all num_scheduler_tokens > 1, prefill
                instance_info.inference_type = RequestInferenceType.PREFILL
            elif all(v == 1 for v in scheduler_output.num_scheduled_tokens.values()):
                # all num_scheduler_token == 1, decode
                instance_info.inference_type = RequestInferenceType.DECODE
            elif all(v >= 1 for v in scheduler_output.num_scheduled_tokens.values()):
                instance_info.inference_type = RequestInferenceType.PREFILL_AND_DECODE
            else:
                instance_info.inference_type = RequestInferenceType.UNKNOWN
            instance_info.num_batched_tokens = scheduler_output.total_num_scheduled_tokens # type: ignore

        instance_info.profiling_data = (
            instance_info.inference_type.value if instance_info.inference_type else "",
            instance_info.num_seqs,
            sum(instance_info.running_seq_lens),
            0.0,
        )

        num_blocks_last_running_request = 0
        reqs: List[Request] = self.running
        if reqs:
            tot_blocks = defaultdict(list)
            for req in reqs:
                if req.status != RequestStatus.RUNNING:
                    continue
                # block_ids (List[List[int]]): A two-level list where
                # the outer list corresponds to KV cache groups
                # each inner list contains the block_ids of the blocks in that group
                block_ids: List[List[int]] = self.kv_cache_manager.get_block_ids(req.request_id)
                for group_id, group in enumerate(block_ids):
                    tot_blocks[group_id].extend(group)
            for group_id, group in tot_blocks.items():
                num_blocks_last_running_request += len(set(group))
            instance_info.num_blocks_last_running_request = num_blocks_last_running_request

        instance_info.timestamp = time.time()
        
        return instance_info

    def schedule(self) -> SchedulerOutput:
        scheduler_output = super().schedule()
        self.update_instance_info_callback(self._get_instance_info(scheduler_output))
        return scheduler_output
