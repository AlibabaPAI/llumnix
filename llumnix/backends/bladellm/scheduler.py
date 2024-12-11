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
from typing import List

from loguru import logger

from blade_llm.service.proto.bladellm_pb2 import WorkerStepRequest
from blade_llm.service.schedulers import PagedScheduler
from blade_llm.service.scheduler_types import SchedulerStepOutput
from blade_llm.service.args import ServingArgs

from llumnix.instance_info import InstanceInfo
from llumnix.llumlet.request import RequestInferenceType
from llumnix.backends.bladellm.metrics import BladellmMetrics

class PagedSchedulerLlumnix(PagedScheduler, BladellmMetrics):
    def __init__(self, serving_args: ServingArgs, *args, **kwargs) -> None:
        super().__init__(serving_args, *args, **kwargs)
        self.block_manager_init_metrics(self.block_manager)

    def _get_instance_info(self, steps: List[WorkerStepRequest]) -> InstanceInfo:
        num_total_gpu_blocks = self._max_processing_units
        num_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
        num_used_gpu_blocks = num_total_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / num_total_gpu_blocks
        if self.waiting:
            num_blocks_waiting_requests = []
            waiting_time_waiting_requests = []
            for seq_group in self.waiting:
                num_prompt_tokens = len(seq_group.paged_reqs[0].token_ids)
                num_blocks = num_prompt_tokens / self.block_size
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

        for gen_group in self.running:
            instance_info.running_seq_lens.extend([len(req_state.token_ids) for req_state in gen_group.paged_reqs])
            instance_info.num_seqs = len(instance_info.running_seq_lens)

        instance_info.inference_type = RequestInferenceType.generate_inference_type(
            exist_prefill=any(len(step.prefill) > 0 for step in steps),
            exist_decode=any(len(step.decode) > 0 for step in steps))
        instance_info.num_batched_tokens = sum([
            len(step.decode) + sum([len(prefill.prompt_tokens) for prefill in step.prefill]) for step in steps
        ])
        instance_info.finished_request_ids = len(self._finished_req_to_remove)
        logger.info("update in scheduler {}".format(instance_info.num_running_requests))
        return instance_info
        
    def step(self) -> SchedulerStepOutput:
        step_out = super().step()
        self.scheduler_step_metrics(self.running, self.waiting, step_out)
        return step_out
