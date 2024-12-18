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

from blade_llm.service.block_space_manager import BlockSpaceManager
from blade_llm.service.schedulers.paged_scheduler import PagedScheduler

from llumnix.backends.bladellm.llm_engine import AsyncLLMEngineLlumnixMixin
from llumnix.metrics.base_metrics import LlumnixMetrics
from llumnix.metrics.dumper import LoggerDumper

class BladeLLMMetrics(LlumnixMetrics):
    def _init_dumper(self,):
        self.dumper = LoggerDumper()

    def block_manager_init_metrics(self, block_manager: BlockSpaceManager):
        self.num_total_gpu_blocks.observe(block_manager.num_total_gpu_blocks)
        self.num_watermark_blocks.observe(block_manager.reserved_blocks)

    def engine_init_metrics(self, engine: AsyncLLMEngineLlumnixMixin):
        self.instance_id.observe(engine.instance_id)

    def scheduler_step_metrics(self, scheduler: PagedScheduler):
        block_manager: BlockSpaceManager = scheduler.block_manager
        self.num_used_gpu_blocks.observe(block_manager.get_blocks_usage()*block_manager.num_total_gpu_blocks)
        self.num_running_requests.observe(len(scheduler.running))
        self.num_waiting_requests.observe(len(scheduler.waiting))

        num_blocks_all_waiting_requests = 0
        for gen_group_state in scheduler.waiting:
            num_blocks_all_waiting_requests += sum([page_req.required_blocks for page_req in gen_group_state.paged_reqs])
        self.num_blocks_all_waiting_requests.observe(num_blocks_all_waiting_requests)

        self.dump()

    def engine_step_metrics(self, scheduler: PagedScheduler):
        block_manager: BlockSpaceManager = scheduler.block_manager
        self.num_used_gpu_blocks.observe(block_manager.get_blocks_usage()*block_manager.num_total_gpu_blocks)
        self.num_running_requests.observe(len(scheduler.running))
        self.num_waiting_requests.observe(len(scheduler.waiting))

        num_blocks_all_waiting_requests = 0
        for gen_group_state in scheduler.waiting:
            num_blocks_all_waiting_requests += sum([page_req.required_blocks for page_req in gen_group_state.paged_reqs])
        self.num_blocks_all_waiting_requests.observe(num_blocks_all_waiting_requests)

        self.dump()