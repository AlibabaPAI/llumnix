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
from llumnix.metrics.variable import Status, PassiveStatus

from llumnix.metrics.base_metrics import LlumnixMetrics
from llumnix.metrics.dumper import LoggerDumper

class BladeLLMMetrics(LlumnixMetrics):
    def __init__(self):
        super().__init__()
        self.num_cached_request_ids = Status("num_cached_request_ids")
        self.num_wait_update_request_ids = PassiveStatus("num_wait_update_request_ids")
        self.num_trans_wrapper_cached_request = PassiveStatus("num_trans_wrapper_cached_request")

    def _init_dumper(self,):
        self.dumper = LoggerDumper()

    def block_manager_init_metrics(self, block_manager: BlockSpaceManager):
        self.num_total_gpu_blocks.observe(block_manager.num_total_gpu_blocks)
        self.num_watermark_blocks.observe(block_manager.reserved_blocks)

    def scheduler_init_metrics(self, scheduler):
        pass

    def engine_init_metrics(self, engine):
        self.instance_id.observe(engine.instance_id)
        self.num_wait_update_request_ids.observe(engine.get_num_wait_update_request_ids)
        self.num_trans_wrapper_cached_request.observe(engine.get_num_trans_wrapper_cached_request)

    def scheduler_step_metrics(self, scheduler):
        block_manager: BlockSpaceManager = scheduler.block_manager
        self.num_used_gpu_blocks.observe(block_manager.get_blocks_usage()*block_manager.num_total_gpu_blocks)
        self.num_running_requests.observe(len(scheduler.running))
        self.num_waiting_requests.observe(len(scheduler.waiting))
        self.num_blocks_all_waiting_requests.observe(scheduler.get_num_blocks_all_waiting_requests())
        self.num_cached_request_ids.observe(scheduler.get_num_cached_request_ids())
        self.num_killed_requests.observe(scheduler.get_num_killed_requests())
        self.num_blocks_first_waiting_request.observe(scheduler.get_num_blocks_first_waiting_request())
        self.num_blocks_last_running_request.observe(scheduler.get_num_blocks_last_running_request())
        self.all_request_ids.observe(scheduler.get_all_request_ids())

        self.dump()

    def engine_step_metrics(self, scheduler):
        block_manager: BlockSpaceManager = scheduler.block_manager
        self.num_used_gpu_blocks.observe(block_manager.get_blocks_usage()*block_manager.num_total_gpu_blocks)
        self.num_running_requests.observe(len(scheduler.running))
        self.num_waiting_requests.observe(len(scheduler.waiting))
        self.num_blocks_all_waiting_requests.observe(scheduler.get_num_blocks_all_waiting_requests())
        self.num_cached_request_ids.observe(scheduler.get_num_cached_request_ids())
        self.num_killed_requests.observe(scheduler.get_num_killed_requests())
        self.num_blocks_first_waiting_request.observe(scheduler.get_num_blocks_first_waiting_request())
        self.num_blocks_last_running_request.observe(scheduler.get_num_blocks_last_running_request())
        self.all_request_ids.observe(scheduler.get_all_request_ids())

        self.dump()
