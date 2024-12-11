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

from llumnix.backends.bladellm.llm_engine import LLMEngineLlumnixMixin
from llumnix.metrics.base_metrics import LlumnixMetrics

class BladellmMetrics(LlumnixMetrics):
    def __init__(self):
        super().__init__()

    def block_manager_init_metrics(self, block_manager: BlockSpaceManager):
        self.num_total_gpu_blocks.observe(block_manager.num_total_gpu_blocks)

    def engine_init_metrics(self, engine: LLMEngineLlumnixMixin):
        self.instance_id.observe(engine.instance_id)

    def scheduler_step_metrics(self, *args, **kwargs):
        ...

    def engine_step_metrics(self, *args, **kwargs):
        ...
