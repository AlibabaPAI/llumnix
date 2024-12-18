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

from blade_llm.service.schedulers import PagedScheduler
from blade_llm.service.scheduler_types import SchedulerStepOutput
from blade_llm.service.args import ServingArgs

from llumnix.backends.bladellm.metrics import BladeLLMMetrics

class SchedulerLlumnixMixin:
    def __init__(self):
        self.llumnix_metrics = BladeLLMMetrics()

class PagedSchedulerLlumnix(PagedScheduler, SchedulerLlumnixMixin):
    def __init__(self, serving_args: ServingArgs, *args, **kwargs) -> None:
        PagedScheduler.__init__(self, serving_args, *args, **kwargs)
        SchedulerLlumnixMixin.__init__(self)
        self.llumnix_metrics.block_manager_init_metrics(self.block_manager)

    def step(self) -> SchedulerStepOutput:
        step_out = super().step()
        self.llumnix_metrics.scheduler_step_metrics(self)
        return step_out
