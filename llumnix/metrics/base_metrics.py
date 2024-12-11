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

from abc import ABC, abstractmethod

from llumnix.metrics.variable import Variable
from llumnix.metrics.dumper import LoggerDumper

class LlumnixMetrics(ABC):
    def __init__(self):
        self.instance_id = Variable("instance_id")

        # used for dispatch
        self.num_total_gpu_blocks = Variable("num_total_gpu_blocks")
        self.num_used_gpu_blocks = Variable("num_used_gpu_blocks")
        self.num_available_gpu_blocks = Variable("num_available_gpu_blocks")
        self.num_blocks_all_waiting_requests = Variable("num_blocks_all_waiting_requests")
        self.num_running_requests = Variable("num_running_requests")
        self.num_waiting_requests = Variable("num_waiting_requests")

        self.dumper = LoggerDumper()

    @abstractmethod
    def block_manager_init_metrics(self, *args, **kwargs):
        ...

    @abstractmethod
    def engine_init_metrics(self, *args, **kwargs):
        ...

    @abstractmethod
    def scheduler_step_metrics(self, *args, **kwargs):
        ...

    @abstractmethod
    def engine_step_metrics(self, *args, **kwargs):
        ...

    