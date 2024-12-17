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

from llumnix.metrics.variable import _REGISTRY, Status
from llumnix.metrics.dumper import Dumper, DummyDumper
from llumnix.instance_info import InstanceInfo


class LlumnixMetrics(ABC):
    def __init__(self):
        self.instance_id = Status("instance_id")

        # used for dispatch and migration
        self.num_total_gpu_blocks = Status("num_total_gpu_blocks")
        self.num_used_gpu_blocks = Status("num_used_gpu_blocks")
        self.num_running_requests = Status("num_running_requests")
        self.num_waiting_requests = Status("num_waiting_requests")

        # used for dispatch
        self.num_blocks_all_waiting_requests = Status("num_blocks_all_waiting_requests")

        # used for migration
        self.num_blocks_last_running_request = Status("num_blocks_last_running_request")
        self.num_blocks_first_waiting_request = Status("num_blocks_first_waiting_request")

        # stastics
        self.num_watermark_blocks = Status("num_watermark_blocks")
        self.num_killed_requests = Status("num_killed_requests")
        self.all_request_ids = Status("all_request_ids")

        self.dumper: Dumper = None
        self._init_dumper()

    def dump(self):
        self.dumper.dump(_REGISTRY.describe_all())

    def to_instance_info(self) -> InstanceInfo:
        return InstanceInfo(**(_REGISTRY.describe_all()))

    def _init_dumper(self,):
        self.dumper = DummyDumper()

    @abstractmethod
    def block_manager_init_metrics(self, block_manager):
        ...

    @abstractmethod
    def engine_init_metrics(self, engine):
        ...

    @abstractmethod
    def scheduler_init_metrics(self, scheduler):
        ...

    @abstractmethod
    def scheduler_step_metrics(self, scheduler):
        ...

    @abstractmethod
    def engine_step_metrics(self, scheduler):
        ...
    