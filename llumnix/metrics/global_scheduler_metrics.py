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

from llumnix.metrics.base_metrics import BaseMetrics
from llumnix.metrics.metrics_types import Counter, Registery, Summary, Status
from llumnix import envs as llumnix_envs
from llumnix.logging.logger import init_logger
from llumnix.metrics.utils import is_metrics_enabled

logger = init_logger(__name__)


class GlobalSchedulerMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.register = Registery()
        self.metrics_sampling_interval = int(llumnix_envs.GLOBAL_SCHEDULER_METRICS_SAMPLE_EVERY_N_RECORDS)

        # metrics for dispatch
        self.dispatch_latency = Summary(
            name="dispatch_latency",
            registry=self.register,
            metrics_sampling_interval=self.metrics_sampling_interval,
        )
        self.dispatch_counter = Counter(
            name="dispatch_counter",
            registry=self.register,
            metrics_sampling_interval=self.metrics_sampling_interval
        )
        self.dispatch_load = Status(
            name="dispatch_load",
            registry=self.register,
            metrics_sampling_interval=self.metrics_sampling_interval,
        )

        self.enable_metrics = is_metrics_enabled(
            llumnix_envs.GLOBAL_SCHEDULER_METRICS_SAMPLE_EVERY_N_RECORDS
        )
        if self.enable_metrics:
            self.start_metrics_export_loop()
