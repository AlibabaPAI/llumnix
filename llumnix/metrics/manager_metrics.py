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
from llumnix.utils import is_enable

logger = init_logger(__name__)


class ManagerMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.register = Registery()

        # metrics for dispatch
        self.dispatch_latency = Summary(name="dispatch_latency", registry=self.register)
        self.dispatch_counter = Counter(name="dispatch_counter", registry=self.register)
        self.dispatch_load = Status(name="dispatch_load", registry=self.register)

        self.enable_metrics = is_enable(llumnix_envs.ENABLE_MANAGER_METRICS)
        if self.enable_metrics:
            self.start_metrics_export_loop()
