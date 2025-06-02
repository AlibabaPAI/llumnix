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

import threading
import time
from typing import Any, Dict, List

from llumnix.metrics.base_metrics import BaseMetrics
from llumnix.metrics.metrics_types import _REGISTRY, Status, MetricEntry
from llumnix.metrics.exporters import MultiExporter
from llumnix.metrics.dumper import Dumper, DummyDumper
from llumnix import envs as llumnix_envs
from llumnix.instance_info import InstanceInfo
from llumnix.logging.logger import init_logger
from llumnix.utils import is_enable
logger = init_logger(__name__)

class EngineMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
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

        self.dumper: Dumper = None
        self._init_dumper()

        self.enable_metrics = is_enable(llumnix_envs.ENABLE_ENGINE_METRICS)
        if self.enable_metrics:
            self.start_metrics_export_loop()


    def start_metrics_export_loop(self):
        def _worker():
            multi_exporter = MultiExporter()
            while True:
                time.sleep(15)
                metrics: List[MetricEntry] = _REGISTRY.describe()
                label_instance: Dict[str, Any] = {"instance_id": self.instance_id.collect()[0].value}
                for metric in metrics:
                    if metric.labels is None:
                        metric.labels = {}
                    metric.labels.update(label_instance)
                multi_exporter.export(metrics)
                _REGISTRY.reset()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def dump(self):
        self.dumper.dump(_REGISTRY.describe_ignore_labels())

    def to_instance_info(self) -> InstanceInfo:
        return InstanceInfo(
            **{
                metric.name: metric.value
                for metric in _REGISTRY.describe_ignore_labels()
            }
        )

    def _init_dumper(self):
        self.dumper = DummyDumper()
