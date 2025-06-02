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

from abc import ABC
import threading
import time
from typing import List

from llumnix.metrics.metrics_types import MetricEntry
from llumnix.metrics.exporters import MultiExporter
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


class BaseMetrics(ABC):
    def __init__(self):
        pass

    def start_metrics_export_loop(self):
        logger.info("Starting {} export loop".format(self.__class__.__name__))

        def _worker():
            multi_exporter = MultiExporter()
            while True:
                time.sleep(15)
                metrics: List[MetricEntry] = self.register.describe()
                multi_exporter.export(metrics)
                self.register.reset()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
