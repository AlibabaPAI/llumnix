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

import json
import urllib
from abc import ABC
from typing import List
import numpy as np

from llumnix.logging.logger import init_logger
from llumnix import envs as llumnix_envs
from llumnix.metrics.metrics_types import MetricEntry

logger = init_logger(__name__)


class _BaseExporter(ABC):
    def export(self, metrics: List[MetricEntry]):
        raise NotImplementedError


class LoggerExporter(_BaseExporter):
    """Export metric info to logger."""

    def export(self, metrics: List[MetricEntry]):
        logger.info("Logger exporter: {}".format(metrics))


class EASExporter(_BaseExporter):
    """Export metric info to EAS sidecar."""

    FAILURE_LIMIT = 5

    def __init__(self, url: str = ""):
        self._url = url or "http://localhost:8080/api/builtin/realtime_metrics"
        self._failed_cnt = 0

    def export(self, metrics: List[MetricEntry]):
        if self._failed_cnt >= EASExporter.FAILURE_LIMIT:
            return
        try:
            data = [
                {
                    "name": f"llumnix_{metric.name}",
                    "tags": metric.labels if metric.labels else {},
                    "value": metric.value if metric.value != -np.inf else -9999,
                }
                for metric in metrics if metric.value is not None and isinstance(metric.value, (int, float))
            ]
            encoded_data = json.dumps(data).encode("utf-8")
            logger.debug("Export metrics to eas: {}".format(data))
            request = urllib.request.Request(
                self._url, data=encoded_data, method="POST"
            )
            response = urllib.request.urlopen(request)
            _ = response.read()
        except Exception as e:
            self._failed_cnt += 1
            logger.warning("Failed to export metric to EAS sidecar, error: {}", e)
            if self._failed_cnt == EASExporter.FAILURE_LIMIT:
                logger.warning(
                    "Stop exporting metric to EAS sidecar due to too many failures."
                )


class MultiExporter:
    def __init__(self):
        self.exporters: List[_BaseExporter] = []
        exporter_names = llumnix_envs.METRICS_OUTPUT_TARGET.split(",")
        if "logger" in exporter_names:
            self.exporters.append(LoggerExporter())
        if "eas" in exporter_names:
            self.exporters.append(EASExporter())

    def export(self, metrics: List[MetricEntry]):
        for exporter in self.exporters:
            exporter.export(metrics)
