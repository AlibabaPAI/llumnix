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

from enum import Enum
from typing import Dict
import subprocess
from dataclasses import dataclass
from llumnix.metrics.timestamps import RequestTimestamps

from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


# Put it in utils.py to avoid circular import.
class LaunchMode(str, Enum):
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"


# Use "" type hint to avoid circular import.
class EntrypointsContext:
    def __init__(self,
                 scaler: "Scaler",
                 manager: "Manager",
                 instances: Dict[str, "Llumlet"],
                 request_output_queue: "QueueServerBase",
                 server: "APIServerActor",
                 server_info: "ServerInfo",
                 log_requests: bool,):
        self.scaler = scaler
        self.manager = manager
        self.instances = instances
        self.request_output_queue = request_output_queue
        self.server = server
        self.server_info = server_info
        self.log_requests = log_requests

def is_gpu_available() -> bool:
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    # pylint: disable=bare-except
    except:
        return False


@dataclass
class LlumnixTraceInfo:
    latencys: Dict[str, float] | None = None
    token_timestamps: RequestTimestamps | None = None

    def calc_latency(self):
        if self.token_timestamps is not None:
            self.latencys = self.token_timestamps.to_latency_breakdown_dict()

    def dict(self):
        return {
            "latencys": self.latencys,
            "token_timestamps": self.token_timestamps,
        }
