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
from llumnix.metrics.queue_server_metrics import QueueServerMetrics


class QueueServerBase(ABC):
    def __init__(self):
        self.port: int = None
        self.queue_server_metrics = QueueServerMetrics()

    @abstractmethod
    async def get(self, timeout):
        raise NotImplementedError

    @abstractmethod
    async def run_server_loop(self):
        raise NotImplementedError

    @abstractmethod
    def cleanup(self):
        raise NotImplementedError
