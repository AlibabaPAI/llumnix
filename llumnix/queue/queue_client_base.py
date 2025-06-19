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
from typing import Any
from collections.abc import Iterable

from llumnix.server_info import ServerInfo
from llumnix import envs as llumnix_envs


class QueueClientBase(ABC):

    def __init__(self):
        self.metric_sample_every_n_records = int(
            llumnix_envs.QUEUE_SERVER_METRICS_SAMPLE_EVERY_N_RECORDS
        )
        self.metric_index = 0

    def is_metric_enable(self):
        return self.metric_sample_every_n_records > 0

    def need_record_latency(self):
        if self.is_metric_enable():
            self.metric_index = (
                self.metric_index + 1
            ) % self.metric_sample_every_n_records
            return self.metric_index == 0
        else:
            return False

    @abstractmethod
    async def put_nowait(self, item: Any, server_info: ServerInfo):
        raise NotImplementedError

    @abstractmethod
    async def put_nowait_batch(self, items: Iterable, server_info: ServerInfo):
        raise NotImplementedError
