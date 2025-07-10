
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

import copy
from typing import Optional
from ray.util.queue import Queue as RayQueue

from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.queue.queue_type import QueueType
from llumnix.server_info import ServerInfo


class RequestProcessingContext():
    def __init__(
        self,
        server_id: str,
        request_output_queue_type: QueueType,
        request_output_queue: RayQueue,
        request_output_queue_ip: str,
        request_output_queue_port: int,
        enable_trace: bool = False,
    ) -> None:
        self.server_id: str = server_id
        self.request_output_queue_type: QueueType = request_output_queue_type
        self.request_output_queue: RayQueue = request_output_queue
        self.request_output_queue_ip: str = request_output_queue_ip
        self.request_output_queue_port: int = request_output_queue_port
        self.enable_trace: bool = enable_trace
        self.trace_timeline: Optional[RequestTimestamps] = RequestTimestamps() if enable_trace else None

    @classmethod
    def deepcopy_from_server_info(
        cls, server_info: ServerInfo, enable_trace: bool = False
    ) -> "RequestProcessingContext":
        server_info_copy: ServerInfo = copy.deepcopy(server_info)
        return cls.from_server_info(server_info_copy, enable_trace)

    @classmethod
    def from_server_info(
        cls, server_info: ServerInfo, enable_trace: bool = False
    ) -> "RequestProcessingContext":
        return cls(
            server_id=server_info.server_id,
            request_output_queue_type=server_info.request_output_queue_type,
            request_output_queue=server_info.request_output_queue,
            request_output_queue_ip=server_info.request_output_queue_ip,
            request_output_queue_port=server_info.request_output_queue_port,
            enable_trace=enable_trace,
        )

    def add_trace_timeline(self, timestamp_attr: str, timestamp: float = None):
        if not self.enable_trace:
            return
        self.trace_timeline.set_timestamp(timestamp_attr, timestamp)

    def get_server_info(self) -> ServerInfo:
        return ServerInfo(
            server_id=self.server_id,
            request_output_queue_type=self.request_output_queue_type,
            request_output_queue=self.request_output_queue,
            request_output_queue_ip=self.request_output_queue_ip,
            request_output_queue_port=self.request_output_queue_port
        )
