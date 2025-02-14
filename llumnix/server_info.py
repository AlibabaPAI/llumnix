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

from llumnix.queue.ray_queue_server import RayQueueServer
from llumnix.queue.queue_type import QueueType


class ServerInfo:
    def __init__(self,
                 server_id: str,
                 request_output_queue_type: QueueType,
                 request_output_queue: RayQueueServer,
                 request_output_queue_ip: str,
                 request_output_queue_port: int) -> None:
        self.server_id = server_id
        self.request_output_queue_type = request_output_queue_type
        if request_output_queue_type == QueueType.RAYQUEUE:
            assert request_output_queue is not None
        self.request_output_queue = request_output_queue.queue if request_output_queue_type == QueueType.RAYQUEUE else None
        self.request_output_queue_ip = request_output_queue_ip
        self.request_output_queue_port = request_output_queue_port
