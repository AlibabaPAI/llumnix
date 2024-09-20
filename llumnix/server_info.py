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

class ServerInfo:
    def __init__(self,
                 server_id: str,
                 output_queue_type: str,
                 request_output_queue: RayQueueServer,
                 request_output_queue_ip: str,
                 request_output_queue_port: int) -> None:
        self.server_id = server_id
        self.output_queue_type = output_queue_type

        if output_queue_type == "rayqueue":
            assert request_output_queue is not None and hasattr(request_output_queue, "queue")
        self.request_output_queue = request_output_queue.queue if hasattr(request_output_queue, "queue") else None

        self.request_output_queue_ip = request_output_queue_ip
        self.request_output_queue_port = request_output_queue_port
