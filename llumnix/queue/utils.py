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

from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.queue.zmq_server import ZmqServer
from llumnix.queue.ray_queue_server import RayQueueServer
from llumnix.queue.zmq_client import ZmqClient
from llumnix.queue.ray_queue_client import RayQueueClient
from llumnix.queue.queue_type import QueueType
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


def init_request_output_queue_server(zmq_ip: str, zmq_port: int, queue_type: QueueType) -> QueueServerBase:
    output_queue_server: QueueServerBase = None
    if queue_type == QueueType.ZMQ:
        output_queue_server = ZmqServer(zmq_ip, zmq_port)
    else:
        output_queue_server = RayQueueServer()
    return output_queue_server

def init_request_output_queue_client(queue_type: QueueType) -> QueueClientBase:
    output_queue_client: QueueClientBase = None
    if queue_type == QueueType.ZMQ:
        output_queue_client= ZmqClient()
    else:
        output_queue_client = RayQueueClient()
    return output_queue_client
