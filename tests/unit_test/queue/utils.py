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

from llumnix.utils import random_uuid
from llumnix.server_info import ServerInfo
from llumnix.queue.utils import init_request_output_queue_server, QueueType

def request_output_queue_server(request_output_queue_type: QueueType):
    ip = '127.0.0.1'
    port = 1234
    output_queue = init_request_output_queue_server(ip, port, request_output_queue_type)
    server_id = random_uuid()
    server_info = ServerInfo(server_id, request_output_queue_type, output_queue, ip, port)
    return output_queue, server_info
