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

from typing import Any
from collections.abc import Iterable
import time

from llumnix.server_info import ServerInfo
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.metrics.timestamps import set_timestamp


class RayQueueClient(QueueClientBase):
    async def put_nowait(self, item: Any, server_info: ServerInfo):
        output_queue = server_info.request_output_queue
        set_timestamp(item, 'queue_client_send_timestamp', time.time())
        return await output_queue.actor.put_nowait.remote(item)

    async def put_nowait_batch(self, items: Iterable, server_info: ServerInfo):
        output_queue = server_info.request_output_queue
        set_timestamp(items, 'queue_client_send_timestamp', time.time())
        return await output_queue.actor.put_nowait_batch.remote(items)
