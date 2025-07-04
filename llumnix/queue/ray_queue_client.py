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
from llumnix.constants import RAY_QUEUE_RPC_TIMEOUT
from llumnix.utils import asyncio_wait_for_ray_remote_call_with_timeout, is_traced_request


class RayQueueClient(QueueClientBase):
    async def put_nowait(self, item: Any, server_info: ServerInfo):
        output_queue = server_info.request_output_queue
        need_record_trace_timestamp = False
        if isinstance(item, list):
            for obj in item:
                if is_traced_request(obj):
                    need_record_trace_timestamp = True
                    break
        else:
            need_record_trace_timestamp = is_traced_request(item)
        send_time = (
            time.perf_counter()
            if need_record_trace_timestamp or self.need_record_latency_metric()
            else None
        )
        return await asyncio_wait_for_ray_remote_call_with_timeout(
            output_queue.actor.put_nowait, (item, send_time),
            timeout=RAY_QUEUE_RPC_TIMEOUT
        )

    async def put_nowait_batch(self, items: Iterable, server_info: ServerInfo):
        output_queue = server_info.request_output_queue
        send_time = (
            time.perf_counter()
            if self.need_record_latency_metric() or is_traced_request(server_info)
            else None
        )
        items_with_send_time = [(item, send_time) for item in items]
        return await asyncio_wait_for_ray_remote_call_with_timeout(
            output_queue.actor.put_nowait_batch, items_with_send_time,
            timeout=RAY_QUEUE_RPC_TIMEOUT
        )
