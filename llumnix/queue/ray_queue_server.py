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

import time
from typing import Iterable

import ray
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.request_output import LlumnixRequestOuput
from llumnix.utils import random_uuid


# TODO(KuilongCui): make rayqueueserver and api server stay in the same node
# for local launch mode
class RayQueueServer(QueueServerBase):
    def __init__(self) -> None:
        super().__init__()
        self.queue = RayQueue(
            actor_options={
                "name": random_uuid(),
                "scheduling_strategy":
                    NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False
                    )
            }
        )

    async def get(self, timeout=None):
        # Server call blocking get to wait for request output tokens.
        item, send_time = await self.queue.actor.get.remote(timeout=timeout)
        if send_time:
            self.queue_server_metrics.queue_trans_latency.observe(
                (time.perf_counter() - send_time) * 1000
            )
        # process trace info
        obj_list = [item] if not isinstance(item, Iterable) else item
        for obj in obj_list:
            if isinstance(obj, LlumnixRequestOuput):
                obj.request_processing_context.add_trace_timeline('queue_client_send_timestamp', send_time)
                obj.request_processing_context.add_trace_timeline('queue_server_receive_timestamp')
        return item

    async def get_nowait_batch(self):
        qsize = await self.queue.actor.qsize.remote()
        items = await self.queue.actor.get_nowait_batch.remote(qsize)
        return items

    async def run_server_loop(self):
        pass

    def cleanup(self):
        try:
            ray.kill(self.queue)
        # pylint: disable=bare-except
        except:
            pass
