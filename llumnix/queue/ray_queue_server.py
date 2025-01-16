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
import ray
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.metrics.timestamps import set_timestamp


class RayQueueServer(QueueServerBase):
    def __init__(self) -> None:
        self.queue = RayQueue(
            actor_options={
                "scheduling_strategy":
                    NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False
                    )
            }
        )

    async def get(self):
        item = await self.queue.actor.get.remote()
        set_timestamp(item, 'queue_server_receive_timestamp', time.time())
        return item

    async def get_nowait_batch(self):
        qsize = await self.queue.actor.qsize.remote()
        items = await self.queue.actor.get_nowait_batch.remote(qsize)
        set_timestamp(items, 'queue_server_receive_timestamp', time.time())
        return items

    async def run_server_loop(self):
        pass

    def cleanup(self):
        try:
            ray.kill(self.queue)
        # pylint: disable=broad-except, unused-variable
        except Exception as e:
            pass
