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
import traceback
import asyncio

import ray
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.queue.utils import QueueType, init_output_queue_client
from llumnix.logger import init_logger

logger = init_logger(__name__)


class AsyncPutQueueActor:
    def __init__(self, instance_id, output_queue_type: QueueType):
        self.instance_id = instance_id
        self.output_queue_type = output_queue_type
        self.request_output_queue_client: QueueClientBase = init_output_queue_client(output_queue_type)
        self.engine_actor_handle = None

    async def put_nowait_to_servers(self,
                                    server_request_outputs,
                                    server_info_dict) -> None:
        try:
            if self.engine_actor_handle is None:
                self.engine_actor_handle = ray.get_actor("instance_{}".format(self.instance_id), namespace="llumnix")
            tasks = []
            for server_id, req_outputs in server_request_outputs.items():
                server_info = server_info_dict[server_id]
                for req_output in req_outputs:
                    if hasattr(req_output, 'request_timestamps'):
                        req_output.request_timestamps.engine_actor_put_queue_timestamp = time.time()
                tasks.append(asyncio.create_task(self.request_output_queue_client.put_nowait(req_outputs, server_info)))
            rets = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, ret in enumerate(rets):
                if isinstance(ret, (TimeoutError, ray.exceptions.RayActorError)):
                    server_id = list(server_request_outputs.keys())[idx]
                    server_info = server_info_dict[server_id]
                    logger.info("Server {} is dead".format(server_id))
                    if self.output_queue_type == QueueType.ZMQ:
                        logger.info("request output queue ip: {}, port: {}".format(server_info.request_output_queue_ip,
                                                                                server_info.request_output_queue_port))
                    req_outputs = list(server_request_outputs.values())[idx]
                    request_ids = [req_output.request_id for req_output in req_outputs]
                    self.engine_actor_handle.abort.remote(request_ids)
        # pylint: disable=W0703
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
