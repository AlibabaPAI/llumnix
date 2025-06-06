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

from typing import Dict, List
import asyncio
import time
import os

import ray
from ray.util.placement_group import PlacementGroup

from llumnix.arg_utils import LlumnixEngineArgs, InstanceArgs
from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.queue.queue_type import QueueType
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.queue.utils import init_request_output_queue_client
from llumnix.server_info import ServerInfo
from llumnix.logging.logger import init_logger
from llumnix.ray_utils import get_instance_name, log_actor_ray_info
from llumnix.metrics.timestamps import set_timestamp
from llumnix.utils import asyncio_wait_for_with_timeout
from llumnix.request_output import LlumnixRequestOuput

logger = init_logger(__name__)


class AsyncPutQueueActor:
    def __init__(self, instance_id: str, request_output_queue_type: QueueType, backend_type: BackendType):
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.instance_id = instance_id
        self.request_output_queue_type = request_output_queue_type
        self.request_output_queue_client: QueueClientBase = init_request_output_queue_client(request_output_queue_type)
        self.engine_actor_handle = None
        self.backend_type = backend_type

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]})"

    async def put_nowait_to_servers(self,
                                    server_request_outputs: Dict[str, List[LlumnixRequestOuput]],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
        if self.engine_actor_handle is None:
            # The lifetime of AsyncPutQueueActor is the same as the lifetime of the instance actor,
            # so we do not handling exception here.
            self.engine_actor_handle = ray.get_actor(get_instance_name(self.instance_id), namespace="llumnix")
        tasks = []
        for server_id, req_outputs in server_request_outputs.items():
            server_info = server_info_dict[server_id]
            set_timestamp(req_outputs, 'engine_actor_put_queue_timestamp', time.time())
            tasks.append(asyncio.create_task(self.request_output_queue_client.put_nowait(req_outputs, server_info)))
        rets = await asyncio.gather(*tasks, return_exceptions=True)
        request_ids = []
        for idx, ret in enumerate(rets):
            # If exception occurs during sending message from queue client to queue server,
            # AsyncPutQueueActor will not die.
            if isinstance(ret, Exception):
                server_id = list(server_request_outputs.keys())[idx]
                server_info = server_info_dict[server_id]
                logger.error("Server {} is dead, exception: {}.".format(server_id, ret))
                if self.request_output_queue_type == QueueType.ZMQ:
                    logger.debug("request output queue ip: {}, port: {}".format(server_info.request_output_queue_ip,
                                                                                server_info.request_output_queue_port))
                req_outputs = list(server_request_outputs.values())[idx]
                request_ids = []
                for req_output in req_outputs:
                    assert isinstance(req_output, LlumnixRequestOuput)
                    request_ids.append(req_output.request_id)
                await asyncio_wait_for_with_timeout(self.engine_actor_handle.abort.remote(request_ids))

    def stop(self):
        if self.request_output_queue_type == QueueType.ZMQ:
            self.request_output_queue_client.close()

def init_backend_engine(instance_id: str,
                        placement_group: PlacementGroup,
                        request_output_queue_type: QueueType,
                        instance_args: InstanceArgs,
                        llumnix_engine_args: LlumnixEngineArgs) -> BackendInterface:
    backend_type = llumnix_engine_args.backend_type
    if backend_type == BackendType.VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.llm_engine import BackendVLLM
        backend_engine = BackendVLLM(instance_id,
                                     placement_group,
                                     request_output_queue_type,
                                     instance_args,
                                     llumnix_engine_args)
    elif backend_type == BackendType.BLADELLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.bladellm.llm_engine import BackendBladeLLM
        backend_engine = BackendBladeLLM(instance_id,
                                         placement_group,
                                         request_output_queue_type,
                                         instance_args,
                                         llumnix_engine_args)
    elif backend_type == BackendType.SIM_VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.sim_llm_engine import BackendSimVLLM
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        backend_engine = BackendSimVLLM(instance_id,
                                        placement_group,
                                        request_output_queue_type,
                                        instance_args,
                                        llumnix_engine_args)
    else:
        raise ValueError(f'Unsupported backend: {backend_type}')
    return backend_engine
