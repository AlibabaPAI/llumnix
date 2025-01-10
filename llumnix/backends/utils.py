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

import ray
from ray.util.placement_group import PlacementGroup

from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.queue.queue_type import QueueType
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.queue.utils import init_request_output_queue_client
from llumnix.server_info import ServerInfo
from llumnix.logger import init_logger
from llumnix.utils import get_instance_name
from llumnix.internal_config import MigrationConfig

logger = init_logger(__name__)


class AsyncPutQueueActor:
    def __init__(self, instance_id: str, request_output_queue_type: QueueType):
        self.instance_id = instance_id
        self.request_output_queue_type = request_output_queue_type
        self.request_output_queue_client: QueueClientBase = init_request_output_queue_client(request_output_queue_type)
        self.engine_actor_handle = None

    async def put_nowait_to_servers(self,
                                    server_request_outputs: Dict[str, List],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
        if self.engine_actor_handle is None:
            self.engine_actor_handle = ray.get_actor(get_instance_name(self.instance_id), namespace="llumnix")
        tasks = []
        for server_id, req_outputs in server_request_outputs.items():
            server_info = server_info_dict[server_id]
            for req_output in req_outputs:
                if hasattr(req_output, 'request_timestamps'):
                    req_output.request_timestamps.engine_actor_put_queue_timestamp = time.time()
            tasks.append(asyncio.create_task(self.request_output_queue_client.put_nowait(req_outputs, server_info)))
        rets = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, ret in enumerate(rets):
            if isinstance(ret, Exception):
                server_id = list(server_request_outputs.keys())[idx]
                server_info = server_info_dict[server_id]
                logger.info("server {} is dead".format(server_id))
                if self.request_output_queue_type == QueueType.ZMQ:
                    logger.info("request output queue ip: {}, port: {}".format(server_info.request_output_queue_ip,
                                                                               server_info.request_output_queue_port))
                req_outputs = list(server_request_outputs.values())[idx]
                request_ids = [req_output.request_id for req_output in req_outputs]
                self.engine_actor_handle.abort_request.remote(request_ids)

def init_backend_engine(instance_id: str,
                        placement_group: PlacementGroup,
                        request_output_queue_type: QueueType,
                        migration_config: MigrationConfig,
                        backend_type: BackendType,
                        engine_args,
                        profiling_result_file_path: str = None) -> BackendInterface:
    if backend_type == BackendType.VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.llm_engine import BackendVLLM
        backend_engine = BackendVLLM(instance_id,
                                        placement_group,
                                        request_output_queue_type,
                                        migration_config,
                                        engine_args)
    elif backend_type == BackendType.BLADELLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.bladellm.llm_engine import BackendBladeLLM
        backend_engine = BackendBladeLLM(instance_id,
                                         placement_group,
                                         request_output_queue_type,
                                         migration_config,
                                         engine_args)
    elif backend_type == BackendType.SIM_VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.simulator import BackendSimVLLM
        backend_engine = BackendSimVLLM(instance_id,
                                        placement_group,
                                        request_output_queue_type,
                                        migration_config,
                                        engine_args,
                                        profiling_result_file_path)
    else:
        raise ValueError(f'Unsupported backend: {backend_type}')
    return backend_engine

def get_engine_world_size(engine_args, backend_type: BackendType):
    if backend_type == BackendType.VLLM:
        engine_config = engine_args.create_engine_config()
        world_size = engine_config.parallel_config.world_size
    else: # BLADE_LLM
        world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
    return world_size
