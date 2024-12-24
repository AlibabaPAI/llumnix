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

from typing import Optional, Tuple, Dict, List
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

logger = init_logger(__name__)


class AsyncPutQueueActor:
    def __init__(self, instance_id, request_output_queue_type: QueueType):
        self.instance_id = instance_id
        self.request_output_queue_type = request_output_queue_type
        self.request_output_queue_client: QueueClientBase = init_request_output_queue_client(request_output_queue_type)
        self.engine_actor_handle = None

    async def put_nowait_to_servers(self,
                                    server_request_outputs: Dict[str, List],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
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

def init_backend_engine(instance_id: str, request_output_queue_type: QueueType,
                        backend_type: BackendType, *args, **kwargs) -> BackendInterface:
    if backend_type == BackendType.VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.llm_engine import BackendVLLM
        backend_engine = BackendVLLM(instance_id, request_output_queue_type, *args, **kwargs)
    elif backend_type == BackendType.SIM_VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.simulator import BackendSimVLLM
        backend_engine = BackendSimVLLM(instance_id, request_output_queue_type, *args, **kwargs)
    elif backend_type == BackendType.BLADELLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.bladellm.llm_engine import BackendBladeLLM
        backend_engine = BackendBladeLLM(instance_id, request_output_queue_type, *args, **kwargs)
    else:
        raise ValueError(f'Unsupported backend: {backend_type}')
    return backend_engine

def initialize_placement_group(
    world_size: int = 1,
    detached: bool = False
) -> Tuple[str, Optional[PlacementGroup]]:
    """Initialize the distributed cluster probably with Ray.

    Args:
        parallel_config: The configurations for parallel execution.
        engine_use_ray: Whether to use Ray for async engine.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.

    Returns:
        A tuple of (`distributed_init_method`, `placement_group`). The
        `distributed_init_method` is the address for initializing the
        distributed backend. `placement_group` includes the specification
        of the resources for each distributed worker.
    """
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed "
            "serving.")

    lifetime = "detached" if detached else None
    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        gpu_bundles = 0
        for bundle in bundles:
            bundle_gpus = bundle.get("GPU", 0)
            if bundle_gpus > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 GPU.")
            if bundle_gpus:
                gpu_bundles += 1
        if world_size > gpu_bundles:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the placement group.")
    else:
        num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
        if world_size > num_gpus_in_cluster:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the cluster.")
        # Create a new placement group
        placement_group_specs = ([{"CPU": 1}] + [{"GPU": 1}] * world_size)
        current_placement_group = ray.util.placement_group(
            placement_group_specs, "STRICT_PACK", lifetime=lifetime)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        ray.get(current_placement_group.ready(), timeout=1800)

    return current_placement_group
