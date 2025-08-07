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

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Any
import asyncio
from collections import defaultdict

import ray

from llumnix.manager import Manager
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.llumlet.llumlet import Llumlet
from llumnix.metrics.llumnix_client_metrics import LlumnixClientMetrics
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.server_info import ServerInfo
from llumnix.constants import INIT_CACHED_CLUSTER_ACTORS_INTERVAL, UPDATE_CACHED_CLUSTER_ACTORS_INTERVAL
from llumnix.ray_utils import (
    get_llumnix_actor_handle,
    update_cluster_actor_handles,
    LlumnixActor,
)
from llumnix.logging.logger import init_logger
from llumnix.utils import (
    RequestIDType,
    asyncio_wait_for_ray_remote_call_with_timeout,
    log_instance_exception,
    log_manager_exception,
)

logger = init_logger(__name__)


class LlumnixClient(ABC):
    def __init__(self,
                 entrypoints_context: EntrypointsContext,
                 loop: asyncio.AbstractEventLoop):
        self.manager: Manager = entrypoints_context.manager
        self.instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.request_output_queue: QueueServerBase = entrypoints_context.request_output_queue
        self.server: APIServerActor = entrypoints_context.server
        self.server_info: ServerInfo = entrypoints_context.server_info
        self.log_requests: bool = entrypoints_context.log_requests

        self.instance_requests: Dict[str, Set[RequestIDType]] = defaultdict(set)
        self.request_instances: Dict[RequestIDType, List[str]] = defaultdict(list)
        self.cached_cluster_instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.instance_num_requests: Dict[str, int] = {}
        for ins_id in self.instances.keys():
            self.instance_num_requests[ins_id] = 0
        self.request_stream_last_completion_tokens: Dict[RequestIDType, int] = {}
        self.num_finished_requests = 0
        self.manager_available = True
        self.request_generate_by_instance_dict: Dict[RequestIDType, int] = {}

        # metrics
        self.llumnix_client_metrics = LlumnixClientMetrics(server_id = self.server_info.server_id)
        loop.create_task(self.get_request_outputs_loop())
        loop.create_task(self.request_output_queue.run_server_loop())
        loop.create_task(self._update_cached_cluster_instances_loop())

    @abstractmethod
    async def get_request_outputs_loop(self):
        raise NotImplementedError

    @abstractmethod
    async def _generate_by_manager(self, request_id: RequestIDType, server_info: ServerInfo, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def _generate_by_instance(self, request_id: RequestIDType, server_info: ServerInfo, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _process_output_order(self, request_id: RequestIDType, request_output: Any):
        raise NotImplementedError

    @abstractmethod
    def cancel_dead_instance_requests(self, dead_instance_ids: List[str]) -> None:
        raise NotImplementedError

    async def is_ready(self) -> bool:
        return await asyncio_wait_for_ray_remote_call_with_timeout(self.manager.is_ready)

    def cleanup(self):
        self.request_output_queue.cleanup()
        instance_ids = list(self.instances.keys())
        logger.info("Server stopped (instance_ids: {}).".format(instance_ids))
        if self.server is not None:
            ray.kill(self.server)

    async def _abort(self, request_id: RequestIDType) -> None:
        instance_ids, instances = self._get_instance_for_abort(request_id)
        if instances:
            for instance_id, instance in zip(instance_ids, instances):
                self.cached_cluster_instances[instance_id] = instance
                logger.info("Abort request {} (instance_id: {}).".format(request_id, instance_id))
                try:
                    # instance should not throw error if aborting request id is not existed
                    await asyncio_wait_for_ray_remote_call_with_timeout(instance.abort, request_id)
                    self._clear_client_request_states(request_id)
                except Exception as e: # pylint: disable=broad-except
                    log_instance_exception(e, instance_id, "_abort", request_id)
        else:
            logger.warning("Failed to abort request {} (instance_ids: {}, instances: {}).".format(
                request_id, instance_ids, instances))

    def _clear_client_request_states(self, request_id: RequestIDType):
        self.request_stream_last_completion_tokens.pop(request_id, None)
        self.request_generate_by_instance_dict.pop(request_id, None)
        instance_ids = self.request_instances.pop(request_id, [])
        for instance_id in instance_ids:
            if instance_id in self.instance_requests and request_id in self.instance_requests[instance_id]:
                self.instance_requests[instance_id].remove(request_id)

    def _get_instance_for_abort(self, request_id: RequestIDType) -> Tuple[str, Llumlet]:
        instance_ids = list(self.request_instances.get(request_id))
        available_instances = []
        available_instance_ids = []
        for instance_id in instance_ids:
            if instance_id in self.cached_cluster_instances:
                available_instances.append(self.cached_cluster_instances[instance_id])
                available_instance_ids.append(instance_id)
            else:
                instance = get_llumnix_actor_handle(LlumnixActor.INSTANCE, instance_id, raise_exc=False)
                if instance is not None:
                    available_instances.append(instance)
                    available_instance_ids.append(instance_id)
        # vllm and blade will check if request_id existed, it will not cause error if request_id is not existed.
        # Function drop_request in blade is not work correctly when enable semi-pd.
        return available_instance_ids, available_instances

    async def _update_cached_cluster_instances_loop(self):
        await asyncio.sleep(INIT_CACHED_CLUSTER_ACTORS_INTERVAL)
        while True:
            try:
                self.cached_cluster_instances = update_cluster_actor_handles(
                    actor_type=LlumnixActor.INSTANCE,
                    cached_cluster_actors=self.cached_cluster_instances,
                )
                await asyncio.sleep(UPDATE_CACHED_CLUSTER_ACTORS_INTERVAL)
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Client get error in _update_cached_cluster_instances_loop, client keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    def _handle_generate_by_manager_error(self, request_id: RequestIDType, e: Exception) -> None:
        log_manager_exception(e, "generate_by_manager", request_id)

    def _handle_generate_by_instance_error(self, request_id: RequestIDType, instance_id: str, e: Exception) -> None:
        log_instance_exception(e, instance_id, "generate_by_instance", request_id)
        if instance_id in self.instances:
            del self.instances[instance_id]
        else:
            logger.warning("Instance {} is not in self.instances.".format(instance_id))
        if instance_id in self.instance_num_requests:
            del self.instance_num_requests[instance_id]
        else:
            logger.warning("Instance {} is not in self.instance_num_requests.".format(instance_id))
