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
from typing import Dict, Tuple, Any
import asyncio

import ray

from llumnix.manager import Manager
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.llumlet.llumlet import Llumlet
from llumnix.metrics.llumnix_client_metrics import LlumnixClientMetrics
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.server_info import ServerInfo
from llumnix.constants import INIT_GLOBAL_INSTANCES_INTERVAL, UPDATE_GLOBAL_INSTANCES_INTERVAL
from llumnix.ray_utils import (
    get_actor_names_by_name_prefix,
    INSTANCE_NAME_PREFIX,
    get_instance,
)
from llumnix.logging.logger import init_logger
from llumnix.utils import (
    RequestIDType,
    asyncio_wait_for_ray_remote_call_with_timeout,
    log_instance_exception,
    log_manager_exception,
)

logger = init_logger(__name__)

# TODO(s5u13b): Find way to abstract generate function.


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
        self.log_request_timestamps: bool = entrypoints_context.log_request_timestamps

        self.request_instance: Dict[RequestIDType, str] = {}
        # TODO(s5u13): Consider a better way to get instance handle without calling ray.
        self.global_instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.instance_num_requests: Dict[str, int] = {}
        for ins_id in self.instances.keys():
            self.instance_num_requests[ins_id] = 0
        self.request_stream_last_completion_tokens: Dict[RequestIDType, int] = {}
        self.num_finished_requests = 0
        self.manager_available = True

        # metrics
        self.llumnix_client_metrics = LlumnixClientMetrics(server_id = self.server_info.server_id)

        loop.create_task(self.get_request_outputs_loop())
        loop.create_task(self.request_output_queue.run_server_loop())
        loop.create_task(self._update_global_instances_loop())

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

    async def is_ready(self) -> bool:
        return await asyncio_wait_for_ray_remote_call_with_timeout(self.manager.is_ready.remote)

    def cleanup(self):
        self.request_output_queue.cleanup()
        instance_ids = list(self.instances.keys())
        try:
            # Not call manager scale down to reduce manager overhead.
            for instance in self.instances.values():
                # Instance might die before.
                try:
                    ray.kill(instance)
                # pylint: disable=bare-except
                except:
                    pass
        # pylint: disable=broad-except
        except Exception:
            logger.exception("Server cleanup failed (instance_ids: {})".format(instance_ids))
        logger.info("Server stopped (instance_ids: {}).".format(instance_ids))

    async def _abort(self, request_id: RequestIDType) -> None:
        instance_id, instance = self._get_instance_for_abort(request_id)
        if instance:
            self.global_instances[instance_id] = instance
            logger.info("Abort request {} (instance_id: {}).".format(request_id, instance_id))
            try:
                await asyncio_wait_for_ray_remote_call_with_timeout(instance.abort.remote, request_id)
                self._clear_client_request_states(request_id)
            except Exception as e: # pylint: disable=broad-except
                log_instance_exception(e, instance_id, "_abort", request_id)
        else:
            logger.warning("Failed to abort request {} (instance_id: {}, instance: {}).".format(
                request_id, instance_id, instance))

    def _clear_client_request_states(self, request_id: RequestIDType):
        self.request_stream_last_completion_tokens.pop(request_id, None)
        self.request_instance.pop(request_id, None)

    def _get_instance_for_abort(self, request_id: RequestIDType) -> Tuple[str, Llumlet]:
        instance_id = self.request_instance.get(request_id, None)
        if instance_id is None:
            instance = None
        else:
            instance = self.global_instances[instance_id] \
                if instance_id in self.global_instances else get_instance(instance_id)

        return instance_id, instance

    async def _update_global_instances_loop(self):
        await asyncio.sleep(INIT_GLOBAL_INSTANCES_INTERVAL)
        while True:
            curr_instance_names = get_actor_names_by_name_prefix(name_prefix=INSTANCE_NAME_PREFIX)
            curr_instance_ids = [curr_instance_name.split("_")[-1] for curr_instance_name in curr_instance_names]
            new_global_instances = {}
            for instance_id in curr_instance_ids:
                if instance_id in self.global_instances:
                    new_global_instances[instance_id] = self.global_instances[instance_id]
                else:
                    instance = get_instance(instance_id)
                    if instance is not None:
                        new_global_instances[instance_id] = instance
            self.global_instances = new_global_instances
            await asyncio.sleep(UPDATE_GLOBAL_INSTANCES_INTERVAL)

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
