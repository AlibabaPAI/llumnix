from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import asyncio

import ray

from llumnix.manager import Manager
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.server_info import ServerInfo
from llumnix.constants import INIT_GLOBAL_INSTANCES_INTERVAL, UPDATE_GLOBAL_INSTANCES_INTERVAL
from llumnix.ray_utils import (
    get_actor_names_by_name_prefix,
    INSTANCE_NAME_PREFIX,
    get_instance,
    execute_actor_method_async_with_retries
)
from llumnix.logging.logger import init_logger

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
        self.log_request_timestamps: bool = entrypoints_context.log_request_timestamps

        self.request_instance: Dict[str, str] = {}
        # TODO(s5u13): Consider a better way to get instance handle without calling ray.
        self.global_instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.instance_num_requests: Dict[str, int] = {}
        for ins_id in self.instances.keys():
            self.instance_num_requests[ins_id] = 0
        self.request_stream_last_completion_tokens: Dict[str, int] = {}
        self.num_finished_requests = 0
        self.manager_available = True

        loop.create_task(self.get_request_outputs_loop())
        loop.create_task(self.request_output_queue.run_server_loop())
        loop.create_task(self._update_global_instances_loop())

    @abstractmethod
    async def get_request_outputs_loop(self):
        raise NotImplementedError

    @abstractmethod
    async def _generate_by_manager(self, request_id: int, server_info: ServerInfo, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def _generate_by_instance(self, request_id: int, server_info: ServerInfo, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _process_output_order(self, request_id: int, request_output: Any):
        raise NotImplementedError

    @abstractmethod
    def _clear_client_request_states(self, request_id: str):
        raise NotImplementedError

    async def is_ready(self) -> bool:
        return await execute_actor_method_async_with_retries(
            self.manager.is_ready.remote, "Manager", "is_ready"
        )

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
        except Exception as e:
            logger.exception("Server cleanup failed (instance_ids: {}): {}".format(instance_ids, e))
        logger.info("Server stops (instance_ids: {}).".format(instance_ids))

    def _get_instance_for_abort(self, request_id: str) -> Tuple[str, Llumlet]:
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
