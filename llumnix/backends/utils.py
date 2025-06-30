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

from typing import Dict, List, Coroutine, Union, Mapping
import asyncio
import time
import os
from enum import Enum
import queue
import threading
from abc import ABC, abstractmethod

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from llumnix.arg_utils import LlumnixEngineArgs, InstanceArgs
from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.queue.queue_type import QueueType
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.server_info import ServerInfo
from llumnix.logging.logger import init_logger
from llumnix.ray_utils import get_instance_name, log_actor_ray_info
from llumnix.metrics.timestamps import set_timestamp
from llumnix.utils import asyncio_wait_for_with_timeout, RequestIDType
from llumnix.request_output import LlumnixRequestOuput, LlumnixRequestOutputs
from llumnix.queue.utils import init_request_output_queue_client
from llumnix.utils import ray_get_with_timeout, exception_wrapper_async
from llumnix.constants import NUM_GPUS_BLADELLM_GPU_ACTOR

logger = init_logger(__name__)

OutputsType = Union[List[LlumnixRequestOuput], LlumnixRequestOutputs]

class RequestOutputForwardingMode(str, Enum):
    ACTOR = "actor"
    THREAD = "thread"


class StopPutQueueSignal:
    pass


class BaseOutputMediator(ABC):
    @abstractmethod
    async def put_nowait_to_servers(self,
                                    server_request_outputs: Mapping[str, OutputsType],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    async def put_nowait_to_servers_func(
        self,
        request_output_queue_type: QueueType,
        request_output_queue_client: QueueClientBase,
        server_request_outputs: Mapping[str, OutputsType],
        server_info_dict: Dict[str, ServerInfo],
    ) -> List[RequestIDType]:
        tasks = []
        for server_id, req_outputs in server_request_outputs.items():
            server_info = server_info_dict[server_id]
            set_timestamp(req_outputs, 'engine_actor_put_queue_timestamp', time.time())
            tasks.append(asyncio.create_task(request_output_queue_client.put_nowait(req_outputs, server_info)))
        rets = await asyncio.gather(*tasks, return_exceptions=True)
        aborted_request_ids = []
        for idx, ret in enumerate(rets):
            # If exception occurs during sending message from queue client to queue server,
            # ActorOutputMediator will not die.
            if isinstance(ret, Exception):
                server_id = list(server_request_outputs.keys())[idx]
                server_info = server_info_dict[server_id]
                logger.error("Server {} is dead, exception: {}.".format(server_id, ret))
                if request_output_queue_type == QueueType.ZMQ:
                    logger.debug("request output queue ip: {}, port: {}".format(server_info.request_output_queue_ip,
                                                                                server_info.request_output_queue_port))
                req_outputs = list(server_request_outputs.values())[idx]
                for req_output in req_outputs:
                    aborted_request_ids.append(req_output.request_id)

        return aborted_request_ids


class ActorOutputMediator(BaseOutputMediator):
    def __init__(self,
                 instance_id: str,
                 request_output_queue_type: QueueType):
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.instance_id = instance_id
        self.request_output_queue_type = request_output_queue_type
        self.request_output_queue_client = init_request_output_queue_client(request_output_queue_type)
        self.engine_actor_handle = None

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]})"

    async def put_nowait_to_servers(self,
                                    server_request_outputs: Mapping[str, OutputsType],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
        # fake metric, for alignment
        for req_outputs in server_request_outputs.values():
            set_timestamp(req_outputs, 'engine_thread_put_queue_timestamp', time.time())
        if self.engine_actor_handle is None:
            # The lifetime of ActorOutputMediator is the same as the lifetime of the instance actor,
            # so we do not handling exception here.
            self.engine_actor_handle = ray.get_actor(get_instance_name(self.instance_id), namespace="llumnix")
        aborted_request_ids = await self.put_nowait_to_servers_func(
            self.request_output_queue_type, self.request_output_queue_client, server_request_outputs, server_info_dict
        )
        if aborted_request_ids:
            await asyncio_wait_for_with_timeout(self.engine_actor_handle.abort.remote(aborted_request_ids))

    def stop(self):
        if self.request_output_queue_type == QueueType.ZMQ:
            self.request_output_queue_client.close()


class ThreadOutputMediator(BaseOutputMediator):
    def __init__(self,
                 instance_id: str,
                 request_output_queue_type: QueueType,
                 abort_request_callback: Coroutine):
        self.request_output_queue_type = request_output_queue_type
        self.request_output_queue_client = init_request_output_queue_client(self.request_output_queue_type)
        self.abort_request_callback = abort_request_callback
        self.server_request_outputs_queue = queue.Queue()
        self.main_loop = asyncio.get_event_loop()
        self.put_queue_loop_thread = threading.Thread(
            target=self._start_put_queue_loop, args=(), daemon=True, name=f"TheadOutputMediator_{instance_id}"
        )
        self.put_queue_loop_thread.start()

    async def put_nowait_to_servers(self,
                                    server_request_outputs: Mapping[str, OutputsType],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
        for req_outputs in server_request_outputs.values():
            set_timestamp(req_outputs, 'engine_thread_put_queue_timestamp', time.time())
        if self.put_queue_loop_thread.is_alive():
            self.server_request_outputs_queue.put_nowait((server_request_outputs, server_info_dict))
        # Ensure engine will die if put queue loop thread is dead.
        else:
            raise RuntimeError("Engine ThreadOutputMediator is dead.")

    def _start_put_queue_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async def put_queue_loop():
            while True:
                item = self.server_request_outputs_queue.get()
                if isinstance(item, StopPutQueueSignal):
                    break
                server_request_outputs, server_info_dict = item
                await self._put_nowait_to_servers(
                    server_request_outputs, server_info_dict,
                )
        try:
            loop.run_until_complete(put_queue_loop())
        finally:
            loop.close()

    async def _put_nowait_to_servers(
        self,
        server_request_outputs: Mapping[str, OutputsType],
        server_info_dict: Dict[str, ServerInfo],
    ) -> None:
        aborted_request_ids = await self.put_nowait_to_servers_func(
            self.request_output_queue_type, self.request_output_queue_client, server_request_outputs, server_info_dict
        )
        abort_request_callback = exception_wrapper_async(self.abort_request_callback)
        if aborted_request_ids:
            asyncio.run_coroutine_threadsafe(
                abort_request_callback(aborted_request_ids), self.main_loop
            )

    def stop(self) -> None:
        self.server_request_outputs_queue.put(StopPutQueueSignal())
        self.put_queue_loop_thread.join()
        if self.request_output_queue_type == QueueType.ZMQ:
            self.request_output_queue_client.close()


class OutputMediator:
    def __init__(self,
                 instance_id: str,
                 request_output_queue_type: QueueType,
                 request_output_forwarding_mode: RequestOutputForwardingMode,
                 abort_request_callback: Coroutine,
                 placement_group: PlacementGroup,
                 backend_type: BackendType,
                 ):
        self.request_output_queue_type = request_output_queue_type
        self.request_output_forwarding_mode = request_output_forwarding_mode
        self.abort_request_callback = abort_request_callback
        if self.request_output_forwarding_mode == RequestOutputForwardingMode.ACTOR:
            # Place the actor mediator together with the instance.
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
            num_gpus = NUM_GPUS_BLADELLM_GPU_ACTOR if backend_type == BackendType.BLADELLM else 0
            self.actor_mediator: ActorOutputMediator = ray.remote(
                num_cpus=1,
                num_gpus=0.1,
                scheduling_strategy=scheduling_strategy,
                name=f"ActorOutputMediator_{instance_id}"
            )(ActorOutputMediator).remote(instance_id, request_output_queue_type)
        else:
            self.thread_mediator: ThreadOutputMediator = ThreadOutputMediator(
                instance_id,
                request_output_queue_type,
                abort_request_callback,
            )

    async def put_request_outputs_to_server(
        self,
        server_request_outputs: Mapping[str, OutputsType],
        server_info_dict: Dict[str, ServerInfo]
    ) -> None:
        if self.request_output_forwarding_mode == RequestOutputForwardingMode.ACTOR:
            await asyncio_wait_for_with_timeout(
                self.actor_mediator.put_nowait_to_servers.remote(
                    server_request_outputs, server_info_dict
                )
            )
        else:
            await self.thread_mediator.put_nowait_to_servers(server_request_outputs, server_info_dict)

    def stop(self):
        if self.request_output_forwarding_mode == RequestOutputForwardingMode.ACTOR:
            try:
                ray_get_with_timeout(self.actor_mediator.stop.remote())
            # pylint: disable=broad-except
            except Exception:
                logger.exception("Error in OutputMediator stop ActorOutputMediator")
        else:
            self.thread_mediator.stop()


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
    elif backend_type == BackendType.VLLM_V1:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm_v1.llm_engine import BackendVLLMV1
        backend_engine = BackendVLLMV1(instance_id,
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
