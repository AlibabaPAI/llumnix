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

import copy
import math
import time
import asyncio
from typing import Dict, Tuple, List

import ray.exceptions

from vllm.engine.async_llm_engine import AsyncStream
from vllm.outputs import RequestOutput
from vllm import SamplingParams

from llumnix.manager import Manager
from llumnix.logging.logger import init_logger
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.metrics.timestamps import RequestTimestamps, set_timestamp
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.server_info import ServerInfo
from llumnix.llumlet.llumlet import Llumlet
from llumnix.constants import WAIT_MANAGER_INTERVAL, INIT_GLOBAL_INSTANCES_INTERVAL, UPDATE_GLOBAL_INSTANCES_INTERVAL
from llumnix.ray_utils import (execute_actor_method_async_with_retries, get_instance, get_actor_names_by_name_prefix,
                               INSTANCE_NAME_PREFIX)
from llumnix.utils import asyncio_wait_for_with_timeout
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.request_output import LlumnixRequestOuput as LlumnixRequestOuputVLLM

logger = init_logger(__name__)


class LlumnixClientVLLM:
    def __init__(self, entrypoints_context: EntrypointsContext, loop: asyncio.AbstractEventLoop):
        self.manager: Manager = entrypoints_context.manager
        self.instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.request_output_queue: QueueServerBase = entrypoints_context.request_output_queue
        self.server: APIServerActor = entrypoints_context.server
        self.server_info: ServerInfo = entrypoints_context.server_info
        self.log_requests: bool = entrypoints_context.log_requests
        self.log_request_timestamps: bool = entrypoints_context.log_request_timestamps

        self.request_streams: Dict[str, AsyncStream] = {}
        self.request_instance: Dict[str, str] = {}
        # TODO(s5u13): Consider a better way to get instance handle without calling ray.
        self.global_instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.instance_num_requests: Dict[str, int] = {}
        self.request_streams_last_completion_tokens: Dict[str, int] = {}
        for ins_id in self.instances.keys():
            self.instance_num_requests[ins_id] = 0
        self.num_finished_requests = 0
        self.manager_available = True

        loop.create_task(self.get_request_outputs_loop())
        loop.create_task(self.request_output_queue.run_server_loop())
        loop.create_task(self._update_global_instances_loop())

    async def generate(self,
                       prompt: str,
                       sampling_params: SamplingParams,
                       request_id: str,
                       *args,
                       **kwargs) -> AsyncStream:
        if sampling_params.n > 1:
            raise ValueError("Unsupported feature: multiple sequence decoding")
        logger.info("Client received request {}".format(request_id))
        # pylint: disable=unexpected-keyword-arg
        results_generator = AsyncStream(request_id, cancel=self.abort_request)
        self.request_streams[request_id] = results_generator
        server_info_copy = copy.deepcopy(self.server_info)

        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            await self._generate_by_manager(request_id, server_info_copy, prompt, sampling_params, *args, **kwargs)
            self.manager_available = True
        # pylint: disable=broad-except
        except Exception as e:
            if isinstance(e, ray.exceptions.RayActorError):
                logger.error("Manager is unavailable.")
            elif isinstance(e, asyncio.TimeoutError):
                logger.error("Failed to generate request {} by manager, manager is hang, "
                             "please check the cause.".format(request_id))
            else:
                logger.exception("Failed to generate request {} by manager, "
                                 "unexpected exception: {}".format(request_id, e))
            # Do not re-generate the request to avoid duplicate requests.
            if self.manager_available:
                self.manager_available = False
                return results_generator
            await self._generate_by_instance(request_id, server_info_copy, prompt, sampling_params, *args, **kwargs)

        return results_generator

    async def _generate_by_manager(self,
                                   request_id: str,
                                   server_info: ServerInfo,
                                   prompt: str,
                                   sampling_params: SamplingParams,
                                   *args,
                                   **kwargs) -> AsyncStream:
        if self.log_request_timestamps:
            # Hack request timestamps in server_info for latency breakdown.
            server_info.request_timestamps = RequestTimestamps()
            set_timestamp(server_info, "api_server_generate_timestamp", time.time())
        await self.manager.generate.remote(request_id, server_info, prompt, sampling_params, *args, **kwargs)

    async def _generate_by_instance(self,
                                    request_id: str,
                                    server_info: ServerInfo,
                                    prompt: str,
                                    sampling_params: SamplingParams,
                                    *args,
                                    **kwargs) -> AsyncStream:
        try:
            if self.instance_num_requests:
                instance_id = min(self.instance_num_requests, key=self.instance_num_requests.get)
                self.instance_num_requests[instance_id] += 1
                expected_steps = math.inf # ignore enable_pd_disagg when skip manager dispatch
                await asyncio_wait_for_with_timeout(
                    self.instances[instance_id].generate.remote(
                        request_id, server_info, expected_steps, prompt, sampling_params, *args, **kwargs
                    )
                )
                logger.warning("Manager is unavailable temporarily, dispatch request {} to instance {}".format(
                    request_id, instance_id))
            else:
                logger.error("Manager is unavailable temporarily, but there is no instance behind this api server, "
                    "sleep {}s, waiting for manager available".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                return await asyncio.create_task(self.generate(prompt, sampling_params, request_id, *args, **kwargs))
        # pylint: disable=broad-except
        except Exception as e:
            if instance_id in self.instances:
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.info("Failed to generate request {} by instance {}, instance is dead.".format(
                        request_id, instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error("Failed to generate request {} by instance {}, instance is hang, "
                                 "please check the cause.".format(request_id, instance_id))
                else:
                    logger.exception("Failed to generate request {} by instance {}, "
                                     "unexpected exception: {}".format(request_id, instance_id, e))
                if instance_id in self.instances:
                    del self.instances[instance_id]
                else:
                    logger.warning("Instance {} is not in self.instances.".format(instance_id))
                if instance_id in self.instance_num_requests:
                    del self.instance_num_requests[instance_id]
                else:
                    logger.warning("Instance {} is not in self.instance_num_requests.".format(instance_id))
                return await asyncio.create_task(self.generate(prompt, sampling_params, request_id, *args, **kwargs))

    async def abort(self, request_id: str) -> None:
        instance_id, instance = self._get_instance_for_abort(request_id)
        if instance:
            self.global_instances[instance_id] = instance
            logger.info("Abort request {} (instance_id: {}).".format(request_id, instance_id))
            try:
                await asyncio_wait_for_with_timeout(instance.abort.remote(request_id))
                self._clear_client_request_states(request_id)
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.info("Instance {} is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error("Instance {} is hang, please check the cause.".format(instance_id))
                else:
                    logger.exception("Failed to abort request {} of instance {}, "
                                     "unexpected exception: {}".format(request_id, instance_id, e))
        else:
            logger.warning("Failed to abort request {} (instance_id: {}, instance: {}).".format(
                request_id, instance_id, instance))

    def abort_request(self, request_id: str) -> None:
        instance_id, instance = self._get_instance_for_abort(request_id)
        if instance:
            logger.info("Abort request {} (instance_id: {}).".format(request_id, instance_id))
            # TODO(s5u13b): Optimize the serialization cost.
            instance.abort.remote(request_id)
        else:
            logger.warning("Failed to abort request {} (instance_id: {}, instance: {}).".format(
                request_id, instance_id, instance))

    def _get_instance_for_abort(self, request_id: str) -> Tuple[str, Llumlet]:
        instance_id = self.request_instance.get(request_id, None)
        if instance_id is None:
            instance = None
        else:
            instance = self.global_instances[instance_id] \
                if instance_id in self.global_instances else get_instance(instance_id)

        return instance_id, instance

    async def is_ready(self) -> bool:
        return await execute_actor_method_async_with_retries(
            self.manager.is_ready.remote, "Manager", "is_ready"
        )

    async def get_request_outputs_loop(self):
        while True:
            request_responses: List[LlumnixRequestOuputVLLM] = await self.request_output_queue.get()
            for request_response in request_responses:
                request_output: RequestOutput = request_response.get_engine_output()
                set_timestamp(request_output, 'api_server_get_queue_timestamp', time.time())
                request_id = request_response.request_id
                # Request could be dispatched twice when manager is dead, the first request will free the request_streams when finished.
                if request_id not in self.request_streams:
                    continue
                # Update when request_id is in self.request_streams.
                self.request_instance[request_id] = request_response.instance_id

                processed_output = self._process_output_order(request_id, request_output)
                if not processed_output:
                    continue
                self.request_streams[request_id].put(processed_output)
                if request_output.finished:
                    logger.info("Client finish request {}.".format(request_id))
                    self._clear_client_request_states(request_id)

    def _clear_client_request_states(self, request_id: str):
        if request_id in self.request_streams:
            self.request_streams[request_id].finish()
            del self.request_streams[request_id]
        else:
            logger.error("Request {} not found.".format(request_id))
        self.request_streams_last_completion_tokens.pop(request_id, None)
        self.request_instance.pop(request_id, None)

    def _process_output_order(
        self, request_id: int, request_output: RequestOutput
    ) -> RequestOutput:
        current_completion_tokens = None
        if hasattr(request_output, "outputs") and len(request_output.outputs) > 0:
            current_completion_tokens = len(request_output.outputs[-1].token_ids)

        if not current_completion_tokens:
            # request_output has no outputs, return the request_output directly.
            return request_output

        last_completion_tokens = self.request_streams_last_completion_tokens.get(request_id, 0)
        if current_completion_tokens <= last_completion_tokens:
            # process the out-of-order output
            logger.info(
                "request {} out-of-order output, last num completion tokens is {}"
                ", num current completion tokens is {}, skip current output...".format(
                    request_id, last_completion_tokens, current_completion_tokens
                )
            )
            if hasattr(request_output, 'request_timestamps'):
                logger.info("out-of-order request({}) output timestamps: {}".format(
                    request_id, request_output.request_timestamps.to_latency_breakdown_dict()))
            return None
        self.request_streams_last_completion_tokens[request_id] = current_completion_tokens

        return request_output

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

    # TODO(s5u13b): Add base class of LlumnixClient.
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
