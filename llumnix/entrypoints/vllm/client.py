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

import math
import asyncio
from typing import Dict, List
import ray.actor

from vllm.engine.async_llm_engine import AsyncStream
from vllm.outputs import RequestOutput
from vllm import SamplingParams

from llumnix.logging.logger import init_logger
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.request_processing_context import RequestProcessingContext
from llumnix.constants import WAIT_MANAGER_INTERVAL, LLUMNIX_TRACE_REQUEST
from llumnix.utils import (
    asyncio_wait_for_ray_remote_call_with_timeout,
    log_instance_exception,
)
from llumnix.request_output import LlumnixRequestOuput as LlumnixRequestOuputVLLM
from llumnix.entrypoints.client import LlumnixClient

logger = init_logger(__name__)


class LlumnixClientVLLM(LlumnixClient):
    def __init__(self, entrypoints_context: EntrypointsContext, loop: asyncio.AbstractEventLoop):
        self.request_stream: Dict[str, AsyncStream] = {}
        super().__init__(entrypoints_context, loop)

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
        self.request_stream[request_id] = results_generator
        request_processing_context: RequestProcessingContext = RequestProcessingContext.deepcopy_from_server_info(
            server_info=self.server_info,
            enable_trace=kwargs.get(LLUMNIX_TRACE_REQUEST, False),
        )
        kwargs.pop(LLUMNIX_TRACE_REQUEST, None)

        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            await self._generate_by_manager(request_id, request_processing_context, prompt, sampling_params, *args, **kwargs)
            self.manager_available = True
        # pylint: disable=broad-except
        except Exception as e:
            self._handle_generate_by_manager_error(request_id, e)
            # Do not re-generate the request to avoid duplicate requests.
            if self.manager_available:
                self.manager_available = False
                return results_generator
            await self._generate_by_instance(request_id, request_processing_context, prompt, sampling_params, *args, **kwargs)

        return results_generator

    # pylint: disable=arguments-differ
    async def _generate_by_manager(self,
                                   request_id: str,
                                   request_processing_context: RequestProcessingContext,
                                   prompt: str,
                                   sampling_params: SamplingParams,
                                   *args,
                                   **kwargs) -> AsyncStream:
        request_processing_context.add_trace_timeline("api_server_generate_timestamp")
        await asyncio_wait_for_ray_remote_call_with_timeout(
            self.manager.generate, request_id, request_processing_context, prompt, sampling_params, *args, **kwargs
        )

    # pylint: disable=arguments-differ
    async def _generate_by_instance(self,
                                    request_id: str,
                                    request_processing_context: RequestProcessingContext,
                                    prompt: str,
                                    sampling_params: SamplingParams,
                                    *args,
                                    **kwargs) -> AsyncStream:
        try:
            if self.instance_num_requests:
                request_processing_context.add_trace_timeline("api_server_generate_timestamp")
                instance_id = min(self.instance_num_requests, key=self.instance_num_requests.get)
                self.instance_num_requests[instance_id] += 1
                expected_steps = math.inf # ignore enable_pd_disagg when skip manager dispatch
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    self.instances[instance_id].generate,
                    request_id, request_processing_context, expected_steps, prompt, sampling_params, *args, **kwargs
                )
                logger.warning("Manager is unavailable temporarily, "
                               "dispatch request {} to instance {}.".format(request_id, instance_id))
            else:
                logger.error("Manager is unavailable temporarily, but there is no instance behind this api server, "
                    "sleep {}s, waiting for manager available".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                asyncio.create_task(self.generate(prompt, sampling_params, request_id, *args, **kwargs))
        # pylint: disable=broad-except
        except Exception as e:
            if instance_id in self.instances:
                self._handle_generate_by_instance_error(request_id, instance_id, e)
            asyncio.create_task(self.generate(prompt, sampling_params, request_id, *args, **kwargs))

    async def abort(self, request_id: str) -> None:
        await self._abort(request_id)

    def abort_request(self, request_id: str) -> None:
        instance_id, instance = self._get_instance_for_abort(request_id)
        if instance:
            logger.info("Abort request {} (instance_id: {}).".format(request_id, instance_id))
            asyncio.create_task(self._abort_request(instance_id, instance, request_id))
        else:
            logger.warning("Failed to abort request {} (instance_id: {}, instance: {}).".format(
                request_id, instance_id, instance))

    async def _abort_request(self, instance_id: str, instance: ray.actor.ActorHandle, request_id: str):
        try:
            await asyncio_wait_for_ray_remote_call_with_timeout(instance.abort, request_id)
            self._clear_client_request_states(request_id)
        # pylint: disable=broad-except
        except Exception as e:
            log_instance_exception(e, instance_id, "_abort_request", request_id)

    async def get_request_outputs_loop(self):
        while True:
            try:
                request_responses: List[LlumnixRequestOuputVLLM] = await self.request_output_queue.get()
                for request_response in request_responses:
                    request_response.request_processing_context.set_timestamp('api_server_get_queue_timestamp')
                    request_output: RequestOutput = request_response.get_engine_output()
                    request_id = request_response.request_id
                    # Request could be dispatched twice when manager is dead, the first request will free the request_streams when finished.
                    if request_id not in self.request_stream:
                        continue
                    # Update when request_id is in self.request_streams.
                    self.request_instance[request_id] = request_response.instance_id

                    processed_output = self._process_output_order(request_id, request_response)
                    if not processed_output:
                        continue
                    self.request_stream[request_id].put(processed_output)
                    if request_output.finished:
                        logger.info("Client finished request {}.".format(request_id))
                        self._clear_client_request_states(request_id)
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Client get error in get_request_outputs_loop, client keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    def _clear_client_request_states(self, request_id: str):
        super()._clear_client_request_states(request_id)
        if request_id in self.request_stream:
            self.request_stream[request_id].finish()
            del self.request_stream[request_id]
        else:
            logger.error("Request {} not found.".format(request_id))

    def _process_output_order(self, request_id: str, request_output: LlumnixRequestOuputVLLM) -> LlumnixRequestOuputVLLM:
        engine_output = request_output.get_engine_output()
        current_completion_tokens = None
        if hasattr(engine_output, "outputs") and len(engine_output.outputs) > 0:
            current_completion_tokens = len(engine_output.outputs[-1].token_ids)

        if not current_completion_tokens:
            # request_output has no outputs, return the request_output directly.
            return request_output

        last_completion_tokens = self.request_stream_last_completion_tokens.get(request_id, 0)
        if current_completion_tokens <= last_completion_tokens:
            # process the out-of-order output
            logger.info(
                "request {} out-of-order output, last num completion tokens is {}"
                ", num current completion tokens is {}, skip current output...".format(
                    request_id, last_completion_tokens, current_completion_tokens
                )
            )
            if request_output.request_processing_context.enable_trace:
                logger.info(
                    "out-of-order request({}) output timestamps: {}".format(
                        request_id,
                        request_output.request_processing_context.trace_timeline.to_latency_breakdown_dict(),
                    )
                )
            return None
        self.request_stream_last_completion_tokens[request_id] = current_completion_tokens

        return request_output
