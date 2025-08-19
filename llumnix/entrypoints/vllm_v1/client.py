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

from typing import Any, Dict, List, Tuple, Optional, Union
import math
import time
import asyncio

import ray.actor

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.core_client import AsyncMPClient, DPAsyncMPClient
from vllm.v1.executor.abstract import Executor
from vllm.engine.async_llm_engine import AsyncStream
from vllm.config import VllmConfig
from vllm import SamplingParams
from vllm.v1.request import EngineCoreRequest

from llumnix.request_processing_context import RequestProcessingContext
from llumnix.logging.logger import init_logger
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.metrics.timestamps import set_timestamp
from llumnix.constants import LLUMNIX_TRACE_REQUEST, WAIT_MANAGER_INTERVAL
from llumnix.utils import asyncio_wait_for_ray_remote_call_with_timeout, log_instance_exception
from llumnix.request_output import LlumnixRequestOutputs
from llumnix.entrypoints.client import LlumnixClient

logger = init_logger(__name__)


def get_completion_tokens(engine_core_output: EngineCoreOutput) -> Optional[int]:
    current_completion_tokens = None
    if isinstance(engine_core_output.kv_transfer_params, dict):
        current_completion_tokens = engine_core_output.kv_transfer_params.get("num_output_tokens", None)
    return current_completion_tokens


class LlumnixClientVLLMV1(LlumnixClient, AsyncMPClient):
    def __init__(
        self,
        entrypoints_context: EntrypointsContext,
        loop: asyncio.AbstractEventLoop,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: Dict[str, str] | None = None,
        client_index: int = 0,
        driver_tensor_queue_union: Union[None, Any] = None
    ):
        LlumnixClient.__init__(self, entrypoints_context, loop)
        AsyncMPClient.__init__(self, vllm_config, executor_class, log_stats, client_addresses, client_index, driver_tensor_queue_union)
        self.engine_core_output_stash: Dict[str, Tuple[List[EngineCoreOutput], int, int]] = {}
        entrypoints_context.llumnix_client = self

    async def generate(self,
                       prompt: str,
                       sampling_params: SamplingParams,
                       request_id: str,
                       *args,
                       **kwargs):
        if sampling_params.n > 1:
            raise ValueError("Unsupported feature: multiple sequence decoding")
        logger.info("Client received request {}".format(request_id))
        # pylint: disable=unexpected-keyword-arg
        request_processing_context: RequestProcessingContext = RequestProcessingContext.deepcopy_from_server_info(
            server_info=self.server_info,
            enable_trace=kwargs.get(LLUMNIX_TRACE_REQUEST, True),
        )

        # force generate by instance
        prefill_instance_id, decode_instance_id = await self._generate_by_instance(
            request_id,
            request_processing_context,
            prompt,
            sampling_params,
            *args,
            **kwargs
        )

        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        # try:
        #     prefill_instance_id, decode_instance_id = await self._generate_by_manager(
        #         request_id,
        #         request_processing_context,
        #         prompt,
        #         sampling_params,
        #         *args,
        #         **kwargs
        #     )
        #     self.manager_available = True
        # # pylint: disable=broad-except
        # except Exception as e:
        #     self._handle_generate_by_manager_error(request_id, e)
        #     # Do not re-generate the request to avoid duplicate requests.
        #     if self.manager_available:
        #         self.manager_available = False
        #     prefill_instance_id, decode_instance_id = await self._generate_by_instance(
        #         request_id,
        #         request_processing_context,
        #         prompt,
        #         sampling_params,
        #         *args,
        #         **kwargs
        #     )
        self.request_instances[request_id].append(prefill_instance_id)
        self.instance_requests[prefill_instance_id].add(request_id)
        if decode_instance_id and decode_instance_id == prefill_instance_id:
            self.request_instances[request_id].append(decode_instance_id)
            self.instance_requests[decode_instance_id].add(request_id)

    # pylint: disable=arguments-differ
    async def _generate_by_manager(self,
                                   request_id: str,
                                   request_processing_context: RequestProcessingContext,
                                   prompt: str,
                                   sampling_params: SamplingParams,
                                   *args,
                                   **kwargs) -> Tuple[str, str]:
        request_processing_context.add_trace_timeline('api_server_generate_timestamp')
        return await asyncio_wait_for_ray_remote_call_with_timeout(
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
            # pylint: disable=no-else-return
            if self.instance_num_requests:
                instance_id = min(self.instance_num_requests, key=self.instance_num_requests.get)
                self.instance_num_requests[instance_id] += 1
                expected_steps = math.inf # ignore enable_pd_disagg when skip manager dispatch
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    self.instances[instance_id].generate,
                    request_id, request_processing_context, expected_steps, prompt, sampling_params, *args, **kwargs
                )
                logger.warning("Manager is unavailable temporarily, "
                               "dispatch request {} to instance {}.".format(request_id, instance_id))
                # return prefill instance id and decode instance id
                return instance_id, instance_id
            else:
                logger.error("Manager is unavailable temporarily, but there is no instance behind this api server, "
                    "sleep {}s, waiting for manager available".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                asyncio.create_task(self.generate(prompt, sampling_params, request_id, *args, **kwargs))
        # pylint: disable=broad-except
        except Exception as e:
            if instance_id in self.instances:
                self._handle_generate_by_instance_error(request_id, instance_id, e)
            return await asyncio.create_task(self.generate(prompt, sampling_params, request_id, *args, **kwargs))

    async def abort(self, request_id: str) -> None:
        await self._abort(request_id)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # Rewrite from AsyncMPClient
        request.client_index = self.client_index
        await self.generate(None, request.sampling_params, request.request_id, engine_core_request=request)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # Rewrite from AsyncMPClient
        for request_id in request_ids:
            await self.abort(request_id)

    def abort_request(self, request_id: str) -> None:
        instance_ids, instances = self._get_instance_for_abort(request_id)
        if instances:
            for instance_id, instance in zip(instance_ids, instances):
                logger.info("Abort request {} (instance_id: {}).".format(request_id, instance_id))
                asyncio.create_task(self._abort_request(instance_id, instance, request_id))
        else:
            logger.warning("Failed to abort request {} (instance_ids: {}, instances: {}).".format(
                request_id, instance_ids, instances))

    async def _abort_request(self, instance_id: str, instance: ray.actor.ActorHandle, request_id: str):
        try:
            await asyncio_wait_for_ray_remote_call_with_timeout(instance.abort, request_id)
            self._clear_client_request_states(request_id)
        # pylint: disable=broad-except
        except Exception as e:
            log_instance_exception(e, instance_id, "_abort_request", request_id)

    async def get_request_outputs_loop(self):
        """Process output order and put EngineCoreOutputs to local queue"""
        while True:
            llumnix_request_outputs: LlumnixRequestOutputs = await self.request_output_queue.get()
            outputs: List[EngineCoreOutput] = []
            for engine_core_output in llumnix_request_outputs.engine_outputs.outputs:
                set_timestamp(engine_core_output, 'api_server_get_queue_timestamp', time.time())

                # current_completion_tokens = engine_core_output.kv_transfer_params.get("num_output_tokens", None)
                # print(f"[zzy][trace] {engine_core_output.request_id}_{current_completion_tokens} -1 get_from_output_queue {time.perf_counter()}")

                request_id = engine_core_output.request_id
                # update the lastest instance_id for adapting migration scene
                if self.request_instances[request_id]:
                    self.request_instances[request_id][-1] = llumnix_request_outputs.instance_id
                else:
                    self.request_instances[request_id].append(llumnix_request_outputs.instance_id)
                processed_output = self._process_output_order(
                    request_id, engine_core_output,
                )
                if not processed_output:
                    continue
                outputs.extend(processed_output)
                last_output = processed_output[-1]
                self.request_stream_last_completion_tokens[request_id] = get_completion_tokens(last_output)
                if last_output.finished:
                    logger.info("Client finished request {}.".format(request_id))
                    request_processing_context = llumnix_request_outputs.request_processing_context_dict[request_id]
                    lantency_dict = request_processing_context.trace_timeline.to_latency_breakdown_dict()
                    print(f"[zzy][trace] lantency of request {request_id}: {lantency_dict}")
                    print(f"[zzy][trace] trace_timeline of request {request_id}: {request_processing_context.trace_timeline}")
                    self._clear_client_request_states(request_id)
            llumnix_request_outputs.engine_outputs.outputs = outputs
            if llumnix_request_outputs.engine_outputs.outputs or llumnix_request_outputs.engine_outputs.scheduler_stats:
                self.outputs_queue.put_nowait(llumnix_request_outputs.engine_outputs)

    def _ensure_output_queue_task(self):
        # Overload AsyncMPClient._ensure_output_queue_task
        pass

    def _clear_client_request_states(self, request_id: str):
        super()._clear_client_request_states(request_id)
        self.engine_core_output_stash.pop(request_id, None)

    # pylint: disable=arguments-renamed
    def _process_output_order(
        self,
        request_id: str,
        engine_core_output: EngineCoreOutput,
        # current_completion_tokens_dict: Dict[str, int],
    ) -> List[EngineCoreOutput]:
        current_completion_tokens = get_completion_tokens(engine_core_output)

        if not current_completion_tokens:
            # No num_output_tokens info, return the engine_core_output directly.
            return [engine_core_output]

        current_new_tokens = len(engine_core_output.new_token_ids)
        last_completion_tokens = self.request_stream_last_completion_tokens.get(request_id, 0)
        support_completion_tokens = last_completion_tokens + current_new_tokens
        if current_completion_tokens > support_completion_tokens:
            logger.info(
                "request {} out-of-order output, last completion tokens is {}"
                ", current completion tokens is {}, current tokens is {}, stash current output..."
                .format(request_id, last_completion_tokens, current_completion_tokens, current_new_tokens)
            )
            if hasattr(engine_core_output, 'request_timestamps'):
                logger.info(
                    "out-of-order request({}) output timestamps: {}".format(
                    request_id, engine_core_output.request_timestamps.to_latency_breakdown_dict()
                    )
                )
            self.engine_core_output_stash.setdefault(request_id, []).append(engine_core_output)
            return []

        return [engine_core_output]

    async def call_utility_async(self, method: str, *args) -> Any:
        instance = list(self.instances.values())[0]
        return await instance.call_engine_utility_async.remote(method, *args)

    def cancel_dead_instance_requests(self, dead_instance_ids: List[str]) -> None:
        for dead_instance_id in dead_instance_ids:
            request_ids = self.instance_requests.get(dead_instance_id, [])
            logger.error("Cancel requests: {}".format(request_ids))
            llumnix_request_outputs = LlumnixRequestOutputs(
                instance_id=dead_instance_id,
                engine_outputs=EngineCoreOutputs(),
                request_processing_context_dict={},
            )
            for request_id in request_ids:
                logger.error(
                    "Request {} is cancelled because instance {} is dead".format(
                        request_id, dead_instance_id
                    )
                )

                llumnix_request_outputs.engine_outputs.outputs.append(
                    EngineCoreOutput(
                        request_id=request_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.ABORT,
                        stop_reason="Server internal error, please retry.",
                    )
                )
                # request status will clear in get_request_outputs_loop()
            if len(llumnix_request_outputs.engine_outputs.outputs)>0:
                self.request_output_queue.put_nowait(llumnix_request_outputs)

            self.instance_requests.pop(dead_instance_id, None)

class LlumnixDPClientVLLMV1(LlumnixClientVLLMV1, DPAsyncMPClient):
    def __init__(self, *args, **kwargs):
        self.current_wave = 0

        LlumnixClientVLLMV1.__init__(self, *args, **kwargs)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_output_queue_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        await self.generate(None, request.sampling_params, request.request_id, engine_core_request=request)
