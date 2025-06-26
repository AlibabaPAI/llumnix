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

from typing import Awaitable, Callable, Dict, List, Optional, Tuple
import copy
import math
import time
import asyncio

import ray.actor
import weakref

from vllm.v1.engine import EngineCoreOutputs, EngineCoreOutput
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.executor.abstract import Executor

from vllm.engine.async_llm_engine import AsyncStream
from vllm.outputs import RequestOutput
from vllm.config import VllmConfig
from vllm import SamplingParams

from llumnix.logging.logger import init_logger
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.metrics.timestamps import RequestTimestamps, set_timestamp
from llumnix.server_info import ServerInfo
from llumnix.constants import WAIT_MANAGER_INTERVAL
from llumnix.utils import asyncio_wait_for_ray_remote_call_with_timeout, log_instance_exception
from llumnix.request_output import LlumnixRequestOutputs
from llumnix.entrypoints.client import LlumnixClient

logger = init_logger(__name__)

class LlumnixClientVLLMV1(LlumnixClient, AsyncMPClient):
    def __init__(
        self,
        entrypoints_context: EntrypointsContext,
        loop: asyncio.AbstractEventLoop,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_index: int = 0,
    ):
        # self.request_stream: Dict[str, AsyncStream] = {}
        
        AsyncMPClient.__init__(self, vllm_config, executor_class, log_stats, client_addresses, client_index)
        
        self.engine_core_output_stash: Dict[int, Tuple[List[EngineCoreOutput], int, int]] = {}
        
        LlumnixClient.__init__(self, entrypoints_context, loop)
        
            
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
        server_info_copy = copy.deepcopy(self.server_info)
        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            await self._generate_by_manager(request_id, server_info_copy, prompt, sampling_params, *args, **kwargs)
            self.manager_available = True
        # pylint: disable=broad-except
        except Exception as e:
            self._handle_generate_by_manager_error(request_id, e)
            # Do not re-generate the request to avoid duplicate requests.
            if self.manager_available:
                self.manager_available = False
            await self._generate_by_instance(request_id, server_info_copy, prompt, sampling_params, *args, **kwargs)

    # pylint: disable=arguments-differ
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
        await asyncio_wait_for_ray_remote_call_with_timeout(
            self.manager.generate, request_id, server_info, prompt, sampling_params, *args, **kwargs
        )

    # pylint: disable=arguments-differ
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
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    self.instances[instance_id].generate,
                    request_id, server_info, expected_steps, prompt, sampling_params, *args, **kwargs
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

    # get outputs loop is inside AsyncMPClient
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
                request_id = engine_core_output.request_id
                self.request_instance[request_id] = llumnix_request_outputs.instance_id
                processed_output = self._process_output_order(
                    request_id, engine_core_output,
                    llumnix_request_outputs.current_completion_tokens_dict
                )
                if not processed_output:
                    continue
                outputs.extend(processed_output)
                last_output = processed_output[-1]
                self.request_stream_last_completion_tokens[request_id] = \
                    llumnix_request_outputs.current_completion_tokens_dict[request_id]
                if engine_core_output.finished:
                    logger.info("Client finished request {}.".format(request_id))
                    self._clear_client_request_states(request_id)
            llumnix_request_outputs.engine_outputs.outputs = outputs
            if llumnix_request_outputs.engine_outputs.outputs or llumnix_request_outputs.engine_outputs.scheduler_stats:
                self.outputs_queue.put_nowait(llumnix_request_outputs)
    
    async def get_output_async(self) -> LlumnixRequestOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        logger.debug("get outputs: %s", str(outputs))
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs
    
    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return
        
        outputs_queue = self.outputs_queue # AsyncMPClient, AsyncMPClient -> AsyncLLM
        request_output_queue = self.request_output_queue # LlumnixClient, Llumlet -> LlumnixClient
        request_instance = self.request_instance # LlumnixClient
        _self_ref = weakref.ref(self)
        async def get_request_outputs_loop():
            """Process output order and put EngineCoreOutputs to local queue"""
            try:
                while True:
                    # Check if self is alive
                    _self = _self_ref()
                    if not _self:
                        # Client has been garbage collected, abort.
                        break
                    llumnix_request_outputs: LlumnixRequestOutputs = await request_output_queue.get()
                    outputs: List[EngineCoreOutput] = []
                    for engine_core_output in llumnix_request_outputs.engine_outputs.outputs:
                        set_timestamp(engine_core_output, 'api_server_get_queue_timestamp', time.time())
                        request_id = engine_core_output.request_id
                        request_instance[request_id] = llumnix_request_outputs.instance_id
                        processed_output = _self._process_output_order(
                            request_id, engine_core_output,
                            llumnix_request_outputs.current_completion_tokens_dict
                        )
                        if not processed_output:
                            continue
                        outputs.extend(processed_output)
                        if engine_core_output.finished:
                            logger.info("Client finished request {}.".format(request_id))
                            _self._clear_client_request_states(request_id)
                    llumnix_request_outputs.engine_outputs.outputs = outputs
                    if llumnix_request_outputs.engine_outputs.outputs or llumnix_request_outputs.engine_outputs.scheduler_stats:
                        outputs_queue.put_nowait(llumnix_request_outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
        
        resources.output_queue_task = asyncio.create_task(
            get_request_outputs_loop(), name="EngineCoreOutputQueueTask")

    def _clear_client_request_states(self, request_id: str):
        super()._clear_client_request_states(request_id)
        self.engine_core_output_stash.pop(request_id, None)

    def _process_output_order(
        self,
        request_id: str,
        engine_core_output: EngineCoreOutput,
        current_completion_tokens_dict: Dict[str, int],
    ) -> List[EngineCoreOutput]:
        current_completion_tokens = current_completion_tokens_dict.get(request_id)
        current_new_tokens = len(engine_core_output.new_token_ids)
        
        if not current_completion_tokens:
            # No usage info, return the request_output directly.
            return [engine_core_output]
        
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
            self.engine_core_output_stash.setdefault(request_id, []).append(
                (engine_core_output, current_completion_tokens, current_new_tokens)
            )
            return []
        
        if current_completion_tokens == support_completion_tokens:
            if not self.engine_core_output_stash.get(request_id):
                # no history output in stash
                return [engine_core_output]
            
            output_stash: List[Tuple[EngineCoreOutput, int, int]] = self.engine_core_output_stash[request_id]
            output_stash.sort(key=lambda x: x[1]) # sort by completion_tokens
            last_correct_output_index = 0
            for output, completion_tokens, new_tokens in output_stash:
                if completion_tokens > current_completion_tokens + new_tokens:
                    break
                last_correct_output_index += 1
                current_completion_tokens = completion_tokens
            if last_correct_output_index == 0:
                return [engine_core_output]
            
            res = [engine_core_output] + [output_info[0] for output_info in output_stash[:last_correct_output_index]]
            self.engine_core_output_stash[request_id] = output_stash[last_correct_output_index:]
            
            return res
        
        if current_completion_tokens == -1:
            # last output of request
            if not self.engine_core_output_stash.get(request_id):
                # no history output in stash
                return [engine_core_output]
            
            output_stash: List[Tuple[EngineCoreOutput, int, int]] = self.engine_core_output_stash[request_id]
            output_stash.sort(key=lambda x: x[1]) # sort by completion_tokens
            # drain output_stash
            res = [engine_core_output] + [output_info[0] for output_info in output_stash]
            self.engine_core_output_stash[request_id] = []
            
            return res
            
        return [engine_core_output]
