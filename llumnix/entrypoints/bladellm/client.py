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

import json
import time
import asyncio
import copy
import random
from typing import Dict, List, Tuple, Optional

import ray.exceptions

from blade_llm.service.communications.engine_client import MultiProcessingLLMClient
from blade_llm.service.communications.protocol import Stats
from blade_llm.service.communications.response import LLMResponse
from blade_llm.service.args import ServingArgs
from blade_llm.protocol import ServerRequest, GenerateStreamResponse
from blade_llm.service.communications.response import error_resp

from llumnix.manager import Manager
from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.logging.logger import init_logger
from llumnix.constants import WAIT_MANAGER_INTERVAL, INIT_GLOBAL_INSTANCES_INTERVAL, UPDATE_GLOBAL_INSTANCES_INTERVAL
from llumnix.metrics.timestamps import set_timestamp
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.server_info import ServerInfo
from llumnix.ray_utils import (execute_actor_method_async_with_retries, get_instance, get_actor_names_by_name_prefix,
                               INSTANCE_NAME_PREFIX)
from llumnix.utils import asyncio_wait_for_with_timeout
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.request_output import LlumnixRequestOuput as LlumnixRequestOuputBladeLLM

logger = init_logger(__name__)


class LlumnixClientBladeLLM(MultiProcessingLLMClient):
    def __init__(self,
                 args: ServingArgs,
                 entrypoints_context: EntrypointsContext,
                 loop: asyncio.AbstractEventLoop):
        super().__init__(args, -1, -1)

        self.manager: Manager = entrypoints_context.manager
        self.instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.request_output_queue: QueueServerBase = entrypoints_context.request_output_queue
        self.server: APIServerActor = entrypoints_context.server
        self.server_info: ServerInfo = entrypoints_context.server_info
        self.log_requests: bool = entrypoints_context.log_requests
        self.log_request_timestamps: bool = entrypoints_context.log_request_timestamps

        self.entrypoint_id2llumnix_id = {} # int32 -> int32
        self.llumnix_id2entrypoint_id = {} # int32 -> int32
        self.request_streams: Dict[int, asyncio.Queue] = {}
        self.request_instance: Dict[str, str] = {}
        self.global_instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.timestamps_streams: Dict[int, asyncio.Queue] = {}
        self.request_streams_last_completion_tokens: Dict[str, int] = {}
        self.request_streams_output_stash: Dict[str, List[GenerateStreamResponse]] = {}
        self.instance_num_requests: Dict[str, int] = {}
        for ins_id in self.instances.keys():
            self.instance_num_requests[ins_id] = 0
        self.num_finished_requests = 0
        self.manager_available = True

        loop.create_task(self.get_request_outputs_loop())
        loop.create_task(self.request_output_queue.run_server_loop())
        loop.create_task(self._update_global_instances_loop())

    def get_request_timestamps_generator(self, entrypoint_id: int) -> Optional[asyncio.Queue]:
        return self.timestamps_streams.get(entrypoint_id, None)

    async def _add_request(self, request: ServerRequest) -> LLMResponse:
        if request.sampling_params.n > 1 or request.sampling_params.use_beam_search:
            return error_resp(request.id, err_code=400, err_msg="Unsupported feature: multiple sequence decoding in Llumnix.")

        # To prevent different api_servers from generating the same request_id,
        # a random number is used to replace original req_id.
        llumnix_id = random.randint(1, (1 << 31) - 1)
        self.llumnix_id2entrypoint_id[llumnix_id] = request.id
        self.entrypoint_id2llumnix_id[request.id] = llumnix_id
        logger.info("request id is replaced from [{},{}] to {}".format(request.id, request.external_id, llumnix_id))
        internal_request = copy.deepcopy(request)
        internal_request.id = llumnix_id
        resp_stream = await self._generate(llumnix_id, internal_request.model_dump_json())
        return resp_stream

    async def _generate(self, request_id: int, request: ServerRequest) -> LLMResponse:
        logger.info("Client receive request {}.".format(request_id))
        results_queue = asyncio.Queue()
        self.request_streams[request_id] = results_queue
        if self.log_request_timestamps:
            entrypoint_id = self.llumnix_id2entrypoint_id.get(request_id, None)
            if entrypoint_id is not None:
                self.timestamps_streams[entrypoint_id] = asyncio.Queue()
        server_info_copy = copy.deepcopy(self.server_info)

        # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            await self._generate_by_manager(request_id, server_info_copy, request)
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
                return LLMResponse(request_id, resp_queue=results_queue)
            await self._generate_by_instance(request_id, server_info_copy, request)
        return LLMResponse(request_id, resp_queue=results_queue)

    async def _generate_by_manager(self, request_id: int, server_info: ServerInfo, request: ServerRequest):
        if self.log_request_timestamps:
            # Hack request timestamps in server_info for latency breakdown.
            server_info.request_timestamps = RequestTimestamps()
            set_timestamp(server_info, "api_server_generate_timestamp", time.time())
        # await to catch exception
        await asyncio_wait_for_with_timeout(
            self.manager.generate.remote(str(request_id), server_info, server_request=request)
        )

    async def _generate_by_instance(self, request_id: int, server_info: ServerInfo, request: ServerRequest):
        try:
            if self.instance_num_requests:
                instance_id = min(self.instance_num_requests, key=self.instance_num_requests.get)
                self.instance_num_requests[instance_id] += 1
                await asyncio_wait_for_with_timeout(
                    self.instances[instance_id].generate.remote(request_id, server_info, -1, request)
                )
                logger.warning("Manager is unavailable, "
                               "directly pass request {} to instance {}.".format(request_id, instance_id))
            else:
                logger.error("Manager is unavailable, but there is no instance behind this api server, "
                    "sleep {}s, waiting for manager restarts.".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                return await asyncio.create_task(self._generate(request_id, request))
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
                    logger.exception("Failed to generate request{} by instance {}, "
                                     "unexpected exception: {}".format(request_id, instance_id, e))
                if instance_id in self.instances:
                    del self.instances[instance_id]
                else:
                    logger.warning("Instance {} is not in self.instances.".format(instance_id))
                if instance_id in self.instance_num_requests:
                    del self.instance_num_requests[instance_id]
                else:
                    logger.warning("Instance {} is not in self.instance_num_requests,".format(instance_id))
                return await asyncio.create_task(self._generate(request_id, request))

    async def drop_request(self, req_id: int) -> None:
        llumnix_id = self.entrypoint_id2llumnix_id.get(req_id, None)
        if llumnix_id is None:
            instance_id, instance = None, None
        else:
            instance_id, instance = self._get_instance_for_abort(llumnix_id)
        if instance:
            self.global_instances[instance_id] = instance
            logger.info("Abort request {} (instance_id: {}).".format(llumnix_id, instance_id))
            try:
                await asyncio_wait_for_with_timeout(instance.abort.remote(llumnix_id))
                self._clear_client_request_states(llumnix_id)
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.info("Instance {} is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error("Instance {} is hang, please check the cause.".format(instance_id))
                else:
                    logger.exception("Failed to abort request {} of instance {}, "
                                     "unexpected exception: {}".format(llumnix_id, instance_id, e))
        else:
            logger.warning("Failed to abort request {} (instance_id: {}, instance: {}).".format(
                llumnix_id, instance_id, instance))

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
            self.manager.is_ready.remote, "Manager", "is_ready",
        )

    async def get_request_outputs_loop(self):
        while True:
            request_responses: List[LlumnixRequestOuputBladeLLM] = await self.request_output_queue.get()
            if request_responses is None:
                continue

            for request_response in request_responses:
                request_output = GenerateStreamResponse(**json.loads(request_response.engine_output))
                request_timestamp: RequestTimestamps = request_response.request_timestamps
                set_timestamp(request_timestamp, 'api_server_get_queue_timestamp', time.time())
                request_id = request_output.req_id
                # Request could be dispatched twice when manager is dead, the first request will free
                # the request_streams when finished. Or the request is dropped already.
                if request_id not in self.request_streams:
                    continue
                self.request_instance[request_id] = request_response.instance_id

                if self.log_request_timestamps:
                    # Do not consider the out of order for request timestamp currently.
                    entrypoint_id = self.llumnix_id2entrypoint_id.get(request_id, None)
                    if entrypoint_id is not None:
                        self.timestamps_streams[entrypoint_id].put_nowait(request_timestamp)

                processed_output: List[GenerateStreamResponse] = self._process_output_order(request_id, request_output)
                if not processed_output:
                    continue
                for req in processed_output:
                    self.request_streams[request_id].put_nowait(req)
                last_output = processed_output[-1]
                self.request_streams_last_completion_tokens[request_id] = last_output.usage.completion_tokens
                if processed_output[-1].is_finished or not processed_output[-1].is_ok:
                    logger.debug("Client finish request {}, is_ok: {}, err_info: {}.".format(
                        request_id, last_output.is_ok, last_output.error_info))
                    self._clear_client_request_states(request_id)

    def _clear_client_request_states(self, request_id: str):
        entrypoint_id = self.llumnix_id2entrypoint_id.pop(request_id, -1)
        self.entrypoint_id2llumnix_id.pop(entrypoint_id, None)
        self.request_streams.pop(request_id, None)
        self.request_streams_last_completion_tokens.pop(request_id, None)
        self.request_streams_output_stash.pop(request_id, None)
        self.request_instance.pop(request_id, None)

    def _process_output_order(self, request_id: int, request_output: GenerateStreamResponse) -> List[GenerateStreamResponse]:
        current_completion_tokens = None
        if hasattr(request_output, 'usage'):
            current_completion_tokens = request_output.usage.completion_tokens

        if not current_completion_tokens:
            # No usage info, return the request_output directly.
            return [request_output]

        last_completion_tokens = self.request_streams_last_completion_tokens.get(request_id, 0)
        support_completion_tokens = last_completion_tokens + len(request_output.tokens)
        if current_completion_tokens > support_completion_tokens:
            # process the out-of-order output
            logger.info("request {} out-of-order output, last completion tokens is {}"
                        ", current completion tokens is {}, current tokens is {}, stash current output..."
                        .format(request_id,last_completion_tokens,current_completion_tokens,len(request_output.tokens)))
            if hasattr(request_output, 'request_timestamps'):
                logger.info("out-of-order request({}) output timestamps: {}".format(
                    request_id, request_output.request_timestamps.to_latency_breakdown_dict()))
            self.request_streams_output_stash.setdefault(request_id, []).append(request_output)
            return []

        if current_completion_tokens == support_completion_tokens:
            if not self.request_streams_output_stash.get(request_id, None):
                # no history output in stash
                return [request_output]

            # process the history output in buffer
            output_stash: List[GenerateStreamResponse] = self.request_streams_output_stash[request_id]
            output_stash.sort(key=lambda x: x.usage.completion_tokens)
            last_correct_output_index = 0
            for output in output_stash:
                if output.usage.completion_tokens > current_completion_tokens + len(output.tokens):
                    break
                last_correct_output_index += 1
                current_completion_tokens = output.usage.completion_tokens
            if last_correct_output_index == 0:
                return [request_output]
            res = [request_output] + output_stash[:last_correct_output_index]
            self.request_streams_output_stash[request_id] = output_stash[last_correct_output_index:]
            return res

        return [request_output]

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

    def connect(self):
        pass

    def close(self):
        pass

    async def get_stats(self) -> Stats:
        pass

    async def get_metrics(self) -> str:
        pass

    def support_beam_search(self) -> Tuple[bool, str]:
        return False, "Unsupported feature: multiple sequence decoding in Llumnix."

    async def start_profiler(self) -> None:
        pass

    async def stop_profiler(self) -> None:
        pass
