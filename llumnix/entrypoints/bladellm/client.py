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

import time
import asyncio
import copy
import random
from typing import Dict, List, Tuple, Optional

import msgspec
import ray

from blade_llm.service.communications.engine_client import MultiProcessingLLMClient
from blade_llm.service.communications.protocol_msgspec import Stats
from blade_llm.service.communications.response import LLMResponse
from blade_llm.service.args import ServingArgs
from blade_llm.protocol_msgspec import ServerRequest, GenerateStreamResponse
from blade_llm.service.communications.response import error_resp

from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.logging.logger import init_logger
from llumnix.constants import WAIT_MANAGER_INTERVAL
from llumnix.metrics.timestamps import set_timestamp
from llumnix.server_info import ServerInfo
from llumnix.utils import asyncio_wait_for_ray_remote_call_with_timeout
from llumnix.request_output import LlumnixRequestOuput as LlumnixRequestOuputBladeLLM
from llumnix.entrypoints.client import LlumnixClient

logger = init_logger(__name__)


class LlumnixClientBladeLLM(LlumnixClient, MultiProcessingLLMClient):
    def __init__(self,
                 args: ServingArgs,
                 entrypoints_context: EntrypointsContext,
                 loop: asyncio.AbstractEventLoop):
        MultiProcessingLLMClient.__init__(self, args, -1, -1)

        self.entrypoint_req_id_to_llumnix_req_id = {} # int32 -> int32
        self.llumnix_req_id_to_entrypoint_req_id = {} # int32 -> int32

        self.request_stream: Dict[int, asyncio.Queue] = {}
        self.request_stream_output_stash: Dict[int, List[GenerateStreamResponse]] = {}

        self.timestamps_stream: Dict[int, asyncio.Queue] = {}

        LlumnixClient.__init__(self, entrypoints_context, loop)

    def get_request_timestamps_generator(self, entrypoint_req_id: int) -> Optional[asyncio.Queue]:
        return self.timestamps_stream.get(entrypoint_req_id, None)

    async def _add_request(self, request: ServerRequest) -> LLMResponse:
        self.llumnix_client_metrics.add_request(reqeust_id=request.id)
        if request.sampling_params.n > 1 or request.sampling_params.use_beam_search:
            return error_resp(request.id, err_code=400, err_msg="Unsupported feature: multiple sequence decoding in Llumnix.")

        # To prevent different api_servers from generating the same request_id,
        # a random number is used to replace original req_id.
        llumnix_req_id = random.randint(1, (1 << 31) - 1)
        self.llumnix_req_id_to_entrypoint_req_id[llumnix_req_id] = request.id
        self.entrypoint_req_id_to_llumnix_req_id[request.id] = llumnix_req_id
        logger.info("request id is replaced from [{},{}] to {}".format(request.id, request.external_id, llumnix_req_id))
        internal_request = copy.deepcopy(request)
        internal_request.id = llumnix_req_id
        resp_stream = await self._generate(llumnix_req_id, self.msg_encoder.encode(internal_request))
        return resp_stream

    async def _generate(self, request_id: int, request: bytes) -> LLMResponse:
        logger.info("Client receive request {}.".format(request_id))
        results_queue = asyncio.Queue()
        self.request_stream[request_id] = results_queue
        if self.log_request_timestamps:
            entrypoint_req_id = self.llumnix_req_id_to_entrypoint_req_id.get(request_id, None)
            if entrypoint_req_id is not None:
                self.timestamps_stream[entrypoint_req_id] = asyncio.Queue()
        server_info_copy = copy.deepcopy(self.server_info)

        # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            await self._generate_by_manager(request_id, server_info_copy, request)
            self.manager_available = True
        # pylint: disable=broad-except
        except Exception as e:
            self._handle_generate_by_manager_error(request_id, e)
            # Do not re-generate the request to avoid duplicate requests.
            if self.manager_available:
                self.manager_available = False
                return LLMResponse(request_id, resp_queue=results_queue)
            await self._generate_by_instance(request_id, server_info_copy, request)

        return LLMResponse(request_id, resp_queue=results_queue)

    # pylint: disable=arguments-differ
    async def _generate_by_manager(self, request_id: int, server_info: ServerInfo, request: bytes):
        if self.log_request_timestamps:
            # Hack request timestamps in server_info for latency breakdown.
            server_info.request_timestamps = RequestTimestamps()
            set_timestamp(server_info, "api_server_generate_timestamp", time.time())
        # await to catch exception
        await asyncio_wait_for_ray_remote_call_with_timeout(
            self.manager.generate, str(request_id), server_info, server_request=request
        )

    # pylint: disable=arguments-differ
    async def _generate_by_instance(self, request_id: int, server_info: ServerInfo, request: bytes):
        try:
            if self.instance_num_requests:
                instance_id = min(self.instance_num_requests, key=self.instance_num_requests.get)
                self.instance_num_requests[instance_id] += 1
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    self.instances[instance_id].generate, request_id, server_info, -1, server_request=request
                )
                logger.warning("Manager is unavailable temporarily, "
                               "dispatch request {} to instance {}.".format(request_id, instance_id))
            else:
                logger.error("Manager is unavailable temporarily, but there is no instance behind this api server, "
                    "sleep {}s, waiting for manager available.".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                asyncio.create_task(self._generate(request_id, request))
        # pylint: disable=broad-except
        except Exception as e:
            if instance_id in self.instances:
                self._handle_generate_by_instance_error(request_id, instance_id, e)
            asyncio.create_task(self._generate(request_id, request))

    async def drop_request(self, req_id: int) -> None:
        llumnix_req_id = self.entrypoint_req_id_to_llumnix_req_id.get(req_id, None)
        self.llumnix_client_metrics.remove_request(request_id=req_id)
        await self._abort(llumnix_req_id)

    async def get_request_outputs_loop(self):
        while True:
            try:
                request_responses: List[LlumnixRequestOuputBladeLLM] = await self.request_output_queue.get()
                if request_responses is None:
                    continue

                for request_response in request_responses:
                    request_output = msgspec.convert(self.msg_decoder.decode(request_response.get_engine_output()),
                                                    type=GenerateStreamResponse)
                    request_id = request_output.req_id
                    request_timestamp: RequestTimestamps = request_response.request_timestamps
                    set_timestamp(request_timestamp, 'api_server_get_queue_timestamp', time.time())
                    # Request could be dispatched twice when manager is dead, the first request will free
                    # the request_streams when finished. Or the request is dropped already.
                    if request_id not in self.request_stream:
                        continue
                    self.request_instance[request_id] = request_response.instance_id

                    if self.log_request_timestamps:
                        # Do not consider the out of order for request timestamp currently.
                        entrypoint_req_id = self.llumnix_req_id_to_entrypoint_req_id.get(request_id, None)
                        if entrypoint_req_id is not None:
                            self.timestamps_stream[entrypoint_req_id].put_nowait(request_timestamp)

                    processed_output: List[GenerateStreamResponse] = self._process_output_order(request_id, request_output)
                    if not processed_output:
                        continue
                    for req in processed_output:
                        self.llumnix_client_metrics.observe_tpot_and_ttft(
                            request_id=self.llumnix_req_id_to_entrypoint_req_id.get(
                                request_id, None
                            )
                        )
                        self.request_stream[request_id].put_nowait(req)
                    last_output = processed_output[-1]
                    self.request_stream_last_completion_tokens[request_id] = last_output.usage.completion_tokens
                    if processed_output[-1].is_finished or not processed_output[-1].is_ok:
                        logger.debug("Client finished request {}, is_ok: {}, err_info: {}.".format(
                            request_id, last_output.is_ok, last_output.error_info))
                        self._clear_client_request_states(request_id)
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Client get error in get_request_outputs_loop, client keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    def _clear_client_request_states(self, request_id: int):
        super()._clear_client_request_states(request_id)
        entrypoint_req_id = self.llumnix_req_id_to_entrypoint_req_id.pop(request_id, -1)
        self.entrypoint_req_id_to_llumnix_req_id.pop(entrypoint_req_id, None)
        self.request_stream.pop(request_id, None)
        self.request_stream_output_stash.pop(request_id, None)
        self.llumnix_client_metrics.remove_request(request_id=entrypoint_req_id)

    def _process_output_order(self, request_id: int, request_output: GenerateStreamResponse) -> List[GenerateStreamResponse]:
        current_completion_tokens = None
        if hasattr(request_output, 'usage'):
            current_completion_tokens = request_output.usage.completion_tokens

        if not current_completion_tokens:
            # No usage info, return the request_output directly.
            return [request_output]

        last_completion_tokens = self.request_stream_last_completion_tokens.get(request_id, 0)
        support_completion_tokens = last_completion_tokens + len(request_output.tokens)
        if current_completion_tokens > support_completion_tokens:
            # process the out-of-order output
            logger.info("request {} out-of-order output, last completion tokens is {}"
                        ", current completion tokens is {}, current tokens is {}, stash current output..."
                        .format(request_id,last_completion_tokens,current_completion_tokens,len(request_output.tokens)))
            if hasattr(request_output, 'request_timestamps'):
                logger.info("out-of-order request({}) output timestamps: {}".format(
                    request_id, request_output.request_timestamps.to_latency_breakdown_dict()))
            self.request_stream_output_stash.setdefault(request_id, []).append(request_output)
            return []

        if current_completion_tokens == support_completion_tokens:
            if not self.request_stream_output_stash.get(request_id, None):
                # no history output in stash
                return [request_output]

            # process the history output in buffer
            output_stash: List[GenerateStreamResponse] = self.request_stream_output_stash[request_id]
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
            self.request_stream_output_stash[request_id] = output_stash[last_correct_output_index:]
            return res

        return [request_output]

    def connect(self):
        pass

    def close(self):
        pass

    async def get_stats(self) -> Stats:
        # TODO(KuilongCui): Add get_stats implementation.
        return Stats()

    async def get_metrics(self) -> str:
        instance = self.instances[list(self.instances.keys())[0]]
        metrics = ray.get(instance.execute_engine_method.remote("get_metrics"))
        return metrics

    def support_beam_search(self) -> Tuple[bool, str]:
        return False, "Llumnix does not support multiple sequences decoding."

    async def start_profiler(self) -> None:
        instance = self.instances[list(self.instances.keys())[0]]
        return await instance.execute_engine_method_async.remote("start_profiler")

    async def stop_profiler(self) -> None:
        instance = self.instances[list(self.instances.keys())[0]]
        return await instance.execute_engine_method_async.remote("start_profiler")
