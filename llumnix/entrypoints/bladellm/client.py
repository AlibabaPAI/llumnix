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

import ray

from blade_llm.service.communications.engine_client import MultiProcessingLLMClient
from blade_llm.service.communications.protocol import Stats
from blade_llm.service.communications.response import LLMResponse
from blade_llm.service.args import ServingArgs
from blade_llm.protocol import ServerRequest
from blade_llm.protocol import ServerRequest, GenerateStreamResponse
from blade_llm.service.communications.response import error_resp

from llumnix.server_info import RequestTimestamps
from llumnix.entrypoints.utils import LlumnixEntrypointsContext
from llumnix.backends.bladellm.sequence import GenerateStreamResponseLlumnix
from llumnix.logger import init_logger
from llumnix.utils import random_uuid

logger = init_logger(__name__)

WAIT_MANAGER_INTERVAL = 5


async def manager_generate(request, request_id: str, llumnix_context: LlumnixEntrypointsContext) -> LLMResponse:
    logger.info("Client Add request: {}".format(request))

    results_queue = asyncio.Queue()
    llumnix_context.request_streams[request_id] = results_queue

    # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
    # If manager is unavailable, request will be directly added to the llumlet held by api server.
    try:
        server_info_copy = copy.deepcopy(llumnix_context.server_info)
        if llumnix_context.log_request_timestamps:
            # Hack request timestamps in server_info for latency breakdown.
            server_info_copy.request_timestamps = RequestTimestamps()
            server_info_copy.request_timestamps.api_server_manager_generate_timestamp = time.time()
        # await to catch exception
        await llumnix_context.engine_manager.generate.remote(str(request_id), server_info_copy, request)
        llumnix_context.manager_available = True
    except Exception as e:
        logger.error("Error in manager generate: {}".format(e))
        # Do not re-generate the request to avoid duplicate requests.
        if llumnix_context.manager_available:
            llumnix_context.manager_available = False
            return LLMResponse(request_id, resp_queue=results_queue)
        try:
            if llumnix_context.instance_num_requests:
                instance_id = min(llumnix_context.instance_num_requests, key=llumnix_context.instance_num_requests.get)
                llumnix_context.instance_num_requests[instance_id] += 1
                # TODO[xinyi]: set expected step here
                await llumnix_context.instances[instance_id].generate.remote(request_id, server_info_copy, -1, request)
                logger.info("Manager is unavailable, directly pass request {} to instance {}".format(request_id, instance_id))
            else:
                logger.info("Manager is unavailable, but there is no instance behind this api server, "
                      "sleep {}s, waiting for manager restarts".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                return await asyncio.create_task(manager_generate(request, request_id, llumnix_context))
        except (ray.exceptions.RayActorError, KeyError):
            if instance_id in llumnix_context.instances:
                logger.info("[manager_generate] instance {} is dead".format(instance_id))
                del llumnix_context.instances[instance_id]
                del llumnix_context.instance_num_requests[instance_id]
                return await asyncio.create_task(manager_generate(request, request_id, llumnix_context))
    return LLMResponse(request_id, resp_queue=results_queue)

# TODO[xinyi]: the same to the function in vllm.utils
async def manager_abort(request_id: str, llumnix_context: LlumnixEntrypointsContext) -> None:
    try:
        logger.info("abort request: {}.".format(request_id))
        await llumnix_context.engine_manager.abort.remote(request_id)
    except ray.exceptions.RayActorError:
        logger.info("Manager is unavailable")

async def manager_is_ready(llumnix_context: LlumnixEntrypointsContext):
    ready_status = await llumnix_context.engine_manager.is_ready.remote()
    return ready_status

async def background_process_outputs(llumnix_context: LlumnixEntrypointsContext):
    try:
        while True:
            request_outputs = await llumnix_context.request_output_queue.get()
            if request_outputs is None:
                continue
            for request_output in request_outputs:
                if hasattr(request_output, 'request_timestamps'):
                    request_output.request_timestamps.api_server_background_process_get_queue_timestamp = time.time()
            for request_output in request_outputs:
                request_data = json.loads(request_output)
                request_output = GenerateStreamResponseLlumnix(
                    resp=GenerateStreamResponse(**request_data),
                    request_id=request_data['request_id'],
                )
                request_id = str(request_output.request_id)
                # Request could be dispatched twice when manager is dead, the first request will free the request_streams when finished.
                if request_id not in llumnix_context.request_streams:
                    continue
                await llumnix_context.request_streams[request_id].put(request_output)
                if request_output.is_finished:
                    del llumnix_context.request_streams[request_id]
    except Exception as e:
        import traceback
        logger.error("Error in engine loop: {}".format(e))
        logger.error("exception traceback: {}".format(traceback.format_exc()))

class AsyncLLMEngineClientLlumnix(MultiProcessingLLMClient):
    def __init__(self, args: ServingArgs):
        super().__init__(args, -1)

    def connect(self):
        pass

    def close(self):
        pass
    
    async def _add_request(self, request: ServerRequest) -> LLMResponse:
        if request.sampling_params.n > 1 or request.sampling_params.use_beam_search:
            return error_resp(request.id, err_code=400, err_msg="Unsupported feature: multiple sequence decoding in Llumnix.")
    
        from llumnix.entrypoints.bladellm.api_server import llumnix_context
        request.id = random.randint(0, 2147483647) # 1<<31-1
        resp_stream = await manager_generate(request.model_dump_json(), str(request.id), llumnix_context)
        return resp_stream

    async def drop_request(self, req_id: int):
        from llumnix.entrypoints.bladellm.api_server import llumnix_context
        await manager_abort(req_id, llumnix_context)

    async def get_stats(self) -> Stats:
        pass

    async def get_metrics(self) -> str:
        pass

    def start_profiler(self) -> None:
        pass

    def stop_profiler(self) -> None:
        pass
