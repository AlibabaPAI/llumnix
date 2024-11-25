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
import json
from typing import Tuple

import ray

from blade_llm.service.clients import BaseLLMClient, LLMResponse
from blade_llm.protocol import ServerRequest, GenerateStreamResponse
from blade_llm.utils.status import Stats
from blade_llm.service.schedulers import DynamicBatchingScheduler

from llumnix.server_info import RequestTimestamps
from llumnix.entrypoints.utils import LlumnixEntrypointsContext
from llumnix.backends.bladellm.sequence import GenerateStreamResponseLlumnix
from llumnix.entrypoints.utils import (
    init_per_token_latency_breakdown_dict,
    record_per_token_latency_breakdown,
)
from llumnix.logger import init_logger


logger = init_logger(__name__)

WAIT_MANAGER_INTERVAL = 5


async def manager_generate(request,
                           request_id: str,
                           llumnix_context: LlumnixEntrypointsContext) -> LLMResponse:
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
        await llumnix_context.engine_manager.generate.remote(request_id, server_info_copy, request)
        llumnix_context.manager_available = True
    except ray.exceptions.RayActorError:
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

async def background_process_outputs(llumnix_context):
    try:
        while True:
            request_outputs = await llumnix_context.request_output_queue.get()
            if request_outputs is None:
                continue
            for request_output in request_outputs:
                if hasattr(request_output, 'request_timestamps'):
                    request_output.request_timestamps.api_server_background_process_get_queue_timestamp = time.time()
            for request_output in request_outputs:
                request_output = GenerateStreamResponseLlumnix(**json.loads(request_output))
                request_id = request_output.request_id
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

class MeasureEntrypoint:
    def __init__(self, request_id, start_time, expected_resp_len):
        self.request_id = request_id
        self.final_output = None
        self.per_token_latency = []
        self.generation_text = []
        self.per_token_latency_breakdown_dict = init_per_token_latency_breakdown_dict()
        self.start_time = start_time
        self.expected_resp_len = expected_resp_len
        self.final_output = None
    
    @property
    def generation(self) -> bool:
        return f"{''.join(self.generation_text)}"


def measure_single_resp(resp: GenerateStreamResponse, measure: MeasureEntrypoint):
        now = time.time()
        measure.per_token_latency.append([now, (now - measure.start_time)*1000])
        measure.start_time = now
        measure.generation_text.extend([t.text for t in resp.tokens])
        measure.final_output = resp
        if hasattr(resp, 'request_timestamps'):
            resp.request_timestamps.api_server_generate_benchmark_timestamp_end = now
            record_per_token_latency_breakdown(measure.per_token_latency_breakdown_dict, resp.request_timestamps)

def measure_resp(measure_handle: MeasureEntrypoint):
    final_output = measure_handle.final_output
    assert final_output is not None
    from llumnix.entrypoints.bladellm.api_server import llumnix_context
    if llumnix_context.log_requests:
        llumnix_context.num_finished_requests += 1
        logger.info("Finished request {}.".format(measure_handle.request_id))
        logger.info("num_finished_requests {}.".format(llumnix_context.num_finished_requests))

    num_output_tokens = len(final_output.usage.completion_tokens)
    num_input_tokens = len(final_output.usage.prompt_tokens)
    if not max(measure_handle.expected_resp_len, 1) == max(num_output_tokens, 1):
        "request_id={}, expected_resp_len={}, num_output_tokens={}, num_input_tokens={}".format(
            measure_handle.request_id, measure_handle.expected_resp_len, num_output_tokens, num_input_tokens)
    ret = {
        'request_id': measure_handle.request_id,
        'generated_text': measure_handle.generation,
        'num_output_tokens_cf': num_output_tokens,
        'per_token_latency': measure_handle.per_token_latency,
        'per_token_latency_breakdown_dict': measure_handle.per_token_latency_breakdown_dict
    }

class AsyncLLMEngineClientLlumnix(BaseLLMClient):
    def __init__(self, scheduler_cls):
        super().__init__()
        self.scheduler_cls = scheduler_cls

    async def add_request(self, request: ServerRequest) -> LLMResponse:
        measure = MeasureEntrypoint(request.id, time.time(), request.stopping_criterial.max_new_tokens)
        from llumnix.entrypoints.bladellm.api_server import llumnix_context
        resp_stream = await manager_generate(request.model_dump_json(), request.id, llumnix_context)
        return resp_stream, measure
    
    async def drop_request(self, request_id: int):
        from llumnix.entrypoints.bladellm.api_server import llumnix_context
        await manager_abort(request_id, llumnix_context)
    
    def support_beam_search(self):
        return (False, "llumnix currently doesn't support beam_search.")
    
    # TODO(KuilongCui): maybe there is a better way to check this
    def support_chat_stream(self) -> Tuple[bool, str]:
        if self.scheduler_cls == DynamicBatchingScheduler:
            return False, "DynamicBatchingScheduler not support chat_stream"
        else:
            return True, ""

    # TODO(kuilongCui): check get_stats
    def get_stats(self) -> Stats:
        pass
        # return self._client.get_stats()

    async def connect(self):
        pass

    async def close(self):
        pass

