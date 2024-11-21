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

from typing import AsyncGenerator
from contextlib import asynccontextmanager
import time
import asyncio
import sys
import json
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from aiohttp import web

from blade_llm.model.config_utils import GenerationConfigProcessor
from blade_llm.service.server import Entrypoint
from blade_llm.protocol import GenerateStreamResponse
import time
from typing import Optional, Tuple
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from blade_llm.service.args import ServingArgs, add_args
from blade_llm.model.config_utils import load_config
from blade_llm.service.metric import init_metric
from blade_llm.service.server import check_ports
from blade_llm.protocol import GenerateRequest, SamplingParams, StoppingCriteria
from blade_llm.service.clients.base_client import AsyncRespStreamer
from blade_llm.model.config_base import ConfigBase
from blade_llm.service.args import ServingArgs
from blade_llm.service.clients import GeneralLLMClient
from blade_llm.service.schedulers import (
    ContinuousBatchingScheduler,
    DynamicBatchingScheduler,
)
from blade_llm.service.schedulers.scheduler_factory import _SCHEDULER_MAP

from blade_llm.service.clients import BaseLLMClient, LLMResponse
from blade_llm.protocol import ServerRequest

from llumnix.entrypoints.bladellm.utils import manager_generate, manager_abort
from llumnix.arg_utils import LlumnixArgumentParser
from llumnix.entrypoints.utils import (setup_ray_cluster,
                                       setup_llumnix,
                                       is_gpu_available,
                                       LlumnixEntrypointsContext,
                                       init_per_token_latency_breakdown_dict,
                                       record_per_token_latency_breakdown)
from llumnix.entrypoints.bladellm.utils import (add_cli_args, _background_process_outputs,
                                            get_args)
from llumnix.backends.bladellm.utils import get_model_conf
from llumnix.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.config import get_llumnix_config, LlumnixConfig
from llumnix.backends.backend_interface import BackendType
from llumnix.backends.utils import random_positive_id

# Code file with __main__ should set the logger name to inherit the llumnix logger configuration.
logger = init_logger("llumnix.entrypoints.vllm.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.

llumnix_context: LlumnixEntrypointsContext = None

# pylint: disable=unused-argument
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.create_task(llumnix_context.request_output_queue.run_server_loop())
    asyncio.create_task(_background_process_outputs(llumnix_context))
    yield
    llumnix_context.request_output_queue.cleanup()

app = FastAPI(lifespan=lifespan)


class DummyAsyncLLMEngineClient():
    async def add_request(self, request: ServerRequest) -> LLMResponse:
        measure = MeasureEntrypoint(request.id, time.time(), request.stopping_criterial.max_new_tokens)
        resp_stream = await manager_generate(request, request.id, llumnix_context)
        return resp_stream, measure
    
    async def drop_request(self, request_id: int):
        await manager_abort(request_id, llumnix_context)

class GeneralLLMClientLlumnix(GeneralLLMClient):
    def __init__(self, args: ServingArgs, client: BaseLLMClient, model_conf: Optional[ConfigBase] = None):
        super().__init__(args, client, model_conf)
        self.scheduler = _SCHEDULER_MAP[args.decode_algo if args.use_lookahead else args.load_model_options.attn_cls]

    def support_beam_search(self):
        if self.args.pipeline_parallel_size > 1 or not self.scheduler == ContinuousBatchingScheduler:
            return (
                False,
                "beam_search can only used with continuous_batching scheduler and pipeline disabled.",
            )
        else:
            return True, ""
    
    def support_chat_stream(self) -> Tuple[bool, str]:
        if self.scheduler == DynamicBatchingScheduler:
            return False, "DynamicBatchingScheduler not support chat_stream"
        else:
            return True, ""


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

# Aiohttp App
async def start_aiohttp_server():
    # Wrap FastAPI app as ASGI middleware for aiohttp
    aiohttp_app = web.Application()
    aiohttp_app.router.add_route("*", "/{path_info:.*}", WSGIMiddleware(app))

    # Start aiohttp server
    return aiohttp_app

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

# Inherit the methods from Blade_llm.Entrypoint. Wrap the class to collect metrics for llumnix.
class EntrypointLlumnix(Entrypoint):
    async def hello(self, request: Request = None):
        web_resp = await super().hello(request)
        content = web_resp.text
        return JSONResponse({"message": content})
    
    async def generate_stream(self, request: Request):
        request_dict = await request.json()
        prompt = request_dict.pop("prompt")
        max_new_tokens = request_dict.pop("max_new_tokens")
        ignore_eos = request_dict.pop("ignore_eos")
        user_req = GenerateRequest(
            prompt=prompt,
            sampling_params=SamplingParams(**request_dict),
            stopping_criterial=StoppingCriteria(
                max_new_tokens=max_new_tokens,
                ignore_eos=ignore_eos,
            ),
        )
        server_request = user_req.to_server_request()
        server_request.id = random_positive_id()
        request_id = server_request.id
        start = time.time()

        results_generator = await manager_generate(server_request, request_id, llumnix_context)

        per_token_latency = []
        start = time.time()

        # Non-streaming case
        texts = []
        final_output = None
        per_token_latency = []
        per_token_latency_breakdown_dict = init_per_token_latency_breakdown_dict()
        stream = results_generator.async_stream()
        async for request_output in stream:
            print("request_output",request_output)
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await manager_abort(request_id, llumnix_context)
                return Response(status_code=499)
            now = time.time()
            per_token_latency.append([now, (now - start)*1000])
            start = now
            texts.extend([t.text for t in request_output.tokens])
            final_output = request_output
            if hasattr(request_output, 'request_timestamps'):
                request_output.request_timestamps.api_server_generate_benchmark_timestamp_end = now
                record_per_token_latency_breakdown(per_token_latency_breakdown_dict, request_output.request_timestamps)
        assert final_output is not None

        if True:#llumnix_context.log_requests:
            llumnix_context.num_finished_requests += 1
            logger.info("Finished request {}.".format(request_id))
            logger.info("num_finished_requests {}.".format(llumnix_context.num_finished_requests))

        generation = final_output.outputs[0].text
        num_output_tokens = len(final_output.outputs[0].token_ids)
        num_input_tokens = len(final_output.prompt_token_ids)
        expected_resp_len = request_dict['max_tokens']
        if not max(expected_resp_len, 1) == max(num_output_tokens, 1):
            "request_id={}, expected_resp_len={}, num_output_tokens={}, num_input_tokens={}".format(
                request_id, expected_resp_len, num_output_tokens, num_input_tokens)
        ret = {
            'request_id': request_id,
            'generated_text': generation,
            'num_output_tokens_cf': num_output_tokens,
            'per_token_latency': per_token_latency,
            'per_token_latency_breakdown_dict': per_token_latency_breakdown_dict
        }
        return JSONResponse(ret)


    def create_web_app(self):
        global app
        app.add_api_route("/", self.hello, methods=["GET"])
        # app.add_api_route("/generate", self.generate, methods=["POST"])
        app.add_api_route("/generate_stream", self.generate_stream, methods=["GET"])
        # app.add_api_route("/chat_stream", self.chat_stream, methods=["GET"])
        # app.add_api_route("/metric", self.metric, methods=["GET"])
        # app.add_api_route("/v1/completions", self.oai_completions, methods=["POST"])
        # app.add_api_route("/v1/chat/completions", self.oai_chat_completions, methods=["POST"])
        print("create_web_app",app,type(app))
        return app



def setup_llumnix_api_server(bladellm_args):
    llumnixParser: LlumnixArgumentParser = LlumnixArgumentParser()

    llumnixParser.add_argument("--ssl-keyfile", type=str)
    llumnixParser.add_argument("--ssl-certfile", type=str)

    add_cli_args(llumnixParser)
    # TODO[xinyi]: now only support use config_file to define llumnix arguments, `llumnix_config` in BladeLLM is the same as the usage in LlumnixEntrypointsArgs
    # TODO[xinyi]: support read from bladellm_args.llumnix_config
    llumnixCfg: LlumnixConfig = get_llumnix_config(bladellm_args.llumnix_config)
    _, engine_manager_args, engine_args = get_args(llumnixCfg, llumnixParser, bladellm_args)
    setup_ray_cluster(llumnixCfg)

    # if gpu is not available, it means that this node is head pod x any llumnix components
    if is_gpu_available():
        world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
        global llumnix_context
        llumnix_context = setup_llumnix(llumnixCfg, engine_manager_args, BackendType.BLADELLM, world_size, engine_args)
        engine_model_conf = get_model_conf(bladellm_args)
        llm_client = GeneralLLMClientLlumnix(bladellm_args, DummyAsyncLLMEngineClient(), engine_model_conf)
        return engine_model_conf, llm_client, EntrypointLlumnix
    return None, None, None