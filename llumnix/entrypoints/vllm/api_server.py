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
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.sampling_params import SamplingParams

from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.vllm.arg_utils import add_cli_args, get_args
from llumnix.entrypoints.vllm.client import LlumnixClientVLLM
from llumnix.logging.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.config import get_llumnix_config
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode, is_gpu_available
from llumnix.constants import SERVER_TIMEOUT_KEEP_ALIVE
from llumnix.metrics.timestamps import set_timestamp

# Code file with __main__ should set the logger name to inherit the llumnix logger configuration.
logger = init_logger("llumnix.entrypoints.vllm.api_server")

llumnix_client: LlumnixClientVLLM = None


# pylint: disable=unused-argument
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.create_task(llumnix_client.request_output_queue.run_server_loop())
    asyncio.create_task(llumnix_client.get_request_outputs_loop())
    yield
    llumnix_client.request_output_queue.cleanup()

app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    # Use LlumnixClientVLLM's generate and abort api to replace with vLLM AsyncLLMEngine's generate and abort api.
    results_generator = await llumnix_client.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await llumnix_client.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # Add some benchmark-related codes comparing to the generate API.
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    _ = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    start = time.time()

    results_generator = await llumnix_client.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    per_token_latency = []
    per_token_latency_breakdown_list = []
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await llumnix_client.abort(request_id)
            return Response(status_code=499)
        now = time.time()
        per_token_latency.append([now, (now - start)*1000])
        start = now
        final_output = request_output
        set_timestamp(request_output, 'api_server_generate_timestamp_end', now)
        if hasattr(request_output, 'request_timestamps'):
            per_token_latency_breakdown_list.append(request_output.request_timestamps.to_latency_breakdown_dict())
    assert final_output is not None

    if llumnix_client.log_requests:
        llumnix_client.num_finished_requests += 1
        logger.info("entrypoints finished request {}".format(request_id))
        logger.info("num_finished_requests {}".format(llumnix_client.num_finished_requests))

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
    }
    if per_token_latency_breakdown_list:
        ret['per_token_latency_breakdown_list'] = per_token_latency_breakdown_list
    return JSONResponse(ret)


@app.get("/is_ready")
async def is_ready() -> bool:
    return await llumnix_client.is_ready()


if __name__ == "__main__":
    parser: LlumnixArgumentParser = LlumnixArgumentParser()

    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)
    parser.add_argument("--log-level", type=str, choices=["debug", "info", "warning", "error"])

    cli_args = add_cli_args(parser)
    cfg = get_llumnix_config(cli_args.config_file, cli_args)
    entrypoints_args, manager_args, engine_args = get_args(cfg, parser, cli_args)

    backend_type = BackendType.VLLM if not manager_args.simulator_mode else BackendType.SIM_VLLM
    launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=backend_type)

    # Launch or connect to the ray cluster for multi-node serving.
    setup_ray_cluster(entrypoints_args)

    # if gpu is not available, it means that this node is head pod without any llumnix components
    if is_gpu_available():
        entrypoints_context = setup_llumnix(manager_args, entrypoints_args, engine_args, launch_args)
        llumnix_client = LlumnixClientVLLM(entrypoints_context)

        # Start the api server after all the components of llumnix are ready.
        logger.info("Start api server on '{}:{}'.".format(entrypoints_args.host, entrypoints_args.port))
        uvicorn.run(app,
                    host=entrypoints_args.host,
                    port=entrypoints_args.port,
                    log_level=entrypoints_args.log_level,
                    timeout_keep_alive=SERVER_TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=entrypoints_args.ssl_keyfile,
                    ssl_certfile=entrypoints_args.ssl_certfile)
