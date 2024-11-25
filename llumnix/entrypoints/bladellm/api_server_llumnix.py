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

import ray
import copy
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
from llumnix.arg_utils import LlumnixArgumentParser
from llumnix.entrypoints.utils import (setup_ray_cluster,
                                       setup_llumnix,
                                       is_gpu_available,
                                       LlumnixEntrypointsContext,
                                       init_per_token_latency_breakdown_dict,
                                       record_per_token_latency_breakdown)
from llumnix.entrypoints.bladellm.utils import (add_cli_args_llumnix, _background_process_outputs,
                                            get_args_llumnix,manager_generate,
                                            manager_abort,
                                            manager_is_ready)
from llumnix.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.config import get_llumnix_config, LlumnixConfig
from llumnix.backends.backend_interface import BackendType
from llumnix.backends.utils import random_positive_id
from llumnix.logger import init_logger
from llumnix.entrypoints.utils import LlumnixEntrypointsContext
from llumnix.server_info import RequestTimestamps


# Code file with __main__ should set the logger name to inherit the llumnix logger configuration.
logger = init_logger("llumnix.entrypoints.bladellm.api_server")

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

@app.get("/is_ready")
async def is_ready():
    return await manager_is_ready(llumnix_context)

if __name__ == "__main__":
    parser = add_args()

    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)

    cli_args = add_cli_args_llumnix(parser)
    logger.info('================ Serving Arguments ================')
    for k, v in cli_args.__dict__.items():
        logger.info(f"{k:>20}: {v}")
    cfg: LlumnixConfig = get_llumnix_config(cli_args.config_file, cli_args)
    _, engine_manager_args, engine_args = get_args_llumnix(cfg, parser, cli_args)

    model_conf = load_config(engine_args.load_model_options.model)

    # check port first
    check_ports(engine_args)

    init_metric(
        engine_args.serving_metric_options.metric_export_interval_sec,
        *engine_args.metric_exporters,
    )

    # Launch or connect to the ray cluster for multi-node serving.
    setup_ray_cluster(cfg)

    # if gpu is not available, it means that this node is head pod without any llumnix components
    if is_gpu_available():
        llumnix_context = setup_llumnix(cfg, engine_manager_args, BackendType.BLADELLM, 1, engine_args)
        # Start the api server after all the components of llumnix are ready.
        logger.info("Start Api Server on '{}:{}'".format(cfg.SERVER.HOST, cfg.SERVER.PORT))
        uvicorn.run(app,
                    host=cfg.SERVER.HOST,
                    port=cfg.SERVER.PORT,
                    log_level="debug",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=cfg.SERVER.SSL_KEYFILE,
                    ssl_certfile=cfg.SERVER.SSL_CERTFILE)
