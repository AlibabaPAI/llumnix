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

from llumnix.arg_utils import LlumnixArgumentParser
from llumnix.entrypoints.utils import (setup_ray_cluster,
                                       setup_llumnix,
                                       is_gpu_available,
                                       LlumnixEntrypointsContext,
                                       _background_process_outputs,
                                       init_per_token_latency_breakdown_dict,
                                       record_per_token_latency_breakdown)
from llumnix.entrypoints.vllm.utils import (add_cli_args,
                                            get_args,
                                            manager_generate,
                                            manager_abort,
                                            manager_is_ready)
from llumnix.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.config import get_llumnix_config, LlumnixConfig

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

    # Use manager_generate and manager_abort to replace with vllm async engine generate and abort api.
    results_generator = await manager_generate(prompt, sampling_params, request_id, llumnix_context)

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
            await manager_abort(request_id, llumnix_context)
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

    results_generator = await manager_generate(prompt, sampling_params, request_id, llumnix_context)

    # Non-streaming case
    final_output = None
    per_token_latency = []
    per_token_latency_breakdown_dict = init_per_token_latency_breakdown_dict()
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await manager_abort(request_id, llumnix_context)
            return Response(status_code=499)
        now = time.time()
        per_token_latency.append([now, (now - start)*1000])
        start = now
        final_output = request_output
        if hasattr(request_output, 'request_timestamps'):
            request_output.request_timestamps.api_server_generate_benchmark_timestamp_end = now
            record_per_token_latency_breakdown(per_token_latency_breakdown_dict, request_output.request_timestamps)
    assert final_output is not None

    if llumnix_context.log_requests:
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
    parser: LlumnixArgumentParser = LlumnixArgumentParser()

    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)

    cli_args = add_cli_args(parser)
    cfg: LlumnixConfig = get_llumnix_config(cli_args.config_file, cli_args)
    _, engine_manager_args, engine_args = get_args(cfg, parser, cli_args)

    # Launch or connect to the ray cluster for multi-node serving.
    setup_ray_cluster(cfg)

    # if gpu is not available, it means that this node is head pod without any llumnix components
    if is_gpu_available():
        llumnix_context = setup_llumnix(engine_manager_args, engine_args, cfg)
        # Start the api server after all the components of llumnix are ready.
        logger.info("Start Api Server on '{}:{}'".format(cfg.SERVER.HOST, cfg.SERVER.PORT))
        uvicorn.run(app,
                    host=cfg.SERVER.HOST,
                    port=cfg.SERVER.PORT,
                    log_level="debug",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=cfg.SERVER.SSL_KEYFILE,
                    ssl_certfile=cfg.SERVER.SSL_CERTFILE)


# TODO[xinyi]: revise in bladellm repo
def main():
    parser = add_args()
    args = parser.parse_args()
    args = ServingArgs.from_cli_args(args)

    # Check whether FP8 paged kvcache quant is appointed to use and could be imported under current arch.
    # If not, fallback to non-quant kvcache.
    if (
        args.load_model_options.kv_cache_quant
        in ['fp8_e5m2', 'fp8_e4m3', "mix_f852i4", "mix_f843i4", "mix_i8i4", "mix_i4i4"]
        and not fp8_paged_enabled()
    ):
        logger.warning(
            "Experimental feature FP8 KV-Cache could not be imported, architecture may be incompatible, fallback to non-quant KV-Cache."
        )
        args.load_model_options.kv_cache_quant = 'no_quant'

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.info('================ Serving Arguments ================')
    for k, v in args.__dict__.items():
        logger.info(f"{k:>20}: {v}")

    # check port first
    check_ports(args)

    init_metric(args.serving_metric_options.metric_export_interval_sec, *args.metric_exporters)

    llm_engine = AsyncLLMEngine(args)

    try:
        generation_conf_processor = GenerationConfigProcessor(args.generation_configs, llm_engine.model_conf)
    except Exception:
        logger.exception('Failed to load generation config processor when create server.')
        generation_conf_processor = None

    loop = asyncio.get_event_loop()
    llm_engine.start(loop)
    llm_client = llm_engine.get_client()

    # start entrypoint server
    web_app = Entrypoint(
        client=llm_client,
        model_conf=llm_engine.model_conf,
        generation_conf_processor=generation_conf_processor,
        chat_template_path=args.load_model_options.chat_template,
        pp_enabled=args.pipeline_parallel_size > 1,
    ).create_web_app()
    logger.info(f"Entrypoint API ready at {args.host}:{args.port}")
    web.run_app(web_app, host=args.host, port=args.port, loop=loop)
