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

from typing import Dict, AsyncGenerator
from contextlib import asynccontextmanager
import argparse
import time
import asyncio
import json
import copy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import ray

from vllm.sampling_params import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncStream

from llumnix.arg_utils import EngineManagerArgs
from llumnix.server_info import ServerInfo, RequestTimestamps
from llumnix.entrypoints.llumnix_utils import (get_ip_address,
                                               launch_ray_cluster, connect_to_ray_cluster,
                                               is_gpu_available, init_llumnix_components)
from llumnix.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.backends.vllm.utils import check_engine_args
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.queue.utils import get_output_queue_server
from llumnix.config import get_llumnix_config, LlumnixConfig

logger = init_logger("llumnix.api_server")

engine_manager = None
instances = {}
instance_num_requests: Dict[str, int] = {}
# request_output_queue could be None if initialzed in lifespan.
request_output_queue: QueueServerBase = None
server_info = None
TIMEOUT_KEEP_ALIVE = 5  # seconds.
request_streams: Dict[str, AsyncStream] = {}
log_requests = None
log_request_timestamps = None
num_finished_requests = 0
WAIT_MANAGER_INTERVAL = 5
manager_available = True


async def _background_process_outputs():
    while True:
        request_outputs = await request_output_queue.get()
        for request_output in request_outputs:
            if hasattr(request_output, 'request_timestamps'):
                request_output.request_timestamps.api_server_background_process_get_queue_timestamp = time.time()
        for request_output in request_outputs:
            request_id = request_output.request_id
            # Request could be dispatched twice when manager is dead, the first request will free the request_streams when finished.
            if request_id not in request_streams:
                continue
            request_streams[request_id].put(request_output)
            if request_output.finished:
                request_streams[request_id].finish()
                del request_streams[request_id]

# pylint: disable=unused-argument
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.create_task(request_output_queue.run_server_loop())
    asyncio.create_task(_background_process_outputs())
    yield
    request_output_queue.cleanup()

app = FastAPI(lifespan=lifespan)

async def manager_generate(prompt, sampling_params, request_id) -> AsyncStream:
    if sampling_params.n > 1 or sampling_params.use_beam_search:
        raise ValueError("Unsupported feature: multiple sequence decoding")
    results_generator = AsyncStream(request_id)
    request_streams[request_id] = results_generator
    # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
    # If manager is unavailable, request will be directly added to the llumlet held by api server.
    global manager_available
    try:
        server_info_copy = copy.deepcopy(server_info)
        if log_request_timestamps:
            # Hack request timestamps in server_info for latency breakdown.
            server_info_copy.request_timestamps = RequestTimestamps()
            server_info_copy.request_timestamps.api_server_manager_generate_timestamp = time.time()
        # await to catch exception
        await engine_manager.generate.remote(request_id, server_info_copy, prompt, sampling_params)
        manager_available = True
    except ray.exceptions.RayActorError:
        # Do not re-generate the request to avoid duplicate requests.
        if manager_available:
            manager_available = False
            return results_generator
        try:
            if instance_num_requests:
                instance_id = min(instance_num_requests, key=instance_num_requests.get)
                instance_num_requests[instance_id] += 1
                await instances[instance_id].generate.remote(request_id, server_info, prompt, sampling_params)
                logger.info("Manager is unavailable, directly pass request {} to instance {}".format(request_id, instance_id))
            else:
                logger.info("Manager is unavailable, but there is no instance behind this api server, "
                      "sleep {}s, waiting for manager restarts".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                return await asyncio.create_task(manager_generate(prompt, sampling_params, request_id))
        except (ray.exceptions.RayActorError, KeyError):
            if instance_id in instances:
                logger.info("[manager_generate] instance {} is dead".format(instance_id))
                del instances[instance_id]
                del instance_num_requests[instance_id]
            return await asyncio.create_task(manager_generate(prompt, sampling_params, request_id))
    return results_generator

async def manager_abort(request_id: str) -> None:
    try:
        logger.info("abort request: {}.".format(request_id))
        await engine_manager.abort.remote(request_id)
    except ray.exceptions.RayActorError:
        logger.info("Manager is unavailable")


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

    results_generator = await manager_generate(prompt, sampling_params, request_id)

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
            await manager_abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

def init_per_token_latency_breakdown_dict() -> Dict[str, int]:
    per_token_latency_breakdown_dict = {
        'step_latency_engine': [],
        'process_model_outputs_latency': [],
        'step_postprocess_latency': [],
        'across_async_put_queue_thread_latency': [],
        'across_async_put_queue_actor_latency': [],
        'queue_rpc_latency': [],
        'background_process_get_queue_latency': [],
        'generate_benchmark_return_output_latency': []
    }
    return per_token_latency_breakdown_dict

def record_per_token_latency_breakdown(per_token_latency_breakdown_dict: Dict[str, int], request_timestamps: RequestTimestamps):
    per_token_latency_breakdown_dict['step_latency_engine'].append(request_timestamps.step_latency_engine)
    per_token_latency_breakdown_dict['process_model_outputs_latency'].append(request_timestamps.process_model_outputs_latency)
    per_token_latency_breakdown_dict['step_postprocess_latency'].append(request_timestamps.step_postprocess_latency)
    per_token_latency_breakdown_dict['across_async_put_queue_thread_latency'].append(request_timestamps.across_async_put_queue_thread_latency)
    per_token_latency_breakdown_dict['across_async_put_queue_actor_latency'].append(request_timestamps.across_async_put_queue_actor_latency)
    per_token_latency_breakdown_dict['queue_rpc_latency'].append(request_timestamps.queue_rpc_latency)
    per_token_latency_breakdown_dict['background_process_get_queue_latency'].append(request_timestamps.background_process_get_queue_latency)
    per_token_latency_breakdown_dict['generate_benchmark_return_output_latency'].append(request_timestamps.generate_benchmark_return_output_latency)

@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    _ = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    start = time.time()

    results_generator = await manager_generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    per_token_latency = []
    per_token_latency_breakdown_dict = init_per_token_latency_breakdown_dict()
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await manager_abort(request_id)
            return Response(status_code=499)
        now = time.time()
        per_token_latency.append([now, (now - start)*1000])
        if hasattr(request_output, 'request_timestamps'):
            request_output.request_timestamps.api_server_generate_benchmark_timestamp_end = now
            record_per_token_latency_breakdown(per_token_latency_breakdown_dict, request_output.request_timestamps)
        start = now
        final_output = request_output

    global num_finished_requests
    if log_requests:
        logger.info("Finished request {}.".format(request_id))
        num_finished_requests += 1
        logger.info("num_finished_requests {}.".format(num_finished_requests))

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
    ready_status = await engine_manager.is_ready.remote()
    return ready_status

class LlumnixArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.cur_namespace = "llumnix"
        super().__init__(*args, **kwargs)

    def set_namespace(self, namespace: str):
        self.cur_namespace = namespace

    def add_argument(self, *args, **kwargs):
        if self.cur_namespace == 'llumnix' and "--help" not in args:
            assert 'default' not in kwargs or kwargs['default'] is None, \
                f"Do not set the default value for '{args[0]}' in CLI, or set it to None. " \
                f"The default value will be retrieved from config/default.py in get_llumnix_config."
        super().add_argument(*args, **kwargs)

if __name__ == "__main__":
    parser: LlumnixArgumentParser = LlumnixArgumentParser()

    parser.set_namespace("llumnix")
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)
    parser.add_argument('--disable-log-requests-server', action='store_true', help='disable logging requests in server')
    parser.add_argument("--ray-cluster-port", type=int)
    parser.add_argument('--launch-ray-cluster', action='store_true', help='if launch ray cluster in api server')
    parser.add_argument("--queue-type", type=str, choices=['rayqueue', 'zmq'], help='queue type for request output queue')
    parser.add_argument("--request-output-queue-port", type=int, help='port for zmq')
    # TODO(KuilongCui): Fix default.py cannot take effects to cli arguments of action type.
    parser.add_argument("--log-request-timestamps", action='store_true', help='if log request timestamps')
    parser.add_argument("--config-file", help="path to config file")
    parser = EngineManagerArgs.add_cli_args(parser)

    parser.set_namespace("vllm")
    parser = AsyncEngineArgs.add_cli_args(parser)

    cli_args = parser.parse_args()
    cfg: LlumnixConfig = get_llumnix_config(cli_args.config_file, cli_args)
    engine_manager_args = EngineManagerArgs.from_llumnix_config(cfg)
    EngineManagerArgs.check_args(engine_manager_args, parser)
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    check_engine_args(engine_args, engine_manager_args)

    logger.info("engine_args: {}".format(engine_args))

    if cfg.RAY.LAUNCH_RAY_CLUSTER:
        # Launch the ray cluster for multi-node serving.
        launch_ray_cluster(cfg.RAY.RAY_CLUSTER_PORT)

    # Connect to a ray cluster.
    connect_to_ray_cluster(port=cfg.RAY.RAY_CLUSTER_PORT)

    # if gpu is not available, it means that this node is head pod without any llumnix components
    if is_gpu_available():
        # Launch the Llumnix componets on current node.
        server_id = random_uuid()
        ip = get_ip_address()
        node_id = ray.get_runtime_context().get_node_id()
        engine_manager, instance_ids, llumlets = \
            init_llumnix_components(engine_manager_args, engine_args, node_id, cfg.SERVER.QUEUE_TYPE)
        request_output_queue = get_output_queue_server(ip, cfg.SERVER.REQUEST_OUTPUT_QUEUE_PORT, cfg.SERVER.QUEUE_TYPE)
        server_info = ServerInfo(server_id, cfg.SERVER.QUEUE_TYPE, request_output_queue, ip,
                                 cfg.SERVER.REQUEST_OUTPUT_QUEUE_PORT)

        for idx, ins_id in enumerate(instance_ids):
            instances[ins_id] = llumlets[idx]
            instance_num_requests[ins_id] = 0
        log_requests = not cfg.SERVER.DISABLE_LOG_REQUESTS_SERVER
        log_request_timestamps = cfg.SERVER.LOG_REQUEST_TIMESTAMPS
        # Start the api server after all the components of llumnix are ready.
        logger.info("Start Api Server on '{}:{}'".format(cfg.SERVER.HOST, cfg.SERVER.PORT))
        logger.info("log_requests: {}, log_request_timestamps: {}".format(log_requests, log_request_timestamps))
        uvicorn.run(app,
                    host=cfg.SERVER.HOST,
                    port=cfg.SERVER.PORT,
                    log_level="debug",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=cfg.SERVER.SSL_KEYFILE,
                    ssl_certfile=cfg.SERVER.SSL_CERTFILE)
