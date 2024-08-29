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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import ray

from vllm.sampling_params import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncStream

from llumnix.arg_utils import EngineManagerArgs
from llumnix.server_info import ServerInfo
from llumnix.entrypoints.llumnix_utils import (launch_ray_cluster, connect_to_ray_cluster,
                                                is_gpu_available, init_llumnix_components)
from llumnix.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.backends.vllm.utils import check_engine_args

logger = init_logger(__name__)
engine_manager = None
instances = {}
instance_num_requests: Dict[str, int] = {}
# request_output_queue could be None if initialzed in lifespan.
request_output_queue = None
server_id = None
TIMEOUT_KEEP_ALIVE = 5  # seconds.
request_streams: Dict[str, AsyncStream] = {}
log_requests = None
num_finished_requests = 0
WAIT_MANAGER_INTERVAL = 5
manager_available = True


async def _background_process_outputs():
    while True:
        qsize = await request_output_queue.actor.qsize.remote()
        request_outputs = await request_output_queue.actor.get_nowait_batch.remote(qsize)
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
    asyncio.create_task(_background_process_outputs())
    yield

app = FastAPI(lifespan=lifespan)

async def manager_generate(prompt, sampling_params, request_id) -> AsyncStream:
    if sampling_params.n > 1 or sampling_params.use_beam_search:
        raise ValueError("Unsupported feature: multiple sequence decoding")
    results_generator = AsyncStream(request_id)
    request_streams[request_id] = results_generator
    # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
    server_info = ServerInfo(server_id, request_output_queue)
    # If manager is unavailable, request will be directly added to the llumlet held by api server.
    global manager_available
    try:
        # await to catch exception
        await engine_manager.generate.remote(request_id, server_info, prompt, sampling_params)
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
                print("Manager is unavailable, directly pass request {} to instance {}".format(request_id, instance_id))
            else:
                print("Manager is unavailable, but there is no instance behind this api server, "
                      "sleep {}s, waiting for manager restarts".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                return await asyncio.create_task(manager_generate(prompt, sampling_params, request_id))
        except (ray.exceptions.RayActorError, KeyError):
            if instance_id in instances:
                print("[manager_generate] instance {} is dead".format(instance_id))
                del instances[instance_id]
                del instance_num_requests[instance_id]
            return await asyncio.create_task(manager_generate(prompt, sampling_params, request_id))
    return results_generator

async def manager_abort(request_id: str) -> None:
    try:
        print("abort request: {}.".format(request_id))
        await engine_manager.abort.remote(request_id)
    except ray.exceptions.RayActorError:
        print("Manager is unavailable")


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


@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    _ = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = await manager_generate(prompt, sampling_params, request_id)

    per_token_latency = []
    start = time.time()

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await manager_abort(request_id)
            return Response(status_code=499)
        now_time = time.time()
        per_token_latency.append([now_time, int((now_time - start)*1000)])
        start = now_time
        final_output = request_output

    global num_finished_requests
    if log_requests:
        # TODO(s5u13b): Use logger.
        print(f"Finished request {request_id}.")
        num_finished_requests += 1
        print(f"num_finished_requests {num_finished_requests}.")

    generation = final_output.outputs[0].text
    num_output_tokens = len(final_output.outputs[0].token_ids)
    num_input_tokens = len(final_output.prompt_token_ids)
    expected_resp_len = request_dict['max_tokens']
    if not max(expected_resp_len, 1) == max(num_output_tokens, 1):
        "request_id={}, expected_resp_len={}, num_output_tokens={}, num_input_tokens={}".format(
            request_id, expected_resp_len, num_output_tokens, num_input_tokens)
    ret = {
        'generated_text': generation,
        'num_output_tokens_cf': num_output_tokens,
        'per_token_latency': per_token_latency,
        'error': None,
        'request_id': request_id
    }
    return JSONResponse(ret)


@app.get("/is_ready")
async def is_ready():
    ready_status = await engine_manager.is_ready.remote()
    return ready_status

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument('--disable-log-requests-server',
                        action='store_true',
                        help='disable logging requests in server')
    parser.add_argument("--ray-cluster-port", type=int, default=30050)
    parser.add_argument('--launch-ray-cluster',
                        action='store_true',
                        help='if launch ray cluster in api server')

    parser = EngineManagerArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_manager_args = EngineManagerArgs.from_cli_args(args)
    engine_args = AsyncEngineArgs.from_cli_args(args)

    check_engine_args(engine_args, engine_manager_args)

    print("engine_args: {}".format(engine_args))

    if args.launch_ray_cluster:
        # Launch the ray cluster for multi-node serving.
        launch_ray_cluster(args.ray_cluster_port)

    # Connect to a ray cluster.
    connect_to_ray_cluster(port=args.ray_cluster_port)

    # if gpu is not available, it means that this node is head pod without any llumnix components
    if is_gpu_available():
        # Launch the Llumnix componets on current node.
        server_id = random_uuid()
        node_id = ray.get_runtime_context().get_node_id()
        engine_manager, instance_ids, llumlets, request_output_queue = \
            init_llumnix_components(engine_manager_args, engine_args, node_id)

        for idx, ins_id in enumerate(instance_ids):
            instances[ins_id] = llumlets[idx]
            instance_num_requests[ins_id] = 0
        log_requests = not args.disable_log_requests_server
        # Start the api server after all the components of llumnix are ready.
        print(f"Start Api Server on '{args.host}:{args.port}'")
        uvicorn.run(app,
                    host=args.host,
                    port=args.port,
                    log_level="debug",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=args.ssl_keyfile,
                    ssl_certfile=args.ssl_certfile)
