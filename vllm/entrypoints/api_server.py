import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import random
import time

# from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.arg_utils import EngineManagerArgs
# from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine_manager import LLMEngineManager
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    session_id = request_dict.pop("session_id")
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    # results_generator = engine.generate(prompt, sampling_params, request_id)
    # results_generator = engine_manager.generate(session_id, prompt, sampling_params, request_id)
    # @@@: loop/callback
    # results_generator = engine_manager.generate_loop(session_id, prompt, sampling_params, request_id)
    results_generator = engine_manager.generate(session_id, prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        # await engine.abort(request_id)
        await engine_manager.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            # await engine.abort(request_id)
            await engine_manager.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs,"req_id":request_id}
    return JSONResponse(ret)


@app.post("/generate_v2")
async def generate_v2(request: Request):
    request_dict = await request.json()
    prompt = request_dict['inputs']
    sampling_config = request_dict['parameters']
    priority_type = request_dict['priority_type']

    best_of = random.randint(1,1)
    use_beam_search = best_of > 1
    output_len = sampling_config['reponse_len']
    session_id = sampling_config['session_id']

    request_dict = {
        "session_id": session_id,
        "prompt": prompt,
        "n": 1,
        "best_of": best_of,
        "use_beam_search": use_beam_search,
        "temperature": 0.0 if use_beam_search else 1.0,
        "top_k": -1 if use_beam_search else 1,
        "max_tokens": max(output_len, 1),
        "ignore_eos": True,
        "stream": False,
    }

    session_id = request_dict.pop("session_id")
    prompt = request_dict.pop("prompt")
    _ = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine_manager.generate(session_id, prompt, sampling_params, request_id, priority_type=priority_type)
    per_token_latency = []
    start = time.time()
    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine_manager.abort(request_id)
            return Response(status_code=499)
        now_time = time.time()
        per_token_latency.append([now_time, int((now_time - start)*1000)])
        start = now_time
        final_output = request_output

    # text_outputs = [output.text for output in final_output.outputs]
    generation = final_output.outputs[0].text
    num_output_tokens = len(final_output.outputs[0].token_ids)
    inference_time = final_output.total_inference_time

    expected_resp_len = sampling_config['reponse_len']
    # print(f'generate check_len: {num_output_tokens=} {expected_resp_len=}')
    assert max(expected_resp_len, 1) == max(num_output_tokens, 1), f"{expected_resp_len=} {num_output_tokens=}"
    ret = {
        'generated_text': generation,
        'num_output_tokens_cf': num_output_tokens,
        'inference_time': inference_time,
        'per_token_latency':per_token_latency,
        'error': None,
        'request_id': request_id
    }

    # return JSONResponse(ret)
    return ret


@app.get("/is_ready")
async def is_ready(request: Request):
    return await engine_manager.is_ready()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8003)
    # parser = AsyncEngineArgs.add_cli_args(parser)
    # args = parser.parse_args()

    # engine_args = AsyncEngineArgs.from_cli_args(args)
    # engine = AsyncLLMEngine.from_engine_args(engine_args)

    parser = EngineManagerArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = EngineManagerArgs.from_cli_args(args)
    engine_manager = LLMEngineManager.from_engine_args(engine_args)

    import asyncio
    loop = asyncio.new_event_loop()

    # @@@: loop/callback
    # engine_manager.start_backgroup_loop(loop)
    engine_manager.register_callback_to_instances()

    # uvicorn.run(app,
    #             host=args.host,
    #             port=args.port,
    #             log_level="debug",
    #             timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
    from uvicorn import Config, Server
    config = Config(app=app, loop=loop, host='localhost',
                    port=args.port, log_level="info")
    uvicorn_server = Server(config)

    loop.run_until_complete(uvicorn_server.serve())
