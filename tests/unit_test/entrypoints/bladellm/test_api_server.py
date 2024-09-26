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

import subprocess
import sys
import time
import asyncio
import json
from multiprocessing import Pool
from pathlib import Path
import pytest
import aiohttp
import requests
from websockets.sync.client import connect

from llumnix.queue.queue_type import QueueType

from blade_llm.protocol import (
    GenerateRequest,
    GenerateStreamResponse,
    SamplingParams,
    StoppingCriteria,
)


# pylint: disable=unused-import
from tests.conftest import setup_ray_env

async def _query_server(prompt: str, max_tokens: int = 5, interface: str = '') -> dict:
    url = "ws://127.0.0.1:8000/{}".format(interface)
    headers = {
        # "Authorization": "<You may need this header for EAS."
    }
    with connect(url, additional_headers=headers) as websocket:
        req = GenerateRequest(
            prompt=prompt,
            sampling_params=SamplingParams(
                temperature=0,
            ),
            stopping_criterial=StoppingCriteria(max_new_tokens=max_tokens, ignore_eos=True),)
        websocket.send(req.model_dump_json())
        texts = []
        idx = 0
        while True:
            await asyncio.sleep(0)
            msg = websocket.recv()
            resp = GenerateStreamResponse(**json.loads(msg))
            print(resp)
            texts.extend([t.text for t in resp.tokens])
            idx += 1
            if resp.is_finished:
                break
    return texts

def _query_http_server(prompt: str = None, timeout: int = 30):
    response = requests.get("http://localhost:8000/")
    response.raise_for_status()
    if response.status_code == 200:
        return True

def _query_server_long(prompt: str) -> dict:
    return _query_server(prompt, max_tokens=500)

def _query_server_generate(prompt: str) -> dict:
    return _query_server(prompt, interface='generate')

def _query_server_generate_stream(prompt: str) -> dict:
    return _query_server(prompt, interface='generate_stream')

@pytest.fixture(params=["zmq"])
def api_server(request):
    output_queue_type = QueueType(request.param)
    script_path = Path(__file__).parent.joinpath(
        "api_server_manager.py").absolute()
    commands = [
        sys.executable,
        "-u",
        str(script_path),
        "--host", "127.0.0.1",
        "--output-queue-type", output_queue_type,
    ]
    print("commands",commands)
    server_proc = subprocess.Popen(commands)
    yield
    server_proc.terminate()
    # Waiting for api server subprocess to terminate.
    time.sleep(1.0)

@pytest.mark.parametrize("interface", ['generate_stream'])
@pytest.mark.asyncio
async def test_api_server(setup_ray_env, api_server, interface: str):
    """
    Run the API server and test it.

    We run both the server and requests in separate processes.

    We test that the server can handle incoming requests, including
    multiple requests at the same time, and that it can handle requests
    being cancelled without crashing.
    """
    # Wait until the server is ready
    timeout = 30
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if _query_http_server(timeout=timeout):
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("API server didn't start within the timeout period")

    if interface == 'generate':
        _query_server = _query_server_generate
    elif interface == 'generate_stream':
        _query_server = _query_server_generate_stream
    

    # Actual tests start here
    prompts = ["warm up"] * 1
    # Try with 1 prompt
    for prompt in prompts:
        result = await _query_server(prompt)
        assert result
    
    # Try with 100 prompts
    prompts = ["test prompt"] * 100
    results = await asyncio.gather(*[_query_server(prompt) for prompt in prompts])
    for result in results:
        assert result
