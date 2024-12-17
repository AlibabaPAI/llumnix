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
from multiprocessing import Pool
from pathlib import Path
import pytest
import requests

from llumnix.queue.queue_type import QueueType

# pylint: disable=unused-import
from tests.conftest import ray_env


def _query_server(prompt: str, max_tokens: int = 5, interface: str = 'generate') -> dict:
    response = requests.post("http://localhost:8000/{}".format(interface),
                             json={
                                 "prompt": prompt,
                                 "max_tokens": max_tokens,
                                 "temperature": 0,
                                 "ignore_eos": True
                             })
    response.raise_for_status()
    return response.json()

def _query_server_long(prompt: str) -> dict:
    return _query_server(prompt, max_tokens=500)

def _query_server_generate(prompt: str) -> dict:
    return _query_server(prompt, interface='generate')

def _query_server_generate_benchmark(prompt: str) -> dict:
    return _query_server(prompt, interface='generate_benchmark')

@pytest.fixture(params=[("zmq", "api_server"), ("rayqueue", "api_server"), ("zmq", "api_server_actor"), ("rayqueue", "api_server_actor")])
def api_server(request):
    request_output_queue_type = QueueType(request.param[0])
    if request.param[1] == "api_server":
        script_path = Path(__file__).parent.joinpath(
            "api_server.py").absolute()
    else:
        script_path = Path(__file__).parent.joinpath(
            "api_server_actor.py").absolute()
    commands = [
        sys.executable,
        "-u",
        str(script_path),
        "--host", "127.0.0.1",
        "--request-output-queue-type", request_output_queue_type,
    ]
    # pylint: disable=consider-using-with
    uvicorn_process = subprocess.Popen(commands)
    yield
    uvicorn_process.terminate()
    # Waiting for api server subprocess to terminate.
    time.sleep(1)

@pytest.mark.parametrize("interface", ['generate', 'generate_benchmark'])
def test_api_server(ray_env, api_server, interface: str):
    """
    Run the API server and test it.

    We run both the server and requests in separate processes.

    We test that the server can handle incoming requests, including
    multiple requests at the same time, and that it can handle requests
    being cancelled without crashing.
    """
    if interface == 'generate':
        _query_server = _query_server_generate
    elif interface == 'generate_benchmark':
        _query_server = _query_server_generate_benchmark

    with Pool(32) as pool:
        # Wait until the server is ready
        prompts = ["warm up"] * 1
        result = None
        while not result:
            try:
                for r in pool.map(_query_server, prompts):
                    result = r
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        # Actual tests start here
        # Try with 1 prompt
        for result in pool.map(_query_server, prompts):
            assert result

        num_aborted_requests = requests.get(
            "http://localhost:8000/stats").json()["num_aborted_requests"]
        assert num_aborted_requests == 0

        # Try with 100 prompts
        prompts = ["test prompt"] * 100
        for result in pool.map(_query_server, prompts):
            assert result

    with Pool(32) as pool:
        # Cancel requests
        prompts = ["canceled requests"] * 100
        pool.map_async(_query_server_long, prompts)
        time.sleep(0.01)
        pool.terminate()
        pool.join()

        # check cancellation stats
        # give it some times to update the stats
        time.sleep(1)

        num_aborted_requests = requests.get(
            "http://localhost:8000/stats").json()["num_aborted_requests"]
        assert num_aborted_requests > 0

    # check that server still runs after cancellations
    with Pool(32) as pool:
        # Try with 100 prompts
        prompts = ["test prompt after canceled"] * 100
        for result in pool.map(_query_server, prompts):
            assert result
