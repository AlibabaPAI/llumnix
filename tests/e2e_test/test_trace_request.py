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


import random
import subprocess
import asyncio
import json
import requests
import pytest

from llumnix.utils import get_ip_address, wait_port_free

# pylint: disable=unused-import
from tests.e2e_test.utils import (
    generate_vllm_serve_command,
    wait_for_llumnix_service_ready,
    generate_bladellm_serve_command,
    shutdown_llumnix_service,
    check_log_exception,
)
from tests.utils import try_convert_to_local_path


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

requests_dict = {
    "/v1/chat/completions": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello world."},
        ],
        "stream": "true",
        "max_tokens": 10,
    },
    "/v1/completions": {"prompt": "Hello world.", "stream": "true", "max_tokens": 10},
    "/generate": {"prompt": "Hello world.", "stream": "true", "max_tokens": 10},
}


def get_requests(api, is_stream):
    req_body = requests_dict[api]
    req_body["stream"] = is_stream
    return req_body


engine_apis = {
    "vLLM": ["/generate"],
    "BladeLLM": ["/v1/completions", "/v1/chat/completions"],
}

test_times = 0

@pytest.mark.asyncio
@pytest.mark.parametrize("model", [try_convert_to_local_path("Qwen/Qwen2.5-7B")])
@pytest.mark.parametrize("engine", ["engine_vLLM", "engine_BladeLLM"])
@pytest.mark.parametrize("request_output_queue_type", ["rayqueue", "zmq"])
@pytest.mark.parametrize("enable_request_trace", [True, False])
@pytest.mark.parametrize("is_stream", [True, False])
async def test_request_trace(
    ray_env,
    shutdown_llumnix_service,
    check_log_exception,
    model,
    engine,
    request_output_queue_type,
    enable_request_trace,
    is_stream,
):
    engine = engine.split("_")[1]
    global test_times

    ip = get_ip_address()
    base_port = 30000 + random.randint(0, 46) + test_times * 100
    if "BladeLLM" in engine:
        base_port += 2500
    tensor_parallel_size = 1
    instance_count = 1
    endpoint = f"http://{ip}:{base_port}"

    if engine == "vLLM":
        generate_serve_command_func = generate_vllm_serve_command
    else:
        generate_serve_command_func = generate_bladellm_serve_command

    ip_ports = []

    launch_commands = []

    for i in range(instance_count):
        wait_port_free(base_port + i, force=True)
        ip_ports.append(f"{ip}:{base_port + i}")
    launch_commands.append(
        generate_serve_command_func(
            result_filename=str(base_port) + ".out",
            ip=ip,
            port=base_port,
            model=model,
            enforce_eager=True,
            enable_pd_disagg=False,
            enable_simulator=False,
            tensor_parallel_size=tensor_parallel_size,
            enable_migration=False,
            max_instances=instance_count,
        )
    )
    for launch_command in launch_commands:
        subprocess.run(launch_command, shell=True, check=True)

    await asyncio.sleep(3)

    wait_for_llumnix_service_ready(ip_ports)

    await asyncio.sleep(3)

    for api in engine_apis[engine]:
        url = f"{endpoint}{api}"

        request = get_requests(api, is_stream)
        headers = {"X-Llumnix-Trace": "true"} if enable_request_trace else {}
        with requests.post(
            url, json=request, stream=is_stream, headers=headers, timeout=60
        ) as resp:
            if not is_stream:
                output = resp.json()
                if enable_request_trace:
                    assert (
                        "llumnix_trace_info" in output and output["llumnix_trace_info"]
                    ), "llumnix debug info is missing"
                if not enable_request_trace:
                    assert (
                        "llumnix_trace_info" not in output
                    ), "reqeust trace mode is not enabled but return llumnix trace info"
            if is_stream:
                has_trace_info = False
                for chunk_bytes in resp.iter_content(
                    chunk_size=None, decode_unicode=False
                ):
                    chunk = (
                        chunk_bytes.decode("utf-8")
                        .removeprefix("data:")
                        .replace("\x00", "")
                        .strip()
                    )
                    try:
                        if chunk in ("", "[DONE]"):
                            continue
                        chunk_json = json.loads(str(chunk.strip()))
                        if (
                            "llumnix_trace_info" in chunk_json
                            and chunk_json["llumnix_trace_info"]
                        ):
                            has_trace_info = True
                    except json.JSONDecodeError:
                        print(f"json decode error, {chunk.strip()}")
                if enable_request_trace:
                    assert has_trace_info, "llumnix trace info is missing"
                if not enable_request_trace:
                    assert (
                        not has_trace_info
                    ), "reqeust trace mode is not enabled but return llumnix trace info"
    await asyncio.sleep(1)

    test_times += 1
