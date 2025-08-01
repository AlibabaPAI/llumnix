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


import itertools
import random
import subprocess
import asyncio
import json
import numpy as np
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
    to_markdown_table
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
async def test_request_trace(
    request,
    ray_env,
    shutdown_llumnix_service,
    check_log_exception,
    model,
    engine,
    request_output_queue_type,
):
    engine = engine.split("_")[1]
    global test_times

    ip = get_ip_address()
    base_port = 15000 + random.randint(0, 46) + test_times * 100
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
            enable_routine_migration=False,
            enable_pre_stop_migration=False,
            max_units=instance_count,
        )
    )
    for launch_command in launch_commands:
        subprocess.run(launch_command, shell=True, check=True)

    await asyncio.sleep(3)

    wait_for_llumnix_service_ready(ip_ports)

    await asyncio.sleep(3)

    is_stream_list = [True, False]
    enable_request_trace_list = [True, False]
    for is_stream, enable_request_trace in itertools.product(is_stream_list, enable_request_trace_list):
        llumnix_trace_info = None
        for api in engine_apis[engine]:
            url = f"{endpoint}{api}"

            request_json = get_requests(api, is_stream)
            headers = {"X-Llumnix-Trace": "true"} if enable_request_trace else {}
            with requests.post(
                url, json=request_json, stream=is_stream, headers=headers, timeout=60
            ) as resp:
                if not is_stream:
                    output = resp.json()
                    if enable_request_trace:
                        assert (
                            "llumnix_trace_info" in output and output["llumnix_trace_info"]
                        ), "llumnix debug info is missing"
                        llumnix_trace_info = output["llumnix_trace_info"]
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
            if (
                enable_request_trace
                and api in ["/generate", "/v1/chat/completions"]
                and not is_stream
            ):
                with open("trace_info.txt", "a", encoding="utf-8") as f:
                    f.write(process_llumnix_trace_info(request.node.name, llumnix_trace_info))

    await asyncio.sleep(1)

    test_times += 1

def process_llumnix_trace_info(title, trace_info_list):
    """
    Process the llumnix trace info and return a formatted string.
    """
    latnecy_dict = {}
    for trace_info in trace_info_list:
        for key, value in trace_info['latencys'].items():
            if key not in latnecy_dict:
                latnecy_dict[key] = []
            latnecy_dict[key].append(value)

    data = [["latency(ms)", "mean", "p50", "p99", "min", "max"]]
    for key, value in latnecy_dict.items():
        latencies_array = np.array(value)
        p50 = np.percentile(latencies_array, 50)
        p99 = np.percentile(latencies_array, 99)
        mean = np.mean(latencies_array)
        max_value = np.max(latencies_array)
        min_value = np.min(latencies_array)
        data.append([key, f"{mean:.4f}", f"{p50:.4f}", f"{p99:.4f}", f"{min_value:.4f}", f"{max_value:.4f}"])

    return (
        title
        + "\n"
        + to_markdown_table(data)
        + "\n"
        + "-------------------------------------\n"
    )
    