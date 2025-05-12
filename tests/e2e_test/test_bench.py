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
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import json
import os

import pytest
import torch
import numpy as np

from llumnix.utils import get_ip_address, try_convert_to_local_path

# pylint: disable=unused-import
from tests import conftest
from tests.conftest import ray_env
from tests.e2e_test.utils import (generate_vllm_launch_command, generate_bench_command, to_markdown_table,
                    wait_for_llumnix_service_ready, shutdown_llumnix_service, wait_port_free,
                    generate_vllm_serve_command, generate_bladellm_launch_command, check_log_exception,
                    generate_bladellm_serve_command)

BENCH_TEST_TIMEOUT_MINS = 60
# Used to caculate port to avoid port conficts between tests.
test_times = 0


def parse_log_file(title: str):
    json_files = [f for f in os.listdir('.') if f.endswith('_latency_info.json')]

    def get_markdown_data(key: str, head_name: str):
        try:
            latencies = []
            for json_file in json_files:
                with open(json_file, 'r', encoding="utf-8") as file:
                    data = json.load(file)[-1]
                    latencies.extend(data.get(key, []))
            latencies_array = np.array(latencies)
        # pylint: disable=broad-except
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            return [head_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]

        p25 = np.percentile(latencies_array, 25)
        p50 = np.percentile(latencies_array, 50)
        p75 = np.percentile(latencies_array, 75)
        p95 = np.percentile(latencies_array, 95)
        p99 = np.percentile(latencies_array, 99)
        mean = np.mean(latencies_array)

        data = [head_name, f"{p25:.2f}", f"{p50:.2f}", f"{p75:.2f}", f"{p95:.2f}", f"{p99:.2f}", f"{mean:.2f}"]
        return data

    data = [["latency(ms)", "p25", "p50", "p75", "p95", "p99", "mean"]]
    data.append(get_markdown_data("decode_token_latencies", "decode"))
    data.append(get_markdown_data('prefill_token_latencies', 'prefill'))

    return (
        title
        + "\n"
        + to_markdown_table(data)
        + "\n"
        + "-------------------------------------\n"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for simple benchmark")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen2.5-7B')])
@pytest.mark.parametrize("request_output_queue_type", ["rayqueue", "zmq"])
@pytest.mark.parametrize("enable_pd_disagg", [False, True])
@pytest.mark.parametrize("engine", ["engine_vLLM", "engine_BladeLLM"])
async def test_simple_benchmark(request, ray_env, shutdown_llumnix_service, check_log_exception,
                                enable_pd_disagg, model, request_output_queue_type, engine):
    engine = engine.split("_")[1]

    num_prompts = 500

    if "vLLM" in engine and enable_pd_disagg:
        conftest.SKIP_REASON = "Do not test the vLLM pd-disagg mode; only consider its correctness for now."

    if conftest.SKIP_REASON is not None and len(conftest.SKIP_REASON) > 0:
        pytest.skip(conftest.SKIP_REASON)

    global test_times

    ip = get_ip_address()
    base_port = 20000 + test_times * 100
    wait_port_free(base_port)
    if "BladeLLM" in engine:
        base_port += 5000

    ip_ports = []
    device_count = min(4, torch.cuda.device_count())
    num_instances = device_count

    if "vLLM" in engine:
        generate_serve_command = generate_vllm_serve_command
    elif "BladeLLM" in engine:
        generate_serve_command = generate_bladellm_serve_command
    else:
        raise ValueError(f"Unknown engine: {engine}")

    for i in range(device_count):
        port = base_port + i
        ip_port = f"{ip}:{port}"
        ip_ports.append(ip_port)

    serve_command = generate_serve_command(result_filename=str(base_port)+".out",
                                            ip=ip,
                                            port=base_port,
                                            model=model,
                                            request_output_queue_type=request_output_queue_type,
                                            max_instances=num_instances)
    subprocess.run(serve_command, shell=True, check=True)
    wait_for_llumnix_service_ready(ip_ports)

    def run_bench_command(command):
        # pylint: disable=consider-using-with
        process = subprocess.Popen(command, shell=True)
        return process

    tasks = []
    for ip_port in ip_ports:
        port = ip_port.split(":")[1]
        bench_command = generate_bench_command(
            backend=engine,
            ip_ports=f"{ip}:{port}",
            model=model,
            num_prompts=num_prompts,
            dataset_type="sharegpt",
            dataset_path="/mnt/dataset/sharegpt_gpt4/sharegpt_gpt4.jsonl",
            qps=8,
            results_filename=f"{port}.out"
        )
        tasks.append(bench_command)

    with ThreadPoolExecutor() as executor:
        future_to_command = {executor.submit(run_bench_command, command): command for command in tasks}

        for future in as_completed(future_to_command):
            process = future.result()
            process.wait()
            assert process.returncode == 0, "bench_test failed with return code {}.".format(process.returncode)

    await asyncio.sleep(5)

    with open("performance.txt", "a", encoding="utf-8") as f:
        f.write(parse_log_file(title=request.node.name))

    await asyncio.sleep(3)

    test_times += 1
