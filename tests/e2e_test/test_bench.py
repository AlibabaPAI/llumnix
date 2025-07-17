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

from functools import partial
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import json
import os
import random

import pytest
import torch
import numpy as np

from llumnix.utils import get_ip_address, wait_port_free

# pylint: disable=unused-import
from tests import conftest
from tests.conftest import ray_env
from tests.e2e_test.utils import (generate_bench_command, to_markdown_table,
                    wait_for_llumnix_service_ready, wait_for_llumnix_service_ready_vllm_v1,
                    shutdown_llumnix_service, generate_special_test_config, check_log_exception,
                    generate_vllm_serve_command, generate_vllm_v1_serve_command,
                    generate_bladellm_serve_command)
from tests.utils import try_convert_to_local_path


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


config_schema = "engine, enable_pd_disagg, request_output_queue_type, enable_engine_semi_pd_disagg, enable_adaptive_pd"

generate_special_bench_test_config = partial(generate_special_test_config, schema=config_schema)

def generate_bench_test_config():
    vllm_base_config = ["engine_vLLM", False, "zmq", False, False]

    vllm_config = [
        vllm_base_config,

        # rayqueue
        generate_special_bench_test_config([("request_output_queue_type", "rayqueue")], vllm_base_config),
    ]

    vllm_v1_base_config = ["engine_vLLM_v1", False, "zmq", False, False]

    vllm_v1_config = [
        vllm_v1_base_config,

        # rayqueue
        generate_special_bench_test_config([("request_output_queue_type", "rayqueue")], vllm_v1_base_config),
    ]


    bladellm_base_config = ["engine_BladeLLM", False, "zmq", False, False]

    bladellm_config = [
        bladellm_base_config,

        # pd
        generate_special_bench_test_config([("enable_pd_disagg", True)], bladellm_base_config),

        # rayqueue
        generate_special_bench_test_config([("request_output_queue_type", "rayqueue")], bladellm_base_config),

        # semi-pd
        generate_special_bench_test_config([("enable_engine_semi_pd_disagg", True)], bladellm_base_config),

        # adaptive-pd
        generate_special_bench_test_config([("enable_engine_semi_pd_disagg", True), ("enable_adaptive_pd", True)],
                                           bladellm_base_config),
    ]

    return vllm_config + vllm_v1_config + bladellm_config

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for simple benchmark")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen2.5-7B')])
@pytest.mark.parametrize(config_schema, generate_bench_test_config())
async def test_simple_benchmark(request, ray_env, shutdown_llumnix_service, check_log_exception, model,
                                engine, enable_pd_disagg, request_output_queue_type, enable_engine_semi_pd_disagg,
                                enable_adaptive_pd):
    engine = "_".join(engine.split("_")[1:])

    global test_times

    qps = 5 if not (enable_pd_disagg or enable_engine_semi_pd_disagg) else 0.5
    num_prompts = 300 if not (enable_pd_disagg or enable_engine_semi_pd_disagg) else 50
    ip = get_ip_address()
    base_port = 20000 + random.randint(0, 96) + test_times * 100
    if "BladeLLM" in engine:
        base_port += 2500

    ip_ports = []
    device_count = min(4, torch.cuda.device_count())
    num_instances = device_count
    pd_ratio = "1:1" if not enable_adaptive_pd else "1:3"

    if engine == "vLLM":
        generate_serve_command = generate_vllm_serve_command
    elif engine == "vLLM_v1":
        generate_serve_command = generate_vllm_v1_serve_command
    elif engine == "BladeLLM":
        generate_serve_command = generate_bladellm_serve_command
    else:
        raise ValueError(f"Unknown engine: {engine}")

    for i in range(device_count):
        port = base_port + i
        wait_port_free(port, force=True)
        ip_port = f"{ip}:{port}"
        ip_ports.append(ip_port)

    serve_command = generate_serve_command(result_filename=str(base_port)+".out",
                                            ip=ip,
                                            port=base_port,
                                            model=model,
                                            pd_ratio=pd_ratio,
                                            enable_migration="vLLM_v1" not in engine,
                                            dispatch_policy="load",
                                            enable_pd_disagg=enable_pd_disagg,
                                            enable_engine_semi_pd_disagg=enable_engine_semi_pd_disagg,
                                            enable_adaptive_pd=enable_adaptive_pd,
                                            request_output_queue_type=request_output_queue_type,
                                            max_instances=num_instances)
    subprocess.run(serve_command, shell=True, check=True)
    # TODO(zhaozhiyu): remove this special judge in the future
    if engine =="vLLM_v1":
        # special wait_for_llumnix_service_ready for vllm v1
        wait_for_llumnix_service_ready_vllm_v1(ip_ports)
    else:
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
            qps=qps,
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

    if engine != "vLLM_v1":
        with open("performance.txt", "a", encoding="utf-8") as f:
            f.write(parse_log_file(title=request.node.name))

    await asyncio.sleep(3)

    test_times += 1
