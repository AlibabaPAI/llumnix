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

from llumnix.entrypoints.utils import get_ip_address

# pylint: disable=unused-import
from tests.conftest import ray_env
from .utils import (generate_launch_command, generate_bench_command, to_markdown_table,
                    wait_for_llumnix_service_ready, shutdown_llumnix_service,
                    generate_serve_command)

BENCH_TEST_TIMEOUT_MINS = 30


def parse_log_file(mark_parametrizes: dict):
    json_files = [f for f in os.listdir('.') if f.endswith('_latency_info.json')]

    def get_markdown_data(key: str, head_name: str):
        latencies = []

        for json_file in json_files:
            with open(json_file, 'r', encoding="utf-8") as file:
                data = json.load(file)[-1]
                latencies.append(data.get(key, []))

        latencies_array = np.array(latencies)

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
        json.dumps(mark_parametrizes)
        + "\n"
        + to_markdown_table(data)
        + "\n"
        + "-------------------------------------\n"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for simple benchmark")
@pytest.mark.parametrize("model", ['/mnt/model/Qwen-7B'])
@pytest.mark.parametrize("launch_mode", ['global', 'local'])
@pytest.mark.parametrize("enable_pd_disagg", [False, True])
@pytest.mark.parametrize("enable_simulator", [False, True])
@pytest.mark.parametrize("output_queue_type", ["rayqueue","zmq"])
async def test_simple_benchmark(
    ray_env,
    shutdown_llumnix_service,
    model,
    launch_mode,
    enable_pd_disagg,
    enable_simulator,
    output_queue_type,
):
    if enable_simulator and enable_pd_disagg:
        pytest.skip("When enabling simulator, prefill-decode disaggregation is not tested.")

    if output_queue_type == "zmq" and not (
        launch_mode == "global" and not enable_simulator and not enable_pd_disagg
    ):
        pytest.skip(
            "Only test zmq queue type when simulator is enabled and prefill-decode disaggregation is disabled."
        )

    if launch_mode == 'local':
        num_prompts = 500 if not enable_pd_disagg else 50
    else:
        num_prompts = 50 if not enable_pd_disagg else 50

    if enable_simulator:
        num_prompts = 50

    ip = get_ip_address()
    base_port = 37037
    ip_ports = []
    if launch_mode == 'local':
        device_count = torch.cuda.device_count()
        if enable_pd_disagg:
            for i in range(device_count//2):
                port = base_port+i
                ip_port = f"{ip}:{port}"
                ip_ports.append(ip_port)
                launch_command = generate_launch_command(result_filename=str(base_port+i)+".out",
                                                         launch_ray_cluster=False,
                                                         ip=ip,
                                                         port=port,
                                                         model=model,
                                                         enable_pd_disagg=enable_pd_disagg,
                                                         instance_type="prefill",
                                                         output_queue_type=output_queue_type)
                subprocess.run(launch_command, shell=True, check=True)
            for i in range(device_count//2):
                port = base_port+i+device_count//2
                ip_port = f"{ip}:{port}"
                ip_ports.append(ip_port)
                launch_command = generate_launch_command(result_filename=str(base_port+i)+".out",
                                                         launch_ray_cluster=False,
                                                         ip=ip,
                                                         port=port,
                                                         model=model,
                                                         enable_pd_disagg=enable_pd_disagg,
                                                         instance_type="decode",
                                                         output_queue_type=output_queue_type)
                subprocess.run(launch_command, shell=True, check=True)
        else:
            for i in range(device_count):
                port = base_port+i
                ip_port = f"{ip}:{port}"
                ip_ports.append(ip_port)
                launch_command = generate_launch_command(result_filename=str(base_port+i)+".out",
                                                         launch_ray_cluster=False,
                                                         ip=ip,
                                                         port=port,
                                                         model=model,
                                                         enable_simulator=enable_simulator,
                                                         output_queue_type=output_queue_type)
                subprocess.run(launch_command, shell=True, check=True)
    else: # global
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            port = base_port+i
            ip_port = f"{ip}:{port}"
            ip_ports.append(ip_port)
        serve_command = generate_serve_command(result_filename=str(base_port)+".out",
                                               ip=ip,
                                               port=base_port,
                                               model=model,
                                               enable_pd_disagg=enable_pd_disagg,
                                               enable_simulator=enable_simulator,
                                               output_queue_type=output_queue_type)
        subprocess.run(serve_command, shell=True, check=True)
    wait_for_llumnix_service_ready(ip_ports)

    def run_bench_command(command):
        # pylint: disable=consider-using-with
        process = subprocess.Popen(command, shell=True)
        return process

    tasks = []
    for i in range(device_count):
        bench_command = generate_bench_command(
            ip_ports=f"{ip}:{base_port + i}",
            model=model,
            num_prompts=num_prompts,
            dataset_type="sharegpt",
            dataset_path="/mnt/dataset/sharegpt_gpt4/sharegpt_gpt4.jsonl",
            qps=5,
            results_filename=f"{base_port + i}.out"
        )
        tasks.append(bench_command)

    with ThreadPoolExecutor() as executor:
        future_to_command = {executor.submit(run_bench_command, command): command for command in tasks}

        for future in as_completed(future_to_command):
            try:
                process = future.result()
                process.wait(timeout=60*BENCH_TEST_TIMEOUT_MINS)

                assert process.returncode == 0, "bench_test failed with return code {}.".format(process.returncode)
            # pylint: disable=broad-except
            except subprocess.TimeoutExpired:
                process.kill()
                assert False, "bench_test timed out after {} minutes.".format(BENCH_TEST_TIMEOUT_MINS)

    if num_prompts >= 500:
        with open("performance.txt", "a", encoding="utf-8") as f:
            mark_parametrizes = {
                "model": model.split("/")[-1],
                "num_prompts": num_prompts,
                "launch_mode": launch_mode,
                "enable_pd_disagg" :enable_pd_disagg,
                "enable_simulator": enable_simulator,
                "output_queue_type": output_queue_type
            }
            f.write(parse_log_file(mark_parametrizes))

    await asyncio.sleep(3)
