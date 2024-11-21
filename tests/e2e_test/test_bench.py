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

from .test_e2e import generate_launch_command, clear_ray_state
from .utils import to_markdown_table, backup_instance_log

def launch_llumnix_service(command):
    subprocess.run(command, shell=True, check=True)

def generate_bench_command(ip_ports: str, model: str, num_prompts: int, dataset_type: str, dataset_path: str,
                           qps: int, results_filename: str = "", query_distribution: str = "poisson",
                           coefficient_variation: float = 1.0, priority_ratio: float = 0.0):
    command = (
        f"python -u ./benchmark/benchmark_serving.py "
        f"--ip_ports {ip_ports} "
        f"--backend vLLM "
        f"--tokenizer {model} "
        f"--trust_remote_code "
        f"--log_filename bench_{ip_ports.split(':')[1]} "
        f"--random_prompt_count {num_prompts} "
        f"--dataset_type {dataset_type} "
        f"--dataset_path {dataset_path} "
        f"--qps {qps} "
        f"--distribution {query_distribution} "
        f"--coefficient_variation {coefficient_variation} "
        f"--priority_ratio {priority_ratio} "
        f"--log_latencies "
        f"--fail_on_response_failure "
        f"{'> bench_'+results_filename if len(results_filename)> 0 else ''}"
    )
    return command

def shutdown_llumnix_service():
    try:
        subprocess.run('pkill -f llumnix.entrypoints.vllm.api_server', shell=True, check=True)
    # pylint: disable=broad-except
    except Exception:
        pass

def parse_log_file():
    json_files = [f for f in os.listdir('.') if f.endswith('_latency_info.json')]

    def get_markdown_data(key: str, head_name: str):
        latencies = []

        for json_file in json_files:
            with open(json_file, 'r', encoding="utf-8") as file:
                data = json.load(file)[0]
                latencies.append(data.get(key, []))

        latencies_array = np.array(latencies)

        p25 = np.percentile(latencies_array, 25)
        p50 = np.percentile(latencies_array, 50)
        p75 = np.percentile(latencies_array, 75)
        p95 = np.percentile(latencies_array, 95)
        p99 = np.percentile(latencies_array, 99)
        mean = np.mean(latencies_array)

        data = [
            [head_name, "p25", "p50", "p75", "p95", "p99", "mean"],
            ["latency(ms)", f"{p25:.2f}", f"{p50:.2f}", f"{p75:.2f}", f"{p95:.2f}", f"{p99:.2f}", f"{mean:.2f}"]
        ]

        return data

    decode_data = get_markdown_data('decode_token_latencies', 'decode')
    prefill_data = get_markdown_data('prefill_token_latencies', 'prefill')

    return to_markdown_table(prefill_data) + "\n\n" + to_markdown_table(decode_data)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required for simple benchmark")
@pytest.mark.parametrize("model", ['/mnt/model/Qwen-7B'])
async def test_simple_benchmark(model):
    device_count = torch.cuda.device_count()
    base_port = 37037
    for i in range(device_count):
        launch_command = generate_launch_command(result_filename=str(base_port+i)+".out",
                                                 launch_ray_cluster=False, port=base_port+i, model=model)
        subprocess.run(launch_command, shell=True, check=True)

    await asyncio.sleep(30)

    def run_bench_command(command):
        # pylint: disable=consider-using-with
        process = subprocess.Popen(command, shell=True)
        return process

    tasks = []
    for i in range(device_count):
        bench_command = generate_bench_command(
            ip_ports=f"127.0.0.1:{base_port + i}",
            model=model,
            num_prompts=200,
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
                process.wait(timeout=60*30)

                if process.returncode != 0:
                    backup_instance_log()

                assert process.returncode == 0, "bench_test failed with return code {}.".format(process.returncode)
            # pylint: disable=broad-except
            except subprocess.TimeoutExpired:
                process.kill()
                backup_instance_log()
                print("bench_test timed out after 30 minutes.")

    with open("performance.txt", "w", encoding="utf-8") as f:
        f.write(parse_log_file())

    # TODO(KuilongCui): change clear_state function to fixture
    shutdown_llumnix_service()
    clear_ray_state()
    await asyncio.sleep(3)
