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

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from collections import defaultdict
import re

import pytest
import torch
import ray

from llumnix.utils import get_ip_address, wait_port_free

# pylint: disable=unused-import
from tests import conftest
from tests.conftest import ray_env
from tests.e2e_test.utils import (generate_bench_command, to_markdown_table,
                                  wait_for_llumnix_service_ready, shutdown_llumnix_service,
                                  generate_bladellm_serve_command, check_log_exception,
                                  generate_vllm_serve_command)
from tests.utils import try_convert_to_local_path


size_pattern = re.compile(r'total_kv_cache_size:\s*([\d.]+)\s*(B|KB|MB|GB|KB|TB)')
speed_pattern = re.compile(r'speed:\s*([\d.]+)GB/s')

MIGRATION_BENCH_TIMEOUT_MINS = 30
# Used to caculate port to avoid port conficts between tests.
test_times = 0


def parse_instance_log_file(log_files):
    speed_dict = defaultdict(list)

    for log_file in log_files:
        with open(log_file, 'r', encoding="utf-8") as file:
            for line in file:
                size_match = size_pattern.search(line)
                speed_match = speed_pattern.search(line)

                if size_match and speed_match:
                    total_kv_cache_size = size_match.group(0).split(": ")[1].strip()
                    speed = float(speed_match.group(1))
                    speed_dict[total_kv_cache_size].append(speed)

    average_speed = {}
    for transfer_size, speeds in speed_dict.items():
        if len(speeds) <= 2:
            continue

        speeds.sort()
        trimmed_speeds = speeds[1:-1]

        if len(trimmed_speeds) > 0:
            average_speed[transfer_size] = sum(trimmed_speeds) / len(trimmed_speeds)

    assert len(average_speed) > 0, "Migration should have occurred, but it was not detected. "

    return average_speed

def wait_for_all_instances_finished():
    actor_infos = ray.util.list_named_actors(True)
    instance_actor_handles = []
    for actor_info in actor_infos:
        if actor_info['name'].startswith("instance"):
            instance_actor_handles.append(ray.get_actor(actor_info['name'], namespace=actor_info['namespace']))
    while True:
        all_finished = True
        for instance in instance_actor_handles:
            running_request_ids = ray.get(instance.execute_engine_method.remote("get_running_queue"))
            waiting_request_ids = ray.get(instance.execute_engine_method.remote("get_waiting_queue"))
            if len(running_request_ids) != 0 or len(waiting_request_ids) != 0:
                all_finished = False
        if all_finished:
            break

def get_instance_num_blocks():
    actor_infos = ray.util.list_named_actors(True)
    instance_actor_handles = []
    for actor_info in actor_infos:
        if actor_info['name'].startswith("instance"):
            instance_actor_handles.append(ray.get_actor(actor_info['name'], namespace=actor_info['namespace']))
    instance_num_blocks_list = []
    for instance in instance_actor_handles:
        instance_info = ray.get(instance.get_instance_info.remote())
        instance_num_blocks_list.append(instance_info.num_available_gpu_blocks)

    return instance_num_blocks_list

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen2.5-7B')])
@pytest.mark.parametrize("migration_request_status", ['running', 'waiting'])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("migration_backend", ['gloo', 'grpc', 'kvtransfer'])
@pytest.mark.parametrize("use_ray_spmd_worker", [True, False])
@pytest.mark.parametrize("engine", ["engine_vLLM", "engine_BladeLLM"])
async def test_migration_benchmark(request, ray_env, shutdown_llumnix_service, check_log_exception, model,
                                   migration_request_status, tensor_parallel_size, migration_backend,
                                   use_ray_spmd_worker, engine):
    engine = engine.split("_")[1]

    num_prompts = 500

    if any(item in request.node.name for item in ["waiting", "grpc"]) or \
        tensor_parallel_size == 2:
        num_prompts = int(num_prompts/10)

    if "BladeLLM" in engine and use_ray_spmd_worker:
        conftest.SKIP_REASON = "use_ray_spmd_worker is vLLM config, just skip it in BladeLLM."

    if "BladeLLM" in engine and migration_request_status == 'waiting':
        conftest.SKIP_REASON = "BladeLLM does not support migrating waiting request temporarily."

    if "BladeLLM" in engine and migration_backend not in ['grpc', 'kvtransfer']:
        conftest.SKIP_REASON = f"BladeLLM does not support migration backend {migration_backend}"

    if "vLLM" in engine and tensor_parallel_size == 2:
        conftest.SKIP_REASON = "vLLM tensor_parallel_size=2 has already been tested in the correctness test."

    if "vLLM" in engine and migration_backend != 'gloo':
        conftest.SKIP_REASON = f"vLLM does not support migration backend {migration_backend}."

    if conftest.SKIP_REASON is not None and len(conftest.SKIP_REASON) > 0:
        pytest.skip(conftest.SKIP_REASON)

    if use_ray_spmd_worker:
        os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
        os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "1"
    else:
        os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "0"
        os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "0"

    global test_times
    print("Going to set new env...")
    request_migration_policy = 'SR' if migration_request_status == 'running' else 'FCW'
    ip = get_ip_address()
    base_port = 30000 + test_times * 100
    if "BladeLLM" in engine:
        base_port += 5000
    ip_ports = []
    instance_output_logs = []
    device_count = min(4, torch.cuda.device_count())
    num_instances = device_count // tensor_parallel_size

    for i in range(num_instances):
        port = base_port + i
        wait_port_free(port, force=True)
        ip_ports.append(f"{ip}:{port}")
    result_filename = f"{base_port}.out"
    instance_output_logs.append("instance_"+result_filename)

    if "vLLM" in engine:
        generate_serve_command = generate_vllm_serve_command
    elif "BladeLLM" in engine:
        generate_serve_command = generate_bladellm_serve_command
    else:
        raise ValueError(f"Unknown engine: {engine}")

    launch_command = generate_serve_command(
                        result_filename=result_filename,
                        ip=ip,
                        port=base_port,
                        model=model,
                        dispatch_policy="flood",
                        migration_backend=migration_backend,
                        request_migration_policy=request_migration_policy,
                        tensor_parallel_size=tensor_parallel_size,
                        enforce_eager=False,
                        max_instances=num_instances)
    print(f"Going to run command: {launch_command}")
    subprocess.run(launch_command, shell=True, check=True)

    wait_for_llumnix_service_ready(ip_ports)

    instance_num_blocks_list_before_bench = get_instance_num_blocks()

    def run_bench_command(command):
        # pylint: disable=consider-using-with
        process = subprocess.Popen(command, shell=True)
        return process

    tasks = []
    for i in range(num_instances):
        bench_command = generate_bench_command(
            backend=engine,
            ip_ports=ip_ports[i],
            model=model,
            num_prompts=num_prompts,
            dataset_type="sharegpt",
            dataset_path="/mnt/dataset/sharegpt_gpt4/sharegpt_gpt4.jsonl",
            qps=15,
            results_filename=f"{base_port+i}.out"
        )
        tasks.append(bench_command)

    # Execute the commands concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        future_to_command = {executor.submit(run_bench_command, command): command for command in tasks}

        for future in as_completed(future_to_command):
            process = future.result()
            process.wait()
            assert process.returncode == 0, "migration_test failed with return code {}.".format(process.returncode)

    wait_for_all_instances_finished()
    instance_num_blocks_list_after_bench = get_instance_num_blocks()

    assert instance_num_blocks_list_before_bench == instance_num_blocks_list_after_bench

    if num_prompts == 500:
        average_speed = parse_instance_log_file(instance_output_logs)
        sorted_keys = sorted(average_speed.keys(), key=lambda x: float(x.split()[0])*1024 if 'GB' in x else float(x.split()[0]))
        data = [
            ['migration_size'] + sorted_keys,
            [f'{migration_backend}_speed(GB/s)'] + [f"{average_speed[key]:.2f}" for key in sorted_keys]
        ]
        with open("performance.txt", "a", encoding="utf-8") as f:
            f.write(f"Run Test: {request.node.name}\n")
            f.write(to_markdown_table(data))
        await asyncio.sleep(10.0)

    await asyncio.sleep(3)

    test_times += 1
