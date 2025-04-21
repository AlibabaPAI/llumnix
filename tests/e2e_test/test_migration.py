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

from llumnix.utils import get_ip_address, try_convert_to_local_path

# pylint: disable=unused-import
from tests.conftest import ray_env
from tests.e2e_test.utils import (generate_bench_command, to_markdown_table, wait_for_llumnix_service_ready,
                                  shutdown_llumnix_service, generate_bladellm_serve_command,
                                  check_log_exception, generate_vllm_serve_command)

size_pattern = re.compile(r'total_kv_cache_size:\s*([\d.]+)\s*(B|KB|MB|GB|KB|TB)')
speed_pattern = re.compile(r'speed:\s*([\d.]+)GB/s')

MIGRATION_BENCH_TIMEOUT_MINS = 30
# Used to caculate port to avoid port conficts between tests.
test_times = 0

# TODO(s5u13b): Refine e2e tests for two backend engines.


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
            all_request_ids = ray.get(instance.execute_engine_method.remote("get_all_request_ids"))
            if len(all_request_ids) != 0:
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
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen-7B')])
@pytest.mark.parametrize("migration_backend", ['rayrpc', 'gloo', 'nccl', 'grpc', 'kvtransfer'])
@pytest.mark.parametrize("migration_request_status", ['running', 'waiting'])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("use_ray_spmd_worker", [True, False])
@pytest.mark.parametrize("engine", ["engine_vLLM", "engine_BladeLLM"])
async def test_migration_benchmark(request, ray_env, shutdown_llumnix_service, model, tensor_parallel_size,
                                   migration_backend, migration_request_status, use_ray_spmd_worker, engine):
    engine = engine.split("_")[1]

    # TODO(s5u13b): fix this bug
    if "BladeLLM" in engine and tensor_parallel_size > 1:
        pytest.skip("Error in BladeLLM for tensor parallel size > 1.")

    if "BladeLLM" in engine and use_ray_spmd_worker:
        pytest.skip("use_ray_spmd_worker is vLLM config, just skip it in BladeLLM.")

    if engine == "BladeLLM" and migration_backend not in ['grpc', 'kvtransfer']:
        pytest.skip(f"BladeLLM does not support migration backend {migration_backend}")

    if engine == "vLLM" and migration_backend not in ['rayrpc', 'gloo', 'nccl']:
        pytest.skip(f"vLLM does not support migration backend {migration_backend}.")

    if migration_request_status == 'waiting' and engine == 'BladeLLM':
        pytest.skip("BladeLLM does not support migrating waiting request temporarily.")

    if migration_request_status == 'waiting' and migration_backend != 'rayrpc':
        pytest.skip("When the migrated request status is waiting, only test the rayrpc migration backend.")

    if tensor_parallel_size == 2 and migration_backend == 'nccl':
        pytest.skip("When the migration backend is nccl, tensor parallelism is not supported.")
    if use_ray_spmd_worker and migration_backend != 'gloo':
        pytest.skip("When use_ray_spmd_worker is True, only test the gloo migration backend.")
    if use_ray_spmd_worker and tensor_parallel_size == 2:
        pytest.skip("When using ray spmd worker, ray will raise RayCgraphCapacityExceeded exeception when tensor parallelism is enabled.")
    if use_ray_spmd_worker and migration_request_status == 'waiting':
        pytest.skip("When using ray spmd worker, only migrating running request will have different migration process.")

    if use_ray_spmd_worker:
        os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
        os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "1"
    else:
        os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "0"
        os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "0"

    global test_times

    request_migration_policy = 'SR' if migration_request_status == 'running' else 'FCW'
    ip = get_ip_address()
    base_port = 30000 + test_times * 100
    ip_ports = []
    instance_output_logs = []
    device_count = min(4, torch.cuda.device_count())
    num_instances = device_count // tensor_parallel_size

    if engine == "vLLM":
        for i in range(num_instances):
            ip_ports.append(f"{ip}:{base_port+i}")
        result_filename = f"{base_port}.out"
        instance_output_logs.append("instance_"+result_filename)
        launch_command = generate_vllm_serve_command(
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
        subprocess.run(launch_command, shell=True, check=True)
    else:
        for i in range(num_instances):
            ip_ports.append(f"{ip}:{base_port+i}")
        result_filename = f"{base_port}.out"
        instance_output_logs.append("instance_"+result_filename)
        launch_command = generate_bladellm_serve_command(
                            result_filename=result_filename,
                            ip=ip,
                            port=base_port,
                            model=model,
                            dispatch_policy="flood",
                            migration_backend=migration_backend,
                            tensor_parallel_size=tensor_parallel_size,
                            max_instances=num_instances)
        subprocess.run(launch_command, shell=True, check=True)

    wait_for_llumnix_service_ready(ip_ports)

    instance_num_blocks_list_before_bench = get_instance_num_blocks()

    def run_bench_command(command):
        # pylint: disable=consider-using-with
        process = subprocess.Popen(command, shell=True)
        return process

    tasks = []
    for i in range(num_instances // 2):
        bench_command = generate_bench_command(
            backend=engine,
            ip_ports=ip_ports[i],
            model=model,
            num_prompts=500,
            dataset_type="sharegpt",
            dataset_path="/mnt/dataset/sharegpt_gpt4/sharegpt_gpt4.jsonl",
            qps=10,
            results_filename=f"{base_port+i}.out"
        )
        tasks.append(bench_command)

    # Execute the commands concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        future_to_command = {executor.submit(run_bench_command, command): command for command in tasks}

        for future in as_completed(future_to_command):
            try:
                process = future.result()
                process.wait(timeout=60*MIGRATION_BENCH_TIMEOUT_MINS)

                assert process.returncode == 0, "migration_test failed with return code {}.".format(process.returncode)
            # pylint: disable=broad-except
            except subprocess.TimeoutExpired:
                process.kill()
                assert False, "migration_test timed out after {} minutes.".format(MIGRATION_BENCH_TIMEOUT_MINS)

    wait_for_all_instances_finished()
    instance_num_blocks_list_after_bench = get_instance_num_blocks()

    assert instance_num_blocks_list_before_bench == instance_num_blocks_list_after_bench

    if migration_request_status == 'running' and tensor_parallel_size == 1 and not use_ray_spmd_worker:
        average_speed = parse_instance_log_file(instance_output_logs)
        sorted_keys = sorted(average_speed.keys(), key=lambda x: float(x.split()[0])*1024 if 'GB' in x else float(x.split()[0]))
        data = [
            ['migration_size'] + sorted_keys,
            [f'{migration_backend}_speed(GB/s)'] + [f"{average_speed[key]:.2f}" for key in sorted_keys]
        ]
        with open("performance.txt", "a", encoding="utf-8") as f:
            f.write(f"Run Test: {request.node.name}")
            f.write(to_markdown_table(data))
        await asyncio.sleep(10.0)

    await asyncio.sleep(3)

    check_log_exception()

    test_times += 1
