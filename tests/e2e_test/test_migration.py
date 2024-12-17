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
from collections import defaultdict
import re
import pytest
import torch
import ray

# pylint: disable=unused-import
from tests.conftest import ray_env
from tests.e2e_test.utils import (generate_launch_command, generate_bench_command, to_markdown_table,
                    wait_for_llumnix_service_ready, shutdown_llumnix_service)

size_pattern = re.compile(r'total_kv_cache_size:\s*([\d.]+)\s*(B|KB|MB|GB|KB|TB)')
speed_pattern = re.compile(r'speed:\s*([\d.]+)GB/s')

MIGRATION_BENCH_TIMEOUT_MINS = 30


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
    named_actors = ray.util.list_named_actors(True)
    instance_actor_handles = []
    for actor in named_actors:
        if actor['name'].startswith("instance"):
            instance_actor_handles.append(ray.get_actor(actor['name'], namespace=actor['namespace']))
    while True:
        all_finished = True
        for instance in instance_actor_handles:
            all_request_ids = ray.get(instance.execute_engine_method.remote("get_all_request_ids"))
            if len(all_request_ids) != 0:
                all_finished = False
        if all_finished:
            break

def get_instance_num_blocks():
    named_actors = ray.util.list_named_actors(True)
    instance_actor_handles = []
    for actor in named_actors:
        if actor['name'].startswith("instance"):
            instance_actor_handles.append(ray.get_actor(actor['name'], namespace=actor['namespace']))
    instance_num_blocks_list = []
    for instance in instance_actor_handles:
        instance_info = ray.get(instance.get_instance_info.remote())
        instance_num_blocks_list.append(instance_info.num_available_gpu_blocks)

    return instance_num_blocks_list

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="at least 2 gpus required for migration bench")
@pytest.mark.parametrize("model", ['/mnt/model/Qwen-7B'])
@pytest.mark.parametrize("migration_backend", ['rayrpc', 'gloo', 'nccl'])
@pytest.mark.parametrize("migrated_request_status", ['running', 'waiting'])
async def test_migration_benchmark(ray_env, shutdown_llumnix_service, model, migration_backend, migrated_request_status):
    if migrated_request_status == 'waiting' and migration_backend != 'rayrpc':
        pytest.skip("When the migrated request status is waiting, only test the rayrpc migration backend.")

    request_migration_policy = 'SR' if migrated_request_status == 'running' else 'FCW'
    ip = "127.0.0.1"
    base_port = 37037
    ip_ports = []
    instance_output_logs = []
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        port = base_port + i
        ip_ports.append(f"{ip}:{base_port+i}")
        output_log = f"{base_port+i}.out"
        instance_output_logs.append("instance_"+output_log)
        launch_command = generate_launch_command(result_filename=output_log,
                                                 launch_ray_cluster=False,
                                                 ip=ip,
                                                 port=port,
                                                 model=model,
                                                 dispatch_policy="flood",
                                                 migration_backend=migration_backend,
                                                 request_migration_policy=request_migration_policy)
        subprocess.run(launch_command, shell=True, check=True)

    wait_for_llumnix_service_ready(ip_ports)

    instance_num_blocks_list_before_bench = get_instance_num_blocks()

    def run_bench_command(command):
        # pylint: disable=consider-using-with
        process = subprocess.Popen(command, shell=True)
        return process

    tasks = []
    for i in range(device_count // 2):
        bench_command = generate_bench_command(
            ip_ports=f"127.0.0.1:{base_port + i}",
            model=model,
            num_prompts=500,
            dataset_type="sharegpt",
            dataset_path="/mnt/dataset/sharegpt_gpt4/sharegpt_gpt4.jsonl",
            qps=10,
            results_filename=f"{base_port + i}.out"
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

    if migrated_request_status == 'running':
        average_speed = parse_instance_log_file(instance_output_logs)
        sorted_keys = sorted(average_speed.keys(), key=lambda x: float(x.split()[0]))
        data = [
            ['migration_size'] + sorted_keys,
            [f'{migration_backend}_speed(GB/s)'] + [f"{average_speed[key]:.2f}" for key in sorted_keys]
        ]
        with open("performance.txt", "a", encoding="utf-8") as f:
            f.write(to_markdown_table(data))

    await asyncio.sleep(3)
