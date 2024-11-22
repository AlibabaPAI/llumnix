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

import asyncio
from collections import defaultdict
import re
import subprocess
import pytest
import torch
import pandas as pd

from .test_e2e import generate_launch_command
from ..test_bench import generate_bench_command, clear_ray_state, shutdown_llumnix_service
from ..utils import to_markdown_table

size_pattern = re.compile(r'total_kv_cache_size:\s*([\d.]+)\s*(B|KB|MB|GB|KB|TB)')
speed_pattern = re.compile(r'speed:\s*([\d.]+)GB/s')


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

    averger_speed = {}
    for transfer_size, speeds in speed_dict.items():
        if len(speeds) <= 2:
            continue

        speeds.sort()
        trimmed_speeds = speeds[1:-1]
        averger_speed[transfer_size] = sum(trimmed_speeds) / len(trimmed_speeds)

    assert len(averger_speed) > 0, "Migration should have occurred, but it was not detected. "

    return averger_speed

def parse_manager_log_file(log_file):
    df = pd.read_csv(log_file)
    instance_id_set = set(df["instance_id"])
    for instance_id in instance_id_set:
        df_instance = df[df["instance_id"] == instance_id]
        num_available_gpu_blocks_list = df_instance["num_available_gpu_blocks"].to_numpy().tolist()
        assert num_available_gpu_blocks_list[0] == num_available_gpu_blocks_list[-1]

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="at least 2 gpus required for migration bench")
@pytest.mark.parametrize("model", ['/mnt/model/Qwen-7B'])
@pytest.mark.parametrize("migration_backend", ['rpc', 'gloo', 'nccl'])
async def test_migration_benchmark(model, migration_backend):
    base_port = 37037
    instance_output_logs = []

    device_count = torch.cuda.device_count()
    for i in range(device_count):
        output_log = f"{base_port+i}.out"
        instance_output_logs.append("instance_"+output_log)
        launch_command = generate_launch_command(result_filename=output_log, launch_ray_cluster=False, port=base_port+i,
                                                 model=model, dispatch_policy="flood", migration_backend=migration_backend,
                                                 log_instance_info=True)
        subprocess.run(launch_command, shell=True, check=True)
    await asyncio.sleep(60)

    async def run_bench_command(command):
        process = await asyncio.create_subprocess_shell(command)
        await process.wait()
        assert process.returncode == 0

    for i in range(device_count//2):
        bench_command = generate_bench_command(ip_ports=f"127.0.0.1:{base_port+i}", model=model, num_prompts=300,
                                               dataset_type="sharegpt",
                                               dataset_path="/mnt/dataset/sharegpt_gpt4/sharegpt_gpt4.jsonl" ,
                                               qps=10)
        await asyncio.wait_for(run_bench_command(bench_command), timeout=60*30)
    await asyncio.sleep(30)

    parse_manager_log_file("manager_instance.csv")

    averger_speed = parse_instance_log_file(instance_output_logs)

    sorted_keys = sorted(averger_speed.keys(), key=lambda x: float(x.split()[0]))

    data = [
        ['migration_size'] + sorted_keys,
        [f'{migration_backend}_speed(GB/s)'] + [f"{averger_speed[key]:.2f}" for key in sorted_keys]
    ]

    with open("performance.txt", "a", encoding="utf-8") as f:
        f.write(to_markdown_table(data))

    shutdown_llumnix_service()
    clear_ray_state()
    await asyncio.sleep(3)
