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
import time
import subprocess
import uuid
from typing import Optional

import pytest
import requests

from llumnix.utils import get_ip_address, try_convert_to_local_path


def generate_vllm_launch_command(
    result_filename: str = "",
    launch_ray_cluster: bool = False,
    HEAD_NODE_IP: str = "127.0.0.1",
    ip: str = get_ip_address(),
    port: int = 37000,
    instances_num: int = 1,
    dispatch_policy: str = "load",
    migration_backend: str = "gloo",
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    max_model_len: int = 4096,
    log_instance_info: bool = False,
    log_request_timestamps: bool = False,
    request_migration_policy: str = 'SR',
    max_num_batched_tokens: int = 16000,
    enable_pd_disagg: bool = False,
    instance_type: str = "no_constraints",
    tensor_parallel_size: int = 1,
    enable_simulator: bool = False,
    request_output_queue_type: str = "zmq",
    config_path: str = "configs/vllm.yml",
    enable_migration: bool = True,
    enforce_eager: bool = True,
    **kwargs
):
    command = (
        f"RAY_DEDUP_LOGS=0 HEAD_NODE_IP={HEAD_NODE_IP} HEAD_NODE=1 "
        f"nohup python -u -m llumnix.entrypoints.vllm.api_server "
        f"--host {ip} "
        f"--port {port} "
        f"--initial-instances {instances_num} "
        f"{'--log-filename manager ' if log_instance_info else ''}"
        f"{'--log-instance-info ' if log_instance_info else ''}"
        f"{'--log-request-timestamps ' if log_request_timestamps else ''}"
        f"{'--enable-migration' if enable_migration else ''} "
        f"--model {model} "
        f"--worker-use-ray "
        f"{'--enforce-eager' if enforce_eager else ''} "
        f"--max-model-len {max_model_len} "
        f"--dispatch-policy {dispatch_policy} "
        f"--trust-remote-code "
        f"--request-migration-policy {request_migration_policy} "
        f"--migration-backend {migration_backend} "
        f"--migration-buffer-blocks 32 "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--request-output-queue-type {request_output_queue_type} "
        f"--request-output-queue-port {port + 10} "
        f"{'--launch-ray-cluster ' if launch_ray_cluster else ''}"
        f"{'--enable-pd-disagg ' if enable_pd_disagg else ''}"
        f"--config-file {config_path} "
        f"--instance-type {instance_type} "
        f"--max-num-batched-tokens {max_num_batched_tokens} "
        f"{'--simulator-mode ' if enable_simulator else ''}"
        f"{'--profiling-result-file-path /mnt/model/simulator/Qwen-7B.pkl ' if enable_simulator else ''}"
        f"{'--disable-async-output-proc ' if enable_simulator else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

def generate_vllm_serve_command(
    result_filename: str = "",
    ip: str = get_ip_address(),
    port: int = 37000,
    dispatch_policy: str = "load",
    migration_backend: str = "gloo",
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    max_model_len: int = 4096,
    log_instance_info: bool = False,
    log_request_timestamps: bool = True,
    request_migration_policy: str = 'SR',
    max_num_batched_tokens: int = 16000,
    enable_pd_disagg: bool = False,
    pd_ratio: str = "1:1",
    enable_simulator: bool = False,
    request_output_queue_type: str = "zmq",
    config_path: str = "configs/vllm.yml",
    tensor_parallel_size: int = 1,
    enable_migration: bool = True,
    enforce_eager: bool = True,
    max_instances: int = 4,
    **kwargs
):
    command = (
        f"RAY_DEDUP_LOGS=0 "
        f"nohup python -u -m llumnix.entrypoints.vllm.serve "
        f"--host {ip} "
        f"--port {port} "
        f"{'--log-filename manager ' if log_instance_info else ''}"
        f"{'--log-instance-info ' if log_instance_info else ''}"
        f"{'--log-request-timestamps ' if log_request_timestamps else ''}"
        f"{'--enable-migration' if enable_migration else ''} "
        f"--model {model} "
        f"--worker-use-ray "
        f"{'--enforce-eager' if enforce_eager else ''} "
        f"--max-model-len {max_model_len} "
        f"--dispatch-policy {dispatch_policy} "
        f"--trust-remote-code "
        f"--request-migration-policy {request_migration_policy} "
        f"--migration-backend {migration_backend} "
        f"--migration-buffer-blocks 32 "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--request-output-queue-type {request_output_queue_type} "
        f"--request-output-queue-port {port + 10} "
        f"--max-num-batched-tokens {max_num_batched_tokens} "
        f"--pd-ratio {pd_ratio} "
        f"--enable-port-increment "
        f"--max-instances {max_instances} "
        f"{'--enable-pd-disagg ' if enable_pd_disagg else ''}"
        f"{'--simulator-mode ' if enable_simulator else ''}"
        f"--config-file {config_path} "
        f"{'--profiling-result-file-path /mnt/model/simulator/Qwen-7B.pkl ' if enable_simulator else ''}"
        f"{'--disable-async-output-proc ' if enable_simulator else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

NAMING_URL = "file:/tmp/llumnix/naming"

def generate_bladellm_launch_command(
    config_file: str = "configs/bladellm.yml",
    result_filename: str = "",
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    HEAD_NODE_IP: str = "127.0.0.1",
    ip: str = get_ip_address(),
    port: int = 37000,
    max_num_batched_tokens: int = 4096,
    enable_llumnix: bool = True,
    enable_pd_disagg: bool = False,
    enable_migration: bool = True,
    dispatch_policy: str = "load",
    instance_type: str = "prefill",
    engine_disagg_transfer_type: str = "rdma",
    max_gpu_memory_utilization: float = 0.85,
    migration_backend: str = "grpc",
    tensor_parallel_size: int = 1,
    cuda_visiable_device: Optional[str] = None,
    **kwargs
):
    command = (
        f"RAY_DEDUP_LOGS=0 HEAD_NODE_IP={HEAD_NODE_IP} HEAD_NODE=1 "
        f"{f'CUDA_VISIBLE_DEVICES={cuda_visiable_device} ' if cuda_visiable_device else ''}"
        f"nohup blade_llm_server "
        f"--host {ip} "
        f"--port {port} "
        f"--model {model} "
        f"{'--enable_llumnix' if enable_llumnix else ''} "
        f"--llumnix_config {config_file} "
        f"--disable_prompt_cache "
        f"--log_level INFO "
        f"-tp {tensor_parallel_size} "
        f"--dist_init_addr {ip}:{port+30} "
        f"--attn_cls ragged_flash "
        f"--ragged_flash_max_batch_tokens {max_num_batched_tokens} "
        f"--disable_frontend_multiprocessing "
        f"--max_gpu_memory_utilization {max_gpu_memory_utilization} "
        f"{'--enable_disagg' if enable_pd_disagg else ''} "
        f"--disagg_pd.inst_id={str(uuid.uuid4().hex)[:8]} "
        f"--disagg_pd.disagg_transfer_type={engine_disagg_transfer_type} "
        f"--disagg_pd.inst_role={instance_type} "
        f"--naming_url={NAMING_URL} "
        f"INSTANCE.GRPC_MIGRATION_BACKEND_SERVER_PORT {port + 20} "
        f"MANAGER.DISPATCH_POLICY {dispatch_policy} "
        f"MANAGER.ENABLE_MIGRATION {enable_migration} "
        f"INSTANCE.MIGRATION_BACKEND {migration_backend} "
        f"{'> instance_'+result_filename if len(result_filename) > 0 else ''} 2>&1 &"
    )
    return command

def generate_bladellm_serve_command(
    config_file: str = "configs/bladellm.yml",
    result_filename: str = "",
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    ip: str = get_ip_address(),
    port: int = 37000,
    max_num_batched_tokens: int = 16000,
    enable_llumnix: bool = True,
    enable_pd_disagg: bool = False,
    enable_migration: bool = True,
    dispatch_policy: str = "load",
    instance_type: str = "prefill",
    engine_disagg_transfer_type: str = "ipc",
    max_gpu_memory_utilization: float = 0.85,
    migration_backend: str = "grpc",
    tensor_parallel_size: int = 1,
    max_instances: int = 4,
    **kwargs
):
    command = (
        f"RAY_DEDUP_LOGS=0 "
        f"nohup python -u -m llumnix.entrypoints.bladellm.serve "
        f"--host {ip} "
        f"--port {port} "
        f"--model {model} "
        f"{'--enable_llumnix' if enable_llumnix else ''} "
        f"--llumnix_config {config_file} "
        f"--disable_prompt_cache "
        f"--log_level INFO "
        f"-tp {tensor_parallel_size} "
        f"--attn_cls ragged_flash "
        f"--ragged_flash_max_batch_tokens {max_num_batched_tokens} "
        f"--disable_frontend_multiprocessing "
        f"--max_gpu_memory_utilization {max_gpu_memory_utilization} "
        f"{'--enable_disagg' if enable_pd_disagg else ''} "
        f"--disagg_pd.inst_id={str(uuid.uuid4().hex)[:8]} "
        f"--disagg_pd.disagg_transfer_type={engine_disagg_transfer_type} "
        f"--disagg_pd.inst_role={instance_type} "
        f"--disagg_pd.token_port={port + 10} "
        f"--naming_url={NAMING_URL} "
        f"INSTANCE.GRPC_MIGRATION_BACKEND_SERVER_PORT {port + 20} "
        f"MANAGER.DISPATCH_POLICY {dispatch_policy} "
        f"MANAGER.ENABLE_MIGRATION {enable_migration} "
        f"INSTANCE.MIGRATION_BACKEND {migration_backend} "
        f"MANAGER.ENABLE_PORT_INCREMENT True "
        f"MANAGER.MAX_INSTANCES {max_instances} "
        f"{'> instance_'+result_filename if len(result_filename) > 0 else ''} 2>&1 &"
    )
    return command


def generate_vllm_request(prompt):
    request = {
        "prompt": prompt,
        "stream": False,
        "n": 1,
        "best_of": 1,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": False,
    }
    return request

def process_vllm_api_server_output(output):
    return output['text'][0]

def generate_bladellm_request(prompt):
    request = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "n": 1,
        "best_of": 1,
        "temperature": 0.0,
        "stream": "false",
        "ignore_eos": "false",
        "presence_penalty": 1.1,
        "repetition_penalty": 1.1,
    }
    return request

def process_bladellm_api_server_output(output):
    return output['choices'][0]['message']['content']

def wait_for_llumnix_service_ready(ip_ports, timeout=120):
    start_time = time.time()
    while True:
        all_ready = True
        for ip_port in ip_ports:
            try:
                response = requests.get(f"http://{ip_port}/is_ready", timeout=5)
                print(f"Entrypoint {ip_port} is ready.")
                if 'true' not in response.text.lower():
                    all_ready = False
                    break
            except requests.RequestException:
                all_ready = False
                break

        if all_ready:
            return True

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"Wait for llumnix service timeout ({timeout}s).")

        time.sleep(5.0)

def generate_bench_command(backend: str,
                           ip_ports: str,
                           model: str,
                           num_prompts: int,
                           dataset_type: str,
                           dataset_path: str,
                           qps: int,
                           results_filename: str = "",
                           query_distribution: str = "poisson",
                           coefficient_variation: float = 1.0,
                           priority_ratio: float = 0.0):
    command = (
        f"python -u ./benchmark/benchmark_serving.py "
        f"--ip_ports {ip_ports} "
        f"--backend {backend} "
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

def shutdown_llumnix_service_func():
    subprocess.run('pkill -f llumnix.entrypoints.vllm.api_server', shell=True, check=False)
    subprocess.run('pkill -f benchmark_serving.py', shell=True, check=False)
    subprocess.run('pkill -f llumnix.entrypoints.vllm.serve', shell=True, check=False)
    subprocess.run('pkill -f blade_llm_server', shell=True, check=False)
    subprocess.run('pkill -f llumnix.entrypoints.bladellm.serve', shell=True, check=False)
    subprocess.run('pkill -f multiprocessing', shell=True, check=False)
    subprocess.run('rm -rf /tmp/kvt-*', shell=True, check=False)
    subprocess.run(f'rm -rf {NAMING_URL.split(":")[1] + "/*"}', shell=True, check=False)
    time.sleep(5.0)

@pytest.fixture
def shutdown_llumnix_service():
    subprocess.run('rm -rf instance_*.out', shell=True, check=False)
    subprocess.run('rm -rf nohup.out', shell=True, check=False)
    yield
    shutdown_llumnix_service_func()

def count_tracebacks_in_instances(directory):
    def count_traceback_in_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return content.count("Traceback (most recent call last)")
        # pylint: disable=bare-except
        except:
            return 0

    total_count = 0
    for filename in os.listdir(directory):
        if filename.startswith('instance_'):
            file_path = os.path.join(directory, filename)
            count = count_traceback_in_file(file_path)
            total_count += count
    return total_count

def check_log_exception():
    total_traceback = count_tracebacks_in_instances('.')
    assert total_traceback == 0, f'There are {total_traceback} tracebacks in log files, check the log files.'

def to_markdown_table(data):
    headers = data[0]
    rows = data[1:]

    col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

    header_row = " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(headers))
    separator_row = " | ".join('-' * col_widths[i] for i in range(len(headers)))

    data_rows = []
    for row in rows:
        data_row = " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row))
        data_rows.append(data_row)

    table = f"{header_row}\n{separator_row}\n" + "\n".join(data_rows) + "\n\n"
    return table
