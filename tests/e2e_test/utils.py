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

import copy
import os
import time
import subprocess
import uuid
from typing import List, Optional, Tuple, Dict

import pytest
import requests
import aiohttp

from llumnix.utils import get_ip_address

from tests import conftest
from tests.utils import try_convert_to_local_path

def parse_launch_command(launch_command: str) -> Tuple[Dict[str, str], str]:
    """Parse launch command to process_env and command"""
    process_env = {}
    command_parts = launch_command.split()
    command_start_index = 0
    for i, part in enumerate(command_parts):
        if '=' in part:
            # Check that it's a KEY=VALUE pair at the start, not an argument like --key=value
            if not part.startswith('-'):
                key, value = part.split('=', 1)
                process_env[key] = value
                # Mark this part as an environment variable
                command_start_index = i + 1
            else:
                # It's an argument, so environment variable declarations have ended
                break
        else:
            # This is the first part without an '=', so the command starts here
            break

    actual_command = ' '.join(command_parts[command_start_index:])
    env = os.environ.copy()
    for k, v in process_env.items():
        env[k] = v

    return env, actual_command

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
    request_migration_policy: str = 'SR',
    max_num_batched_tokens: int = 16000,
    enable_pd_disagg: bool = False,
    enable_adaptive_pd: bool = False,
    instance_type: str = "neutral",
    tensor_parallel_size: int = 1,
    enable_simulator: bool = False,
    request_output_queue_type: str = "zmq",
    config_file: str = "configs/vllm.yml",
    enable_migration: bool = True,
    enforce_eager: bool = False,
    request_output_forwarding_mode: str = "thread",
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
        f"{'--launch-ray-cluster ' if launch_ray_cluster else ''}"
        f"{'--enable-pd-disagg ' if enable_pd_disagg else ''}"
        f"--config-file {config_file} "
        f"--instance-type {instance_type} "
        f"--request-output-forwarding-mode {request_output_forwarding_mode} "
        f"--max-num-batched-tokens {max_num_batched_tokens} "
        f"{'--enable-adaptive-pd ' if enable_adaptive_pd else ''}"
        f"{'--simulator-mode ' if enable_simulator else ''}"
        f"{'--profiling-result-file-path /mnt/model/simulator/Qwen2.5-7B.pkl ' if enable_simulator else ''}"
        f"{'--disable-async-output-proc ' if enable_simulator else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    print(f"Going to run command: {command}")
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
    request_migration_policy: str = 'SR',
    max_num_batched_tokens: int = 16000,
    enable_pd_disagg: bool = False,
    enable_adaptive_pd: bool = False,
    pd_ratio: str = "1:1",
    enable_simulator: bool = False,
    request_output_queue_type: str = "zmq",
    config_path: str = "configs/vllm.yml",
    tensor_parallel_size: int = 1,
    enable_migration: bool = True,
    enforce_eager: bool = False,
    max_instances: int = 4,
    request_output_forwarding_mode: str = "thread",
    **kwargs
):
    command = (
        f"RAY_DEDUP_LOGS=0 "
        f"nohup python -u -m llumnix.entrypoints.vllm.serve "
        f"--host {ip} "
        f"--port {port} "
        f"{'--log-filename manager ' if log_instance_info else ''}"
        f"{'--log-instance-info ' if log_instance_info else ''}"
        f"{'--enable-migration' if enable_migration else ''} "
        f"--model {model} "
        f"--worker-use-ray "
        f"--max-num-seqs 512 "
        f"{'--enforce-eager' if enforce_eager else ''} "
        f"--max-model-len {max_model_len} "
        f"--dispatch-policy {dispatch_policy} "
        f"--trust-remote-code "
        f"--request-migration-policy {request_migration_policy} "
        f"--migration-backend {migration_backend} "
        f"--migration-buffer-blocks 32 "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--request-output-queue-type {request_output_queue_type} "
        f"--max-num-batched-tokens {max_num_batched_tokens} "
        f"--pd-ratio {pd_ratio} "
        f"--enable-port-increment "
        f"--max-instances {max_instances} "
        f"{'--enable-pd-disagg ' if enable_pd_disagg else ''}"
        f"{'--enable-adaptive-pd ' if enable_adaptive_pd else ''}"
        f"{'--simulator-mode ' if enable_simulator else ''}"
        f"--request-output-forwarding-mode {request_output_forwarding_mode} "
        f"--config-file {config_path} "
        f"{'--profiling-result-file-path /mnt/model/simulator/Qwen2.5-7B.pkl ' if enable_simulator else ''}"
        f"{'--disable-async-output-proc ' if enable_simulator else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    print(f"Going to run command: {command}")
    return command

def generate_raw_vllm_v1_serve_command(
    result_filename: str = "",
    launch_ray_cluster: bool = False,
    HEAD_NODE_IP: str = "127.0.0.1",
    ip: str = get_ip_address(),
    port: int = 37000,
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 16000,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    **kwargs
):
    """Generate raw vllm v1 serve command without Llumnix"""
    command = (
        f"VLLM_USE_V1=1 RAY_DEDUP_LOGS=0 VLLM_FORCE_DETOKENIZE=1 "
        f"nohup vllm serve {model} "
        f"--host {ip} "
        f"--port {port} "
        f"{'--enforce-eager' if enforce_eager else ''} "
        f"--max-model-len {max_model_len} "
        f"--trust-remote-code "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--max-num-batched-tokens {max_num_batched_tokens} "
        f"{'> raw_vllm_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    print(f"Going to run command: {command}")
    return command

def generate_vllm_v1_launch_command(
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
    config_file: str = "configs/vllm.yml",
    enable_migration: bool = True,
    enforce_eager: bool = False,
    request_output_forwarding_mode: str = "thread",
    **kwargs
):
    return None
def generate_vllm_v1_serve_command(
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
    enable_adaptive_pd: bool = False,
    pd_ratio: str = "1:1",
    enable_simulator: bool = False,
    request_output_queue_type: str = "zmq",
    config_path: str = "configs/vllm_v1.yml",
    tensor_parallel_size: int = 1,
    enable_migration: bool = True,
    enforce_eager: bool = False,
    max_instances: int = 4,
    request_output_forwarding_mode: str = "thread",
    **kwargs
):
    command = (
        f"{'NCCL_SOCKET_IFNAME=eth0 ' if tensor_parallel_size > 1 else ''}"
        f"VLLM_USE_V1=1 VLLM_ENABLE_LLUMNIX=1 VLLM_FORCE_DETOKENIZE=1 RAY_DEDUP_LOGS=0 "
        f"nohup python -m llumnix.entrypoints.vllm_v1.serve "
        f"--host {ip} "
        f"--port {port} "
        f"{'--log-filename manager ' if log_instance_info else ''}"
        f"{'--log-instance-info ' if log_instance_info else ''}"
        f"{'--log-request-timestamps ' if log_request_timestamps else ''}"
        f"{'--enable-migration' if enable_migration else ''} "
        f"--model {model} "
        f"--distributed-executor-backend ray "
        f"{'--enforce-eager' if enforce_eager else ''} "
        f"--max-model-len {max_model_len} "
        f"--dispatch-policy {dispatch_policy} "
        f"--trust-remote-code "
        f"--request-migration-policy {request_migration_policy} "
        f"--migration-backend {migration_backend} "
        f"--migration-buffer-blocks 32 "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--request-output-queue-type {request_output_queue_type} "
        f"--max-num-batched-tokens {max_num_batched_tokens} "
        # f"--pd-ratio {pd_ratio} "
        f"--enable-port-increment "
        f"--max-instances {max_instances} "
        f"{'--enable-pd-disagg ' if enable_pd_disagg else ''}"
        f"{'--enable-adaptive-pd ' if enable_adaptive_pd else ''}"
        f"{'--simulator-mode ' if enable_simulator else ''}"
        f"--request-output-forwarding-mode {request_output_forwarding_mode} "
        # f"--config-file {config_path} "
        f"{'--profiling-result-file-path /mnt/model/simulator/Qwen2.5-7B.pkl ' if enable_simulator else ''}"
        f"{'--disable-async-output-proc ' if enable_simulator else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    print(f"Going to run command: {command}")
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
    enable_adaptive_pd: bool = False,
    enable_migration: bool = True,
    dispatch_policy: str = "load",
    instance_type: str = "prefill",
    engine_disagg_transfer_type: str = "rdma",
    max_gpu_memory_utilization: float = 0.60,
    migration_backend: str = "grpc",
    tensor_parallel_size: int = 1,
    cuda_visiable_device: Optional[str] = None,
    request_output_queue_type: str = "zmq",
    enforce_eager: bool = False,
    pd_ratio: str = "1:1",
    request_output_forwarding_mode: str = "thread",
    enable_engine_semi_pd_disagg: bool = False,
    semi_pd_ins_id: str = "test",
    **kwargs
):
    enable_engine_semi_pd_disagg_option = f'--enable_semi_pd_mode  --semi_pd.inst_id={semi_pd_ins_id} --semi_pd.transfer_type=rdma ' \
        f'--semi_pd.prefill_server_port={port+37}'
    command = (
        f"RAY_DEDUP_LOGS=0 HEAD_NODE_IP={HEAD_NODE_IP} HEAD_NODE=1 "
        f"{f'CUDA_VISIBLE_DEVICES={cuda_visiable_device} ' if cuda_visiable_device else ''}"
        f"nohup blade_llm_server "
        f"--host {ip} "
        f"--port {port} "
        f"--model {model} "
        f"{'--enable_llumnix' if enable_llumnix else ''} "
        f"--disable_prompt_cache "
        f"--log_level INFO "
        f"-tp {tensor_parallel_size} "
        f"--dist_init_addr {ip}:{port+10} "
        f"--attn_cls ragged_flash "
        f"--ragged_flash_max_batch_tokens {max_num_batched_tokens} "
        f"--disable_frontend_multiprocessing "
        f"--max_gpu_memory_utilization {max_gpu_memory_utilization} "
        f"{'--disable_cuda_graph' if enforce_eager else ''} "
        f"{'--enable_disagg' if enable_pd_disagg else ''} "
        f"{enable_engine_semi_pd_disagg_option if enable_engine_semi_pd_disagg else ''} "
        f"--disagg_pd.inst_id={str(uuid.uuid4().hex)[:8]} "
        f"--disagg_pd.disagg_transfer_type={engine_disagg_transfer_type} "
        f"--disagg_pd.inst_role={instance_type} "
        f"--naming_url={NAMING_URL} "
        f"SERVER.REQUEST_OUTPUT_QUEUE_TYPE {request_output_queue_type} "
        f"MANAGER.ENABLE_ENGINE_PD_DISAGG {enable_pd_disagg} "
        f"MANAGER.DISPATCH_POLICY {dispatch_policy} "
        f"MANAGER.ENABLE_MIGRATION {enable_migration and not enable_pd_disagg} "
        f"INSTANCE.MIGRATION_BACKEND {migration_backend} "
        f"MANAGER.ENABLE_ADAPTIVE_PD {enable_adaptive_pd} "
        f"MANAGER.PD_RATIO {pd_ratio} "
        f"INSTANCE.REQUEST_OUTPUT_FORWARDING_MODE {request_output_forwarding_mode} "
        f"{'> instance_'+result_filename if len(result_filename) > 0 else ''} 2>&1 &"
    )
    print(f"Going to run command: {command}")
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
    enable_adaptive_pd: bool = False,
    enable_migration: bool = True,
    dispatch_policy: str = "load",
    instance_type: str = "prefill",
    engine_disagg_transfer_type: str = "rdma",
    max_gpu_memory_utilization: float = 0.60, # TODO(s5u13b): Fix OOM in TP=2.
    migration_backend: str = "grpc",
    tensor_parallel_size: int = 1,
    max_instances: int = 4,
    pd_ratio: str = "1:1",
    request_output_queue_type: str = "zmq",
    enforce_eager: bool = False,
    request_output_forwarding_mode: str = "thread",
    enable_engine_semi_pd_disagg: bool = False,
    **kwargs
):
    enable_engine_semi_pd_disagg_option = f'--enable_semi_pd_mode  --semi_pd.inst_id=test --semi_pd.transfer_type=rdma ' \
        f'--semi_pd.prefill_server_port={port+37}'
    command = (
        f"RAY_DEDUP_LOGS=0 "
        f"nohup python -u -m llumnix.entrypoints.bladellm.serve "
        f"--host {ip} "
        f"--port {port} "
        f"--model {model} "
        f"{'--enable_llumnix' if enable_llumnix else ''} "
        f"--disable_prompt_cache "
        f"--log_level INFO "
        f"-tp {tensor_parallel_size} "
        f"--attn_cls ragged_flash "
        f"--ragged_flash_max_batch_tokens {max_num_batched_tokens} "
        f"--disable_frontend_multiprocessing "
        f"--max_gpu_memory_utilization {max_gpu_memory_utilization} "
        f"{'--disable_cuda_graph' if enforce_eager else ''} "
        f"{'--enable_disagg' if enable_pd_disagg else ''} "
        f"--disagg_pd.inst_id {str(uuid.uuid4().hex)[:8]} "
        f"--disagg_pd.disagg_transfer_type {engine_disagg_transfer_type} "
        f"--disagg_pd.inst_role {instance_type} "
        f"--naming_url {NAMING_URL} "
        f"{'--enable-engine-pd-disagg' if enable_pd_disagg else ''} "
        f"{enable_engine_semi_pd_disagg_option if enable_engine_semi_pd_disagg else ''} "
        f"{'--enable-adaptive-pd ' if enable_adaptive_pd else ''}"
        f"--dispatch-policy {dispatch_policy} "
        f"--pd-ratio {pd_ratio} "
        f"{'--enable-migration' if enable_migration and not enable_pd_disagg else ''} "
        f"--migration-backend {migration_backend} "
        f"--request-output-queue-type {request_output_queue_type} "
        f"--request-output-forwarding-mode {request_output_forwarding_mode} "
        f"--enable-port-increment "
        f"--max-instances {max_instances} "
        f"{'> instance_'+result_filename if len(result_filename) > 0 else ''} 2>&1 &"
    )
    print(f"Going to run command: {command}")
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

def generate_vllm_v1_request(prompt):
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
        "temperature": 0.0,
        "stream": "false",
        "ignore_eos": "false",
        "presence_penalty": 1.1,
        "repetition_penalty": 1.1,
        "temperature": 0.0,
        "top_p": 0.5,
        "top_k": 10,
    }
    return request

def process_vllm_v1_api_server_output(output):
    return output["choices"][0]["message"]["content"]

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
        "semi_p_inst_id": "prefill",
        "semi_d_inst_id": "decode",
    }
    return request

def process_bladellm_api_server_output(output):
    return output['choices'][0]['message']['content']

async def get_llumnix_response(prompt, url, generate_request_func, process_api_server_output_func):
    timeout = aiohttp.ClientTimeout(total=60)
    request = generate_request_func(prompt)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=request) as resp:
            output = await resp.json()
            return process_api_server_output_func(output)

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

def wait_for_llumnix_service_ready_vllm_v1(ip_ports, timeout=120):
    start_time = time.time()
    while True:
        all_ready = False
        for ip_port in ip_ports:
            try:
                request = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": "hello"
                        }
                    ],
                    "max_tokens": 1,
                }
                response = requests.post(f"http://{ip_port}/v1/chat/completions", json=request, timeout=5)
                if response.status_code == 200:
                    all_ready = True
                    break
            except requests.RequestException:
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

def generate_vllm_register_service_command_func(
    engine_type: str,
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    ip: str = get_ip_address(),
    port: int = 37000
):
    command = (
        f"python -u -m llumnix.entrypoints.vllm.register_service "
        f"--engine-type {engine_type} "
        f"--save-path ./service_test "
        f"--save-key vllm "
        f"--model {model} "
        f"--max-model-len 4096 "
        f"--worker-use-ray "
        f"--enforce-eager "
        f"--trust-remote-code "
    )
    return command

def generate_vllm_serve_service_command_func(
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    ip: str = get_ip_address(),
    port: int = 37000,
    max_instances: int = 4,
    result_filename: str = ""
):
    command = (
        f"RAY_DEDUP_LOGS=0 "
        f"nohup python -u -m llumnix.entrypoints.vllm.serve "
        f"--load-registered-service "
        f"--load-registered-service-path ./service_test/vllm "
        f"--host {ip} "
        f"--port {port} "
        f"--enable-pd-disagg "
        f"--enable-migration "
        f"--pd-ratio 1:1 "
        f"--max-instances {max_instances} "
        f"--enable-port-increment "
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

def generate_vllm_v1_register_service_command_func(
    engine_type: str,
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    ip: str = get_ip_address(),
    port: int = 37000
):
    kvt_config_prefill = (
        '--kv-transfer-config \'{"kv_connector": "HybridConnector", "kv_role": "kv_producer", '
        '"kv_connector_extra_config": {"backend": "kvt", "kvt_inst_id": "prefill", '
        '"naming_url": "file:/tmp/vllm.naming"}}\''
    )

    kvt_config_decode = (
        '--kv-transfer-config \'{"kv_connector": "HybridConnector", "kv_role": "kv_consumer", '
        '"kv_connector_extra_config": {"backend": "kvt", "kvt_inst_id": "decode", '
        '"naming_url": "file:/tmp/vllm.naming"}}\''
    )

    command = (
        f"python -u -m llumnix.entrypoints.vllm_v1.register_service "
        f"--engine-type {engine_type} "
        f"--save-path ./service_test "
        f"--save-key vllm_v1 "
        f"--model {model} "
        f"--max-model-len 4096 "
        f"--distributed-executor-backend ray "
        f"--enforce-eager "
        f"--trust-remote-code "
        f"{kvt_config_prefill if engine_type == 'prefill' else kvt_config_decode if engine_type == 'decode' else ''}"
    )
    return command

def generate_vllm_v1_serve_service_command_func(
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    ip: str = get_ip_address(),
    port: int = 37000,
    max_instances: int = 4,
    result_filename: str = ""
):
    command = (
        f"BLLM_KVTRANS_FSNAMING_KEEPALIVE_S=36000 BLLM_KVTRANS_FSNAMING_TOLERATE_S=360000 "
        f"VLLM_USE_V1=1 VLLM_ENABLE_LLUMNIX=1 VLLM_FORCE_DETOKENIZE=1 RAY_DEDUP_LOGS=0 "
        f"nohup python -u -m llumnix.entrypoints.vllm_v1.serve "
        f"--load-registered-service "
        f"--load-registered-service-path ./service_test/vllm_v1 "
        f"--host {ip} "
        f"--port {port} "
        f"--enable-engine-pd-disagg "
        f"--pd-ratio 1:1 "
        f"--max-instances {max_instances} "
        f"--enable-port-increment "
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

def generate_bladellm_register_service_command_func(
    engine_type: str,
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    ip: str = get_ip_address(),
    port: int = 37000
):
    command = (
        f"python -u -m llumnix.entrypoints.bladellm.register_service "
        f"--engine-type {engine_type} "
        f"--save-path ./service_test "
        f"--save-key bladellm "
        f"--model {model} "
        f"--host {ip} " # must set, saved in engine args, default value can take effect
        f"--port {port} " # must set, saved in engine args, can take effect
        f"--enable_llumnix "
        f"--disable_frontend_multiprocessing "
        f"--disable_signal_handler "
        f"--enable_disagg "
        f"{'--disable_cuda_graph' if engine_type == 'prefill' else ''} "
        f"--disagg_pd.inst_id {str(uuid.uuid4().hex)[:8]} "
        f"--disagg_pd.inst_role {engine_type} "
        f"--disagg_pd.disagg_transfer_type rdma "
        f"--naming_url {NAMING_URL} "
    )
    return command

def generate_bladellm_serve_service_command_func(
    model: str = try_convert_to_local_path("facebook/opt-125m"),
    ip: str = get_ip_address(),
    port: int = 37000,
    max_instances: int = 4,
    result_filename: str = ""
):
    command = (
        f"RAY_DEDUP_LOGS=0 "
        f"nohup python -u -m llumnix.entrypoints.bladellm.serve "
        f"--load-registered-service "
        f"--load-registered-service-path ./service_test/bladellm "
        f"--model {model} " # must set, checked when parse args
        f"--host {ip} " # must set, for server
        f"--port {port} " # must set, for server
        f"--enable-engine-pd-disagg "
        f"--pd-ratio 1:1 "
        f"--max-instances {max_instances} "
        f"--enable-port-increment "
        f"{'> instance_'+result_filename if len(result_filename) > 0 else ''} 2>&1 &"
    )
    return command

def shutdown_llumnix_service_func():
    subprocess.run('pkill -f llumnix.entrypoints.vllm.api_server', shell=True, check=False)
    subprocess.run('pkill -f llumnix.entrypoints.vllm_v1.api_server', shell=True, check=False)
    subprocess.run('pkill -f benchmark_serving.py', shell=True, check=False)
    subprocess.run('pkill -f llumnix.entrypoints.vllm.serve', shell=True, check=False)
    subprocess.run('pkill -f llumnix.entrypoints.vllm_v1.serve', shell=True, check=False)
    subprocess.run('pkill -f blade_llm_server', shell=True, check=False)
    subprocess.run('pkill -f llumnix.entrypoints.bladellm.serve', shell=True, check=False)
    subprocess.run('pkill -f multiprocessing', shell=True, check=False)
    subprocess.run("pkill -f 'vllm serve'", shell=True, check=False)
    subprocess.run('rm -rf /tmp/kvt-*', shell=True, check=False)
    subprocess.run(f'rm -rf {NAMING_URL.split(":")[1] + "/*"}', shell=True, check=False)
    time.sleep(1.0)

def cleanup_ci_outputs_func():
    subprocess.run('rm -rf bench_*.out', shell=True, check=False)
    subprocess.run('rm -rf instance_*.out', shell=True, check=False)
    subprocess.run('rm -rf nohup.out', shell=True, check=False)
    subprocess.run('rm -rf core.*', shell=True, check=False)
    subprocess.run('rm -rf nfs*', shell=True, check=False)
    subprocess.run('rm -rf service_test', shell=True, check=False)
    subprocess.run('rm -rf *.png', shell=True, check=False)
    subprocess.run('rm -rf *.npy', shell=True, check=False)
    subprocess.run('rm -rf *.json', shell=True, check=False)
    subprocess.run('rm -rvf /tmp/vllm.naming/*', shell=True, check=False)
    subprocess.run('mkdir -pv /tmp/vllm.naming/', shell=True, check=False)

@pytest.fixture
def shutdown_llumnix_service():
    try:
        cleanup_ci_outputs_func()
        yield
    finally:
        if conftest.SKIP_REASON is None or len(conftest.SKIP_REASON) == 0:
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

@pytest.fixture
def check_log_exception():
    yield
    if conftest.SKIP_REASON is not None and len(conftest.SKIP_REASON) > 0:
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

def generate_special_test_config(key_value_pairs: List[Tuple[str, str]], base_config: List, schema: str):
    new_config = copy.deepcopy(base_config)
    for key, value in key_value_pairs:
        index = schema.replace(" ", "").split(",").index(key)
        new_config[index] = value
    return new_config
