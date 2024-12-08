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
import asyncio
import pytest
import aiohttp
import ray
import torch

from vllm import LLM, SamplingParams
# pylint: disable=unused-import
from .utils import setup_ray_env

def parse_launch_mode(launch_mode: str):
    # 'eief' means that enable init instance by manager and enable fixed node init instance, and so on.
    if launch_mode == 'eief':
        disable_init_instance_by_manager = False
        disable_fixed_node_init_instance = False
    elif launch_mode == 'eidf':
        disable_init_instance_by_manager = False
        disable_fixed_node_init_instance = True
    elif launch_mode == 'dief':
        disable_init_instance_by_manager = True
        disable_fixed_node_init_instance = False
    else:
        disable_init_instance_by_manager = True
        disable_fixed_node_init_instance = True
    return disable_init_instance_by_manager, disable_fixed_node_init_instance

def generate_launch_command(result_filename: str = "", launch_ray_cluster: bool = True, HEAD_NODE_IP: str = "127.0.0.1",
                            ip: str = "127.0.0.1", port: int = 37000, instances_num = 1, dispatch_policy: str = "load",
                            migration_backend = "gloo", model = "facebook/opt-125m", max_model_len: int = 2048,
                            launch_mode: str = 'eief', log_instance_info: bool = False,
                            request_migration_policy: str = 'SR'):
    disable_init_instance_by_manager, disable_fixed_node_init_instance = parse_launch_mode(launch_mode)
    command = (
        f"RAY_DEDUP_LOGS=0 HEAD_NODE_IP={HEAD_NODE_IP} HEAD_NODE=1 "
        f"nohup python -u -m llumnix.entrypoints.vllm.api_server "
        f"--host {ip} "
        f"--port {port} "
        f"{'--disable-init-instance-by-manager ' if disable_init_instance_by_manager else ''}"
        f"{'--disable-fixed-node-init-instance ' if disable_fixed_node_init_instance else ''}"
        f"--initial-instances {instances_num} "
        f"{'--log-filename manager ' if log_instance_info else ''}"
        f"{'--log-instance-info ' if log_instance_info else ''}"
        f"--enable-migration "
        f"--model {model} "
        f"--engine-use-ray "
        f"--worker-use-ray "
        f"--max-model-len {max_model_len} "
        f"--dispatch-policy {dispatch_policy} "
        f"--trust-remote-code "
        f"--request-migration-policy {request_migration_policy} "
        f"--migration-backend {migration_backend} "
        f"--migration-buffer-blocks 32 "
        f"--migration-internal-buffer-num 2 "
        f"--tensor-parallel-size 1 "
        f"--request-output-queue-port {1234+port} "
        f"{'--launch-ray-cluster ' if launch_ray_cluster else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

def launch_llumnix_service(model: str, max_model_len: int, port: int, migration_backend: str, launch_mode: str):
    command = generate_launch_command(model=model, max_model_len=max_model_len,
                                      port=port, migration_backend=migration_backend,
                                      launch_mode=launch_mode)
    subprocess.run(command, shell=True, check=True)

def shutdown_llumnix_service():
    try:
        subprocess.run('pkill -f llumnix.entrypoints.vllm.api_server', shell=True, check=True)
    # pylint: disable=broad-except
    except Exception:
        pass

def clear_ray_state():
    named_actors = ray.util.list_named_actors(True)
    for actor in named_actors:
        try:
            actor_handle = ray.get_actor(actor['name'], namespace=actor['namespace'])
        # pylint: disable=bare-except
        except:
            continue

        try:
            ray.kill(actor_handle)
        # pylint: disable=bare-except
        except:
            continue
    ray.shutdown()

async def get_llumnix_response(prompt, sampling_params, ip_ports):
    timeout = aiohttp.ClientTimeout(total=60)

    request = {
        "prompt": prompt,
        "stream": False,
        **sampling_params,
    }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f'http://{ip_ports}/generate', json=request) as resp:
            output = await resp.json()
            return output

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

vllm_output = {}

@ray.remote(num_gpus=1)
def run_vllm(model, max_model_len, sampling_params):
    vllm_output = {}
    raw_vllm = LLM(model=model, trust_remote_code=True, max_model_len=max_model_len)
    outputs = raw_vllm.generate(prompts, SamplingParams(**sampling_params))

    for output in outputs:
        vllm_output[output.prompt] = output.prompt + output.outputs[0].text

    return vllm_output

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required for e2e test")
@pytest.mark.parametrize("model", ['/mnt/model/Qwen-7B'])
@pytest.mark.parametrize("migration_backend", ['rpc', 'gloo'])
@pytest.mark.parametrize("launch_mode", ['eief', 'eidf', 'dief', 'didf'])
async def test_e2e(setup_ray_env, model, migration_backend, launch_mode):
    if migration_backend == 'gloo' and launch_mode != 'eief':
        pytest.skip("When the migration backend is gloo, the launch mode of llumnix can only be eief")
    max_model_len = 370
    sampling_params = {
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": False,
    }

    # generate llumnix outputs
    base_port = 37037
    launch_llumnix_service(model, max_model_len, base_port, migration_backend, launch_mode)
    await asyncio.sleep(60)

    llumnix_output = {}
    for prompt in prompts:
        response = await asyncio.wait_for(get_llumnix_response(prompt, sampling_params, f"127.0.0.1:{base_port}"),
                                          timeout=60*5)
        llumnix_output[prompt] = response['text'][0]

    shutdown_llumnix_service()

    global vllm_output

    if len(vllm_output) == 0:
        vllm_output = ray.get(run_vllm.remote(model, max_model_len, sampling_params))

    clear_ray_state()
    # compare
    for prompt in prompts:
        assert llumnix_output[prompt] == vllm_output[prompt]
