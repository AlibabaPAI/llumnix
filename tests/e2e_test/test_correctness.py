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


import random
import subprocess
import asyncio
from typing import List

import ray
import pytest
import torch

from llumnix.utils import get_ip_address, wait_port_free

# pylint: disable=unused-import
from tests import conftest
from tests.conftest import ray_env, cleanup_ray_env_func
from tests.e2e_test.utils import (generate_vllm_launch_command, generate_vllm_serve_command,
                                  wait_for_llumnix_service_ready, generate_bladellm_launch_command,
                                  shutdown_llumnix_service, shutdown_llumnix_service_func, generate_bladellm_request,
                                  generate_vllm_request, process_bladellm_api_server_output, process_vllm_api_server_output,
                                  check_log_exception, generate_bladellm_serve_command, get_llumnix_response)
from tests.utils import try_convert_to_local_path


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

engine_prompt_output = {}
engine_pdd_prompt_output = {}

test_times = 0

@ray.remote(num_gpus=1)
def run_vllm(model):
    # pylint: disable=import-outside-toplevel
    from vllm import LLM, SamplingParams, RequestOutput
    sampling_params = {
        "n": 1,
        "best_of": 1,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": False,
    }
    raw_vllm = LLM(model=model, trust_remote_code=True, max_model_len=1024)
    outputs: List[RequestOutput] = raw_vllm.generate(prompts, SamplingParams(**sampling_params), use_tqdm=False)
    vllm_output = {}
    for _, output in enumerate(outputs):
        vllm_output[output.prompt] = output.prompt + output.outputs[0].text
    return vllm_output

async def run_bladellm(model, enable_pd_disagg, enable_migration):
    ip = get_ip_address()
    base_port = 35000 + test_times * 100

    if not enable_pd_disagg:
        launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=base_port,
            enable_llumnix=False,
            enable_migration=enable_migration
        )
        subprocess.run(launch_command, shell=True, check=True)
    else:
        prefill_launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=base_port,
            enable_llumnix=False,
            enable_pd_disagg=True,
            instance_type="prefill",
            enable_migration=enable_migration
        )
        subprocess.run(prefill_launch_command, shell=True, check=True)
        decode_launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=base_port+100,
            enable_llumnix=False,
            enable_pd_disagg=True,
            instance_type="decode",
            enable_migration=enable_migration,
            cuda_visiable_device="1"
        )
        subprocess.run(decode_launch_command, shell=True, check=True)

    await asyncio.sleep(60)

    bladellm_outputs = {}
    for prompt in prompts:
        req_out = await get_llumnix_response(
            prompt,
            f"http://{ip}:{base_port}/v1/chat/completions",
            generate_bladellm_request,
            process_bladellm_api_server_output,
        )
        bladellm_outputs[prompt] = req_out

    shutdown_llumnix_service_func()
    await asyncio.sleep(3)

    return bladellm_outputs

config_schema = "engine, migration_backend, tensor_parallel_size, enable_migration, enable_simulator," \
"enable_pd_disagg, launch_mode, request_output_forwarding_mode"

def generate_correctness_test_config():
    vllm_config = [
        ("engine_vLLM", "gloo", 1, True, False, False, "global", "thread"),

        # migration backend
        ("engine_vLLM", "gloo", 1, True, False, True, "global", "thread"),
        ("engine_vLLM", "rayrpc", 1, True, False, True, "global", "thread"),
        ("engine_vLLM", "nccl", 1, True, False, True, "global", "thread"),

        # migration tp=2
        ("engine_vLLM", "gloo", 2, True, False, True, "global", "thread"),
        ("engine_vLLM", "rayrpc", 2, True, False, True, "global", "thread"),

        # disable migration
        ("engine_vLLM", "gloo", 1, False, False, False, "global", "thread"),

        # simulation
        ("engine_vLLM", "gloo", 1, True, True, False, "global", "thread"),

        # local launch mode
        ("engine_vLLM", "gloo", 1, True, False, False, "local", "thread"),

        # actor token forward
        ("engine_vLLM", "gloo", 1, True, False, False, "global", "actor"),
    ]

    bladellm_config = [
        ("engine_BladeLLM", "grpc", 1, True, False, False, "global", "thread"),

        # tp=2
        ("engine_BladeLLM", "grpc", 2, True, False, False, "global", "thread"),

        # disable migration
        ("engine_BladeLLM", "grpc", 1, False, False, False, "global", "thread"),

        # engine pd
        ("engine_BladeLLM", "grpc", 1, True, False, True, "global", "thread"),

        # local launch mode
        ("engine_BladeLLM", "grpc", 1, True, False, False, "local", "thread"),

        # actor token forward
        ("engine_BladeLLM", "grpc", 1, True, False, False, "global", "actor"),
    ]

    return vllm_config + bladellm_config


@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for correctness test")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen2.5-7B')])
@pytest.mark.parametrize(config_schema, generate_correctness_test_config())
async def test_correctness(ray_env, shutdown_llumnix_service, check_log_exception, model,
                           engine, migration_backend, tensor_parallel_size, enable_migration, enable_simulator,
                           enable_pd_disagg, launch_mode, request_output_forwarding_mode):
    engine = engine.split("_")[1]

    global test_times

    ip = get_ip_address()
    base_port = 30000 + random.randint(0, 46) + test_times * 100
    if "BladeLLM" in engine:
        base_port += 2500
    device_count = min(4, torch.cuda.device_count())
    instance_count = device_count // tensor_parallel_size

    global engine_prompt_output
    global engine_pdd_prompt_output

    if engine == "vLLM":
        generate_request_func = generate_vllm_request
        process_api_server_output_func = process_vllm_api_server_output
        generate_launch_command_func = generate_vllm_launch_command
        generate_serve_command_func = generate_vllm_serve_command
        url = f'http://{ip}:{base_port}/generate'

        if not enable_pd_disagg and len(engine_prompt_output) == 0:
            engine_prompt_output = engine_pdd_prompt_output
            if len(engine_prompt_output) == 0:
                engine_prompt_output = await run_vllm.remote(model)

        if enable_pd_disagg and len(engine_pdd_prompt_output) == 0:
            engine_pdd_prompt_output = engine_prompt_output
            if len(engine_pdd_prompt_output) == 0:
                engine_pdd_prompt_output = await run_vllm.remote(model)
    else:
        generate_request_func = generate_bladellm_request
        process_api_server_output_func = process_bladellm_api_server_output
        generate_launch_command_func = generate_bladellm_launch_command
        generate_serve_command_func = generate_bladellm_serve_command
        url = f'http://{ip}:{base_port}/v1/chat/completions'

        if not enable_pd_disagg and len(engine_prompt_output) == 0:
            engine_prompt_output = await run_bladellm(model, enable_pd_disagg, enable_migration)

        if enable_pd_disagg and len(engine_pdd_prompt_output) == 0:
            engine_pdd_prompt_output = await run_bladellm(model, enable_pd_disagg, enable_migration)

    ip_ports = []

    launch_commands = []
    if launch_mode == "local":
        if enable_pd_disagg:
            prefill_port = base_port
            wait_port_free(prefill_port, force=True)
            ip_ports.append(f"{ip}:{prefill_port}")
            launch_commands.append(generate_launch_command_func(result_filename=str(prefill_port)+".out",
                                                    model=model,
                                                    ip=ip,
                                                    port=prefill_port,
                                                    enforce_eager=True,
                                                    migration_backend=migration_backend,
                                                    enable_pd_disagg=enable_pd_disagg,
                                                    enable_simulator=enable_simulator,
                                                    request_output_forwarding_mode=request_output_forwarding_mode,
                                                    instance_type="prefill",
                                                    enable_migration=enable_migration,
                                                    tensor_parallel_size=tensor_parallel_size))

            decode_port = base_port + 50
            wait_port_free(decode_port, force=True)
            ip_ports.append(f"{ip}:{decode_port}")
            launch_commands.append(generate_launch_command_func(result_filename=str(decode_port)+".out",
                                                    launch_ray_cluster=False,
                                                    model=model,
                                                    ip=ip,
                                                    port=decode_port,
                                                    migration_backend=migration_backend,
                                                    enforce_eager=True,
                                                    enable_simulator=enable_simulator,
                                                    request_output_forwarding_mode=request_output_forwarding_mode,
                                                    enable_pd_disagg=enable_pd_disagg,
                                                    instance_type="decode",
                                                    enable_migration=enable_migration,
                                                    tensor_parallel_size=tensor_parallel_size))
        else:
            wait_port_free(base_port, force=True)
            ip_ports.append(f"{ip}:{base_port}")
            launch_commands.append(generate_launch_command_func(result_filename=str(base_port)+".out",
                                                    model=model,
                                                    ip=ip,
                                                    port=base_port,
                                                    migration_backend=migration_backend,
                                                    enable_pd_disagg=enable_pd_disagg,
                                                    enforce_eager=True,
                                                    enable_simulator=enable_simulator,
                                                    request_output_forwarding_mode=request_output_forwarding_mode,
                                                    enable_migration=enable_migration,
                                                    tensor_parallel_size=tensor_parallel_size))
    else:
        for i in range(instance_count):
            wait_port_free(base_port + i, force=True)
            ip_ports.append(f"{ip}:{base_port + i}")
        launch_commands.append(generate_serve_command_func(result_filename=str(base_port)+".out",
                                               ip=ip,
                                               port=base_port,
                                               model=model,
                                               enforce_eager=True,
                                               migration_backend=migration_backend,
                                               enable_pd_disagg=enable_pd_disagg,
                                               enable_simulator=enable_simulator,
                                               request_output_forwarding_mode=request_output_forwarding_mode,
                                               tensor_parallel_size=tensor_parallel_size,
                                               enable_migration=enable_migration,
                                               max_instances=instance_count))
    for launch_command in launch_commands:
        subprocess.run(launch_command, shell=True, check=True)

    await asyncio.sleep(3)

    wait_for_llumnix_service_ready(ip_ports)

    await asyncio.sleep(3)

    llumnix_output = {}
    for prompt in prompts:
        response = await get_llumnix_response(prompt, url, generate_request_func, process_api_server_output_func)
        llumnix_output[prompt] = response

    # compare
    raw_output = engine_prompt_output if not enable_pd_disagg else engine_pdd_prompt_output
    if not enable_simulator:
        for prompt in prompts:
            assert llumnix_output[prompt] == raw_output[prompt]

    await asyncio.sleep(3)

    test_times += 1
