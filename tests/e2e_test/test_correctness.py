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
from typing import List

import ray
import pytest
import aiohttp
import torch

from llumnix.utils import get_ip_address, try_convert_to_local_path

# pylint: disable=unused-import
from tests.conftest import ray_env, cleanup_ray_env_func
from tests.e2e_test.utils import (generate_vllm_launch_command, generate_vllm_serve_command,
                    wait_for_llumnix_service_ready, generate_bladellm_launch_command,
                    shutdown_llumnix_service, shutdown_llumnix_service_func, generate_bladellm_request,
                    generate_vllm_request, process_bladellm_api_server_output, process_vllm_api_server_output,
                    check_log_exception, generate_bladellm_serve_command)


async def get_llumnix_response(prompt, url, generate_request_func, process_api_server_output_func):
    timeout = aiohttp.ClientTimeout(total=60)
    request = generate_request_func(prompt)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=request) as resp:
            output = await resp.json()
            return process_api_server_output_func(output)

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

async def run_bladellm(model, enable_pd_disagg):
    ip = get_ip_address()
    port = 50000 + test_times * 100

    if not enable_pd_disagg:
        launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=port,
            enable_llumnix=False
        )
        subprocess.run(launch_command, shell=True, check=True)
    else:
        prefill_launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=port,
            enable_llumnix=False,
            enable_pd_disagg=True,
            instance_type="prefill",
        )
        subprocess.run(prefill_launch_command, shell=True, check=True)
        decode_launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=port+100,
            enable_llumnix=False,
            enable_pd_disagg=True,
            instance_type="decode",
            cuda_visiable_device="1"
        )
        subprocess.run(decode_launch_command, shell=True, check=True)

    await asyncio.sleep(60)

    bladellm_outputs = {}
    for prompt in prompts:
        req_out = await get_llumnix_response(
            prompt,
            f"http://{ip}:{port}/v1/chat/completions",
            generate_bladellm_request,
            process_bladellm_api_server_output,
        )
        bladellm_outputs[prompt] = req_out

    shutdown_llumnix_service_func()
    await asyncio.sleep(3)
    return bladellm_outputs

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="at least 2 gpus required for correctness test")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen-7B')])
@pytest.mark.parametrize("launch_mode", ['global', 'local'])
@pytest.mark.parametrize("enable_pd_disagg", [False, True])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("engine", ["engine_vLLM", "engine_BladeLLM"])
async def test_correctness(ray_env, shutdown_llumnix_service,
                           model, launch_mode, enable_pd_disagg, tensor_parallel_size, engine):
    engine = engine.split("_")[1]

    # TODO(chenghao): fix this bug
    if "BladeLLM" in engine and launch_mode == "global" and enable_pd_disagg:
        pytest.skip("Error in BladeLLM for prefill-decode disaggregation in global launch mode.")

    # TODO(s5u13b): fix this bug
    if "BladeLLM" in engine and tensor_parallel_size > 1:
        pytest.skip("Error in BladeLLM for tensor parallel size > 1.")

    if tensor_parallel_size == 2 and launch_mode == "local":
        pytest.skip("Only test tensor parallelism in global launch mode.")

    global test_times

    ip = get_ip_address()
    base_port = 40000 + test_times * 100
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
        enable_migration = True

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
        enable_migration = not enable_pd_disagg

        if not enable_pd_disagg and len(engine_prompt_output) == 0:
            engine_prompt_output = await run_bladellm(model, enable_pd_disagg)

        if enable_pd_disagg and len(engine_pdd_prompt_output) == 0:
            engine_pdd_prompt_output = await run_bladellm(model, enable_pd_disagg)

    launch_commands = []
    if launch_mode == "local":
        if enable_pd_disagg:
            launch_commands.append(generate_launch_command_func(result_filename=str(base_port)+".out",
                                                    model=model,
                                                    ip=ip,
                                                    port=base_port,
                                                    enable_pd_disagg=enable_pd_disagg,
                                                    instance_type="prefill",
                                                    enable_migration=enable_migration,
                                                    tensor_parallel_size=tensor_parallel_size))
            decode_port = base_port + 100
            launch_commands.append(generate_launch_command_func(result_filename=str(decode_port)+".out",
                                                    launch_ray_cluster=False,
                                                    model=model,
                                                    ip=ip,
                                                    port=decode_port,
                                                    enable_pd_disagg=enable_pd_disagg,
                                                    instance_type="decode",
                                                    enable_migration=enable_migration,
                                                    tensor_parallel_size=tensor_parallel_size))
        else:
            launch_commands.append(generate_launch_command_func(result_filename=str(base_port)+".out",
                                                    model=model,
                                                    ip=ip,
                                                    port=base_port,
                                                    tensor_parallel_size=tensor_parallel_size))
    else:
        launch_commands.append(generate_serve_command_func(result_filename=str(base_port)+".out",
                                               ip=ip,
                                               port=base_port,
                                               model=model,
                                               enable_pd_disagg=enable_pd_disagg,
                                               tensor_parallel_size=tensor_parallel_size,
                                               max_instances=instance_count))
    for launch_command in launch_commands:
        subprocess.run(launch_command, shell=True, check=True)
    await asyncio.sleep(3)

    wait_for_llumnix_service_ready(ip_ports=[f"{ip}:{base_port}"])

    llumnix_output = {}
    for prompt in prompts:
        response = await get_llumnix_response(prompt, url, generate_request_func, process_api_server_output_func)
        llumnix_output[prompt] = response

    # compare
    raw_output = engine_prompt_output if not enable_pd_disagg else engine_pdd_prompt_output
    for prompt in prompts:
        assert llumnix_output[prompt] == raw_output[prompt]

    await asyncio.sleep(3)

    check_log_exception()

    test_times += 1
