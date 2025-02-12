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
import time

import pytest
import ray
import aiohttp
import torch

# pylint: disable=unused-import
from tests.conftest import ray_env
from tests.e2e_test.utils import (generate_vllm_launch_command, generate_vllm_serve_command, 
                    wait_for_llumnix_service_ready, generate_bladellm_launch_command,
                    shutdown_llumnix_service, shutdown_llumnix_service_func, generate_bladellm_request,
                    generate_vllm_request, process_bladellm_api_server_output, process_vllm_api_server_output)


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

@ray.remote(num_gpus=1)
def run_vllm(model):
    # pylint: disable=import-outside-toplevel
    from vllm import LLM, SamplingParams
    sampling_params = {
        "n": 1,
        "best_of": 1,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": False,
    }
    raw_vllm = LLM(model=model, trust_remote_code=True, max_model_len=1024)
    outputs = raw_vllm.generate(prompts, SamplingParams(**sampling_params), use_tqdm=False)
    vllm_output = {output.prompt: output.prompt + output.outputs[0].text for output in outputs}
    return vllm_output

async def run_bladellm(model):
    ip = "127.0.0.1"
    port = 50091
    launch_command = generate_bladellm_launch_command(
        model=model,
        ip=ip,
        port=port,
        enable_llumnix=False
    )
    subprocess.run(launch_command, shell=True, check=True)
    time.sleep(30)

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
    return bladellm_outputs

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="at least 2 gpus required for correctness test")
@pytest.mark.parametrize("model", ['/mnt/model/Qwen-7B'])
@pytest.mark.parametrize("launch_mode", ['global', 'local'])
@pytest.mark.parametrize("enable_pd_disagg", [True, False])
@pytest.mark.parametrize("engine", ["engine_vLLM", "engine_bladeLLM"])
async def test_correctness(ray_env, shutdown_llumnix_service, model, launch_mode, enable_pd_disagg, engine):
    engine = engine.split("_")[1]
    if engine == "bladeLLM" and launch_mode == "global":
        pytest.skip("Global launch model for bladeLLM is not supported yet.")
    if engine == "bladeLLM" and enable_pd_disagg:
        pytest.skip("PDD based on llumnix for bladeLLM is not supported yet.")

    ip = "127.0.0.1"
    base_port = 37037

    global engine_prompt_output

    if engine == "vLLM":
        generate_request_func = generate_vllm_request
        process_api_server_output_func = process_vllm_api_server_output
        generate_launch_command_func = generate_vllm_launch_command
        url = f'http://{ip}:{base_port}/generate'

        if len(engine_prompt_output) == 0:
            engine_prompt_output = await run_vllm.remote(model)
    else:
        generate_request_func = generate_bladellm_request
        process_api_server_output_func = process_bladellm_api_server_output
        generate_launch_command_func = generate_bladellm_launch_command
        url = f'http://{ip}:{base_port}/v1/chat/completions'

        if len(engine_prompt_output) == 0:
            engine_prompt_output = await run_bladellm(model)

    launch_commands = []
    if launch_mode == "local":
        if enable_pd_disagg:
            launch_commands.append(generate_vllm_launch_command(result_filename=str(base_port)+".out",
                                                    model=model,
                                                    port=base_port,
                                                    enable_pd_disagg=enable_pd_disagg,
                                                    instance_type="prefill"))
            launch_commands.append(generate_vllm_launch_command(result_filename=str(base_port+1)+".out",
                                                    launch_ray_cluster=False,
                                                    model=model,
                                                    ip=ip,
                                                    port=base_port+1,
                                                    enable_pd_disagg=enable_pd_disagg,
                                                    instance_type="decode"))
        else:
            launch_commands.append(generate_launch_command_func(model=model,
                                                    ip=ip,
                                                    port=base_port))
    else:
        launch_commands.append(generate_vllm_serve_command(result_filename=str(base_port)+".out",
                                               ip=ip,
                                               port=base_port,
                                               model=model,
                                               enable_pd_disagg=enable_pd_disagg))
    for launch_command in launch_commands:
        subprocess.run(launch_command, shell=True, check=True)
        await asyncio.sleep(3)

    wait_for_llumnix_service_ready(ip_ports=[f"{ip}:{base_port}"], timeout=120)

    llumnix_output = {}
    for prompt in prompts:
        response = await asyncio.wait_for(
            get_llumnix_response(prompt, url, generate_request_func, process_api_server_output_func), 
            timeout=60*5)
        llumnix_output[prompt] = response

    # compare
    for prompt in prompts:
        assert llumnix_output[prompt] == engine_prompt_output[prompt]
