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
from tests.conftest import ray_env
from .utils import (generate_launch_command, wait_for_llumnix_service_ready,
                    shutdown_llumnix_service)


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
@pytest.mark.parametrize("migration_backend", ['rayrpc', 'gloo'])
async def test_e2e(ray_env, shutdown_llumnix_service, model, migration_backend):
    max_model_len = 370
    sampling_params = {
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": False,
    }

    global vllm_output

    if len(vllm_output) == 0:
        vllm_output = ray.get(run_vllm.remote(model, max_model_len, sampling_params))

    ray.shutdown()

    await asyncio.sleep(5)

    # generate llumnix outputs
    ip = "127.0.0.1"
    base_port = 37037
    launch_command = generate_launch_command(model=model,
                                             max_model_len=max_model_len,
                                             ip=ip,
                                             port=base_port,
                                             migration_backend=migration_backend)
    subprocess.run(launch_command, shell=True, check=True)

    wait_for_llumnix_service_ready(ip_ports=[f"{ip}:{base_port}"])

    llumnix_output = {}
    for prompt in prompts:
        response = await asyncio.wait_for(get_llumnix_response(prompt, sampling_params, f"{ip}:{base_port}"),
                                          timeout=60*5)
        llumnix_output[prompt] = response['text'][0]

    # compare
    for prompt in prompts:
        assert llumnix_output[prompt] == vllm_output[prompt]
