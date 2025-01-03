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
import os
import json
from websockets.sync.client import connect

from blade_llm import LLM, SamplingParams
from blade_llm.protocol import GenerateStreamResponse


BLADELLM_REPO_DIR= "/mnt/xinyi/"

def generate_launch_command(result_filename: str = "", HEAD_NODE_IP: str = "127.0.0.1",
                            ip: str = "127.0.0.1", port: int = 37000, model = "facebook/opt-125m"):
    command = (
        f"RAY_DEDUP_LOGS=0 HEAD_NODE_IP={HEAD_NODE_IP} HEAD_NODE=1 "
        f"nohup python {os.path.join(BLADELLM_REPO_DIR, 'LLMOps/blade_llm/service/server.py')} "
        f"--host {ip} "
        f"--port {ip} "
        f"--model {model} "
        f"--tensor_parallel_size 1 "
        f"--disable_prompt_cache "
        f"--log_level debug "
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

def launch_llumnix_service(model: str, port: int):
    command = generate_launch_command(model=model, port=port)
    print(command)
    subprocess.run(command, shell=True, check=True)

def shutdown_llumnix_service():
    try:
        subprocess.run('pkill -9 python'.format(BLADELLM_REPO_DIR), shell=True, check=True)
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

async def get_llumnix_response(prompt, sampling_params, stopping_criterial, ip_ports):
    timeout = 60

    request = {
        "prompt": "prompt",
        "sampling_params": sampling_params,
        "stopping_criterial": stopping_criterial,
    }

    url = "ws://{}/generate_stream".format(ip_ports)
    print(url)
    headers = {
        # "Authorization": "<You may need this header for EAS."
    }
    with connect(url, additional_headers=headers, open_timeout=timeout) as websocket:
        websocket.send(json.dumps(request))
        texts = []
        while True:
            await asyncio.sleep(0)
            msg = websocket.recv()
            resp = GenerateStreamResponse(**json.loads(msg))
            print(resp)
            texts.extend([t.text for t in resp.tokens])
            if resp.is_finished:
                break
        return texts


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

@ray.remote(num_gpus=1)
def run_bladellm(model, sampling_params):
    bladellm_output = {}
    raw_bladellm = LLM(model=model)
    outputs = raw_bladellm.submit_request(prompts, SamplingParams(**sampling_params))

    for output in outputs:
        bladellm_output[output.prompt] = output.prompt + output.outputs[0].text

    return bladellm_output

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required for e2e test")
@pytest.mark.parametrize("model", ['/mnt/dataset/Qwen--Qwen1.5-7B-tiny-random'])
async def test_e2e(model):
    max_new_tokens = 20
    sampling_params = {
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "top_k": 1,
        "max_new_tokens": 20,
    }

    stopping_criterial = {
        "max_new_tokens": max_new_tokens
    }
    # generate llumnix outputs
    base_port = 37037
    launch_llumnix_service(model, base_port)
    await asyncio.sleep(60)

    llumnix_output = {}
    for prompt in prompts:
        response = await asyncio.wait_for(get_llumnix_response(prompt, sampling_params, stopping_criterial, f"127.0.0.1:{base_port}"),
                                          timeout=60*5)
        llumnix_output[prompt] = response
        print(prompt, "  ", response)

    shutdown_llumnix_service()

    bladellm_output = ray.get(run_bladellm.remote(model, sampling_params))
    clear_ray_state()

    # compare
    for prompt in prompts:
        assert llumnix_output[prompt] == bladellm_output[prompt]
