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

import pytest
import torch
import ray

from llumnix.utils import get_ip_address, wait_port_free
from llumnix.ray_utils import list_actor_names_by_actor_type, LlumnixActor

# pylint: disable=unused-import
from tests.conftest import ray_env, cleanup_ray_env_func
from tests.e2e_test.utils import (
    check_log_exception,
    shutdown_llumnix_service,
    generate_vllm_register_service_command_func,
    generate_vllm_v1_register_service_command_func,
    generate_bladellm_register_service_command_func,
    generate_vllm_serve_service_command_func,
    generate_vllm_v1_serve_service_command_func,
    generate_bladellm_serve_service_command_func,
    wait_for_llumnix_service_ready,
    generate_vllm_request,
    process_vllm_api_server_output,
    generate_vllm_v1_request,
    process_vllm_v1_api_server_output,
    generate_bladellm_request,
    process_bladellm_api_server_output,
    get_llumnix_response,
    wait_for_llumnix_service_ready_vllm_v1,
)
from tests.utils import try_convert_to_local_path


def check_pd_instance_count():
    curr_instance_names = list_actor_names_by_actor_type(LlumnixActor.INSTANCE)
    p_instance_count = 0
    d_instance_count = 0
    for curr_instance_name in curr_instance_names:
        instance_actor_handle = ray.get_actor(curr_instance_name, namespace="llumnix")
        instance_type = ray.get(instance_actor_handle.get_instance_type.remote())
        if instance_type == "prefill":
            p_instance_count += 1
        elif instance_type == "decode":
            d_instance_count += 1
    assert p_instance_count == 1 and d_instance_count == 1, \
        "The service serve command is supposed to launch 2 prefill instances and 2 decode instances."
    print("Check pd instance count passed.")


test_times = 0


@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for correctness test")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen2.5-7B')])
@pytest.mark.parametrize("engine", ["engine_vLLM", "engine_BladeLLM", "engine_vLLM_v1"])
async def test_service(ray_env, shutdown_llumnix_service, check_log_exception, model, engine):
    engine = "_".join(engine.split("_")[1:])

    if engine == "vLLM_v1":
        model = '/mnt/data/models/Qwen2.5-1.5B-Instruct'

    global test_times

    ip = get_ip_address()
    base_port = 40000 + random.randint(0, 46) + test_times * 100
    if "BladeLLM" in engine:
        base_port += 1250
    elif "vLLM_v1" in engine:
        base_port += 2500
    device_count = min(2, torch.cuda.device_count())
    instance_count = device_count

    if engine == "vLLM":
        generate_register_service_command_func = generate_vllm_register_service_command_func
        genertate_serve_service_command_func = generate_vllm_serve_service_command_func
        generate_request_func = generate_vllm_request
        process_api_server_output_func = process_vllm_api_server_output
        url = f'http://{ip}:{base_port}/generate'
    elif engine == "vLLM_v1":
        generate_register_service_command_func = generate_vllm_v1_register_service_command_func
        genertate_serve_service_command_func = generate_vllm_v1_serve_service_command_func
        generate_request_func = generate_vllm_v1_request
        process_api_server_output_func = process_vllm_v1_api_server_output
        url = f'http://{ip}:{base_port}/v1/chat/completions'
    else:
        generate_register_service_command_func = generate_bladellm_register_service_command_func
        genertate_serve_service_command_func = generate_bladellm_serve_service_command_func
        generate_request_func = generate_bladellm_request
        process_api_server_output_func = process_bladellm_api_server_output
        url = f'http://{ip}:{base_port}/v1/chat/completions'

    ip_ports = []
    for i in range(instance_count):
        wait_port_free(base_port + i)
        ip_ports.append(f"{ip}:{base_port + i}")

    commands = []
    commands.append(
        generate_register_service_command_func(
            model=model, ip=ip, port=base_port, engine_type="prefill"
        )
    )
    commands.append(
        generate_register_service_command_func(
            model=model, ip=ip, port=base_port, engine_type="decode"
        )
    )
    commands.append(
        genertate_serve_service_command_func(
            model=model, ip=ip, port=base_port, max_units=instance_count,
            result_filename=str(base_port)+".out"
        )
    )
    for command in commands:
        print(f"Going to run command: {command}")
        subprocess.run(command, shell=True, check=True)

    if engine == "vLLM_v1":
        wait_for_llumnix_service_ready_vllm_v1(ip_ports)
    else:
        wait_for_llumnix_service_ready(ip_ports)

    await asyncio.sleep(3)

    check_pd_instance_count()

    _ = await get_llumnix_response("Hello, my name is", url, generate_request_func, process_api_server_output_func)

    await asyncio.sleep(3)

    test_times += 1
