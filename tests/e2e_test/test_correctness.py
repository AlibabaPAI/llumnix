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


from functools import partial
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
                                  generate_vllm_request, process_vllm_api_server_output,
                                  generate_vllm_v1_launch_command, generate_vllm_v1_serve_command, generate_raw_vllm_v1_serve_command,
                                  generate_vllm_v1_request, process_vllm_v1_api_server_output,
                                  generate_bladellm_launch_command, generate_bladellm_serve_command,
                                  generate_bladellm_request, process_bladellm_api_server_output,
                                  wait_for_llumnix_service_ready, wait_for_llumnix_service_ready_vllm_v1,
                                  shutdown_llumnix_service, shutdown_llumnix_service_func, check_log_exception, get_llumnix_response,
                                  generate_special_test_config, parse_launch_command)
from tests.utils import try_convert_to_local_path


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

engine_prompt_output = {}
engine_pdd_prompt_output = {}
engine_semi_pd_prompt_output = {}

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

async def run_vllm_v1(model, tensor_parallel_size, enable_pd_disagg, enable_migration):
    ip = get_ip_address()
    base_port = 35000 + test_times * 100
    
    assert enable_pd_disagg is False, "PD disaggregation is not supported in v1"
    assert enable_migration is False, "Migration is not supported in v1"

    raw_serve_command = generate_raw_vllm_v1_serve_command(
        model=model,
        ip=ip,
        port=base_port,
        tensor_parallel_size=tensor_parallel_size,
    )
    subprocess.run(raw_serve_command, shell=True, check=True)

    await asyncio.sleep(60)

    vllm_v1_outputs = {}
    for prompt in prompts:
        req_out = await get_llumnix_response(
            prompt,
            f"http://{ip}:{base_port}/v1/chat/completions",
            generate_vllm_v1_request,
            process_vllm_v1_api_server_output,
        )
        vllm_v1_outputs[prompt] = req_out

    shutdown_llumnix_service_func()
    await asyncio.sleep(3)

    return vllm_v1_outputs

async def run_bladellm(model, enable_pd_disagg, enable_engine_semi_pd_disagg):
    global test_times
    ip = get_ip_address()
    base_port = 35000 + test_times * 100

    if not enable_pd_disagg and not enable_engine_semi_pd_disagg:
        launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=base_port,
            enable_llumnix=False
        )
        subprocess.run(launch_command, shell=True, check=True)
    else:
        prefill_launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=base_port,
            enable_llumnix=False,
            enable_pd_disagg=enable_pd_disagg,
            enable_engine_semi_pd_disagg=enable_engine_semi_pd_disagg,
            semi_pd_ins_id="prefill",
            instance_type="prefill",
            enforce_eager=True,
            cuda_visiable_device="0"
        )
        subprocess.run(prefill_launch_command, shell=True, check=True)
        decode_launch_command = generate_bladellm_launch_command(
            model=model,
            ip=ip,
            port=base_port+100,
            enable_llumnix=False,
            enable_pd_disagg=enable_pd_disagg,
            enable_engine_semi_pd_disagg=enable_engine_semi_pd_disagg,
            semi_pd_ins_id="decode",
            instance_type="decode",
            enforce_eager=True,
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

    test_times += 1

    return bladellm_outputs

config_schema = "engine, migration_backend, tensor_parallel_size, enable_migration, enable_simulator," \
"enable_pd_disagg, launch_mode, request_output_forwarding_mode, enable_engine_semi_pd_disagg, enable_adaptive_pd"

generate_special_correctness_test_config = partial(generate_special_test_config, schema=config_schema)


def generate_correctness_test_config():
    vllm_base_config = ["engine_vLLM", "gloo", 1, True, False, False, "global", "thread", False, False]

    vllm_config = [
        vllm_base_config,

        # migration backend and pd
        generate_special_correctness_test_config([("migration_backend", "gloo"), ("enable_pd_disagg", True)], vllm_base_config),
        generate_special_correctness_test_config([("migration_backend", "rayrpc"), ("enable_pd_disagg", True)], vllm_base_config),
        generate_special_correctness_test_config([("migration_backend", "nccl"), ("enable_pd_disagg", True)], vllm_base_config),

        # migration and tp=2
        generate_special_correctness_test_config(
            [("migration_backend", "gloo"), ("tensor_parallel_size", 2), ("enable_pd_disagg", True)],
            vllm_base_config),
        generate_special_correctness_test_config(
            [("migration_backend", "rayrpc"), ("tensor_parallel_size", 2), ("enable_pd_disagg", True)],
            vllm_base_config),

        # disable migration
        generate_special_correctness_test_config([("enable_migration", False)], vllm_base_config),

        # simulation
        generate_special_correctness_test_config([("enable_simulator", False)], vllm_base_config),

        # local launch mode
        generate_special_correctness_test_config([("launch_mode", "local")], vllm_base_config),
        generate_special_correctness_test_config([("launch_mode", "local"), ("enable_pd_disagg", True)], vllm_base_config),

        # actor token forward
        generate_special_correctness_test_config([("request_output_forwarding_mode", "actor")], vllm_base_config),

        # adaptive pd
        generate_special_correctness_test_config([("enable_pd_disagg", True), ("enable_adaptive_pd", True)], vllm_base_config),
    ]
    
    vllm_v1_base_config = ["engine_vLLM_v1", "gloo", 1, False, False, False, "global", "thread", False, False]
    
    vllm_v1_config = [
        vllm_v1_base_config,
        
        # tp=2
        generate_special_correctness_test_config([("tensor_parallel_size", 2)], vllm_v1_base_config),
    ]
    
    bladellm_base_config = ["engine_BladeLLM", "grpc", 1, True, False, False, "global", "thread", False, False]

    bladellm_config = [
        bladellm_base_config,

        # tp=2
        generate_special_correctness_test_config([("tensor_parallel_size", 2)], bladellm_base_config),

        # disable migration
        generate_special_correctness_test_config([("enable_migration", False)], bladellm_base_config),

        # engine pd
        generate_special_correctness_test_config([("enable_pd_disagg", True)], bladellm_base_config),

        # local launch mode
        generate_special_correctness_test_config([("launch_mode", "local")], bladellm_base_config),

        # actor token forward
        generate_special_correctness_test_config([("request_output_forwarding_mode", "actor")], bladellm_base_config),

        # semi pd
        generate_special_correctness_test_config([("enable_engine_semi_pd_disagg", True)], bladellm_base_config),

        # adaptive pd
        generate_special_correctness_test_config([("enable_engine_semi_pd_disagg", True), ("enable_adaptive_pd", True)], bladellm_base_config),
    ]

    return vllm_config + vllm_v1_config + bladellm_config


@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for correctness test")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen2.5-7B-Instruct')])
@pytest.mark.parametrize(config_schema, generate_correctness_test_config())
async def test_correctness(ray_env, shutdown_llumnix_service, check_log_exception, model,
                           engine, migration_backend, tensor_parallel_size, enable_migration, enable_simulator,
                           enable_pd_disagg, launch_mode, request_output_forwarding_mode, enable_engine_semi_pd_disagg,
                           enable_adaptive_pd):
    engine = "_".join(engine.split("_")[1:])

    global test_times

    ip = get_ip_address()
    base_port = 30000 + random.randint(0, 46) + test_times * 100
    if "BladeLLM" in engine:
        base_port += 2500
    device_count = min(4, torch.cuda.device_count())
    instance_count = device_count // tensor_parallel_size

    global engine_prompt_output
    global engine_pdd_prompt_output
    global engine_semi_pd_prompt_output

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
    elif engine == "vLLM_v1":
        assert not enable_pd_disagg, "PD disaggregation is not supported for vLLM_v1"
        
        generate_request_func = generate_vllm_v1_request
        process_api_server_output_func = process_vllm_v1_api_server_output
        generate_launch_command_func = generate_vllm_v1_launch_command
        generate_serve_command_func = generate_vllm_v1_serve_command
        url = f'http://{ip}:{base_port}/v1/chat/completions'
        
        engine_prompt_output = await run_vllm_v1(model, tensor_parallel_size, enable_pd_disagg, enable_migration)
    elif engine == "BladeLLM":
        generate_request_func = generate_bladellm_request
        process_api_server_output_func = process_bladellm_api_server_output
        generate_launch_command_func = generate_bladellm_launch_command
        generate_serve_command_func = generate_bladellm_serve_command
        url = f'http://{ip}:{base_port}/v1/chat/completions'

        if not enable_pd_disagg and not enable_engine_semi_pd_disagg and len(engine_prompt_output) == 0:
            engine_prompt_output = await run_bladellm(model, enable_pd_disagg, enable_engine_semi_pd_disagg)

        if enable_pd_disagg and len(engine_pdd_prompt_output) == 0:
            engine_pdd_prompt_output = await run_bladellm(model, enable_pd_disagg, enable_engine_semi_pd_disagg)

        if enable_engine_semi_pd_disagg and len(engine_semi_pd_prompt_output) == 0:
            engine_semi_pd_prompt_output = await run_bladellm(model, enable_pd_disagg, enable_engine_semi_pd_disagg)

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
                                                    enable_adaptive_pd=enable_adaptive_pd,
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
                                                    enable_adaptive_pd=enable_adaptive_pd,
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
                                                    enable_adaptive_pd=enable_adaptive_pd,
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
                                               enable_adaptive_pd=enable_adaptive_pd,
                                               enable_engine_semi_pd_disagg=enable_engine_semi_pd_disagg,
                                               enable_simulator=enable_simulator,
                                               request_output_forwarding_mode=request_output_forwarding_mode,
                                               tensor_parallel_size=tensor_parallel_size,
                                               enable_migration=enable_migration,
                                               max_instances=instance_count))
    for launch_command in launch_commands:
        subprocess.run(launch_command, shell=True, check=True)

    await asyncio.sleep(3)

    # TODO(zhaozhiyu): remove this special judge in the future
    if engine =="vLLM_v1":
        # special wait_for_llumnix_service_ready for vllm v1
        wait_for_llumnix_service_ready_vllm_v1(ip_ports)
    else:
        wait_for_llumnix_service_ready(ip_ports)

    await asyncio.sleep(3)

    llumnix_output = {}
    for prompt in prompts:
        response = await get_llumnix_response(prompt, url, generate_request_func, process_api_server_output_func)
        llumnix_output[prompt] = response

    # compare
    if not enable_pd_disagg and not engine_semi_pd_prompt_output:
        raw_output = engine_prompt_output
    elif enable_pd_disagg:
        raw_output = engine_pdd_prompt_output
    else:
        raw_output = engine_semi_pd_prompt_output

    if not enable_simulator:
        for prompt in prompts:
            assert llumnix_output[prompt] == raw_output[prompt]

    await asyncio.sleep(3)

    test_times += 1
