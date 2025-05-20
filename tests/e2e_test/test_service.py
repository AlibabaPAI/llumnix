import subprocess
import asyncio

import pytest
import torch
import ray

from llumnix.utils import get_ip_address, wait_port_free
from llumnix.ray_utils import get_actor_names_by_name_prefix, INSTANCE_NAME_PREFIX

# pylint: disable=unused-import
from tests.conftest import ray_env, cleanup_ray_env_func
from tests.e2e_test.utils import (check_log_exception, shutdown_llumnix_service,
                                  generate_vllm_register_service_command_func,
                                  generate_bladellm_register_service_command_func,
                                  generate_vllm_serve_service_command_func,
                                  generate_bladellm_serve_service_command_func,
                                  wait_for_llumnix_service_ready)
from tests.utils import try_convert_to_local_path


def check_pd_instance_count():
    curr_instance_names = get_actor_names_by_name_prefix(name_prefix=INSTANCE_NAME_PREFIX)
    p_instance_count = 0
    d_instance_count = 0
    for curr_instance_name in curr_instance_names:
        instance_actor_handle = ray.get_actor(curr_instance_name, namespace="llumnix")
        instance_type = ray.get(instance_actor_handle.get_instance_type.remote())
        if instance_type == "prefill":
            p_instance_count += 1
        elif instance_type == "decode":
            d_instance_count += 1
    assert p_instance_count == 2 and d_instance_count == 2, \
        "The service serve command is supposed to launch 2 prefill instances and 2 decode instances."


test_times = 0

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for correctness test")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen2.5-7B')])
@pytest.mark.parametrize("engine", ["engine_vLLM", "engine_BladeLLM"])
async def test_service(ray_env, shutdown_llumnix_service, check_log_exception, model, engine):
    engine = engine.split("_")[1]

    global test_times

    ip = get_ip_address()
    base_port = 60000 + test_times * 100
    if "BladeLLM" in engine:
        base_port += 5000
    device_count = min(4, torch.cuda.device_count())
    instance_count = device_count

    if engine == "vLLM":
        generate_register_service_command_func = generate_vllm_register_service_command_func
        genertate_serve_service_command_func = generate_vllm_serve_service_command_func
    else:
        generate_register_service_command_func = generate_bladellm_register_service_command_func
        genertate_serve_service_command_func = generate_bladellm_serve_service_command_func

    ip_ports = []
    for i in range(instance_count):
        wait_port_free(base_port + i)
        ip_ports.append(f"{ip}:{base_port + i}")

    subprocess.run(generate_register_service_command_func(
        model=model, ip=ip, port=base_port, engine_type="prefill"), shell=True, check=True)
    subprocess.run(generate_register_service_command_func(
        model=model, ip=ip, port=base_port, engine_type="decode"), shell=True, check=True)
    subprocess.run(
        genertate_serve_service_command_func(
            model=model, ip=ip, port=base_port, max_instances=instance_count, result_filename=str(base_port)+".out"
        ),
        shell=True, check=True
    )

    wait_for_llumnix_service_ready(ip_ports)

    await asyncio.sleep(3)

    check_pd_instance_count()

    test_times += 1
