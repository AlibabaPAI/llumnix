import asyncio
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import pytest

from llumnix.utils import get_ip_address, wait_port_free

# pylint: disable=unused-import
from tests.conftest import ray_env
from tests.e2e_test.utils import (generate_vllm_launch_command, generate_bench_command,
                    wait_for_llumnix_service_ready, shutdown_llumnix_service,
                    generate_vllm_serve_command, check_log_exception)
from tests.utils import try_convert_to_local_path

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for simple benchmark")
@pytest.mark.parametrize("model", [try_convert_to_local_path('Qwen/Qwen-7B')])
@pytest.mark.parametrize("engine", ["engine_vLLM"])
async def test_dynamicpd(request, ray_env, shutdown_llumnix_service, check_log_exception, model, engine):
    engine = engine.split("_")[1]
    ip = get_ip_address()
    base_port = 15000 + random.randint(0, 96)
    num_prompts = 500

    ip_ports = []
    device_count = min(4, torch.cuda.device_count())
    num_instances = device_count

    generate_serve_command = generate_vllm_serve_command

    for i in range(device_count):
        port = base_port + i
        wait_port_free(port, force=True)
        ip_port = f"{ip}:{port}"
        ip_ports.append(ip_port)

    serve_command = generate_serve_command(result_filename=str(base_port)+".out",
                                            ip=ip,
                                            port=base_port,
                                            model=model,
                                            request_output_queue_type="zmq",
                                            enable_pd_disagg=True,
                                            enable_dynamic_pd_disagg=True,
                                            pd_ratio="1:3",
                                            enforce_eager=True,
                                            max_instances=num_instances)
    subprocess.run(serve_command, shell=True, check=True)
    wait_for_llumnix_service_ready(ip_ports)

    def run_bench_command(command):
        # pylint: disable=consider-using-with
        process = subprocess.Popen(command, shell=True)
        return process

    tasks = []
    for ip_port in ip_ports:
        port = ip_port.split(":")[1]
        bench_command = generate_bench_command(
            backend=engine,
            ip_ports=f"{ip}:{port}",
            model=model,
            num_prompts=num_prompts,
            dataset_type="sharegpt",
            dataset_path="/mnt/dataset/sharegpt_gpt4/sharegpt_gpt4.jsonl",
            qps=5,
            results_filename=f"{port}.out"
        )
        tasks.append(bench_command)

    with ThreadPoolExecutor() as executor:
        future_to_command = {executor.submit(run_bench_command, command): command for command in tasks}

        for future in as_completed(future_to_command):
            process = future.result()
            process.wait()
            assert process.returncode == 0, "dynamicpd_test failed with return code {}.".format(process.returncode)

    await asyncio.sleep(5)
