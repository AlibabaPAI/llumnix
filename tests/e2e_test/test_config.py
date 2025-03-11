import subprocess
import pytest
import torch

from llumnix.entrypoints.utils import get_ip_address

# pylint: disable=unused-import
from tests.conftest import ray_env
from .utils import shutdown_llumnix_service, generate_config_command, wait_for_llumnix_service_ready

# TODO(KuilongCui): Support bladellm config test.
@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="at least 1 gpus required for correctness test")
@pytest.mark.parametrize("launch_mode", ['global', 'local'])
@pytest.mark.parametrize("backend", ['vllm'])
async def test_config(ray_env, shutdown_llumnix_service, launch_mode, backend):
    config_path = './configs/{}.yml'.format(backend)
    ip = get_ip_address()
    port = 37037
    ip_ports = []
    ip_port = f"{ip}:{port}"
    ip_ports.append(ip_port)
    config_command = generate_config_command(config_path=config_path,
                                             ip=ip,
                                             port=port,
                                             model='facebook/opt-125m',
                                             launch_mode=launch_mode,
                                             result_filename=str(port)+".out")
    subprocess.run(config_command, shell=True, check=True)

    wait_for_llumnix_service_ready(ip_ports)
