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

import gc
import os
import time
import uuid
import asyncio
import threading
from typing import Callable, Awaitable, TypeVar, Coroutine, Dict, Optional, Union, Any
import socket
from functools import partial
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import psutil
from typing_extensions import ParamSpec
import ray

from llumnix.logging.logger import init_logger
from llumnix import envs as llumnix_envs
from llumnix.constants import RAY_RPC_TIMEOUT

logger = init_logger(__name__)


_MAX_PORT = 65536

P = ParamSpec('P')
T = TypeVar("T")

RequestIDType = Union[str, int]

@dataclass
class MigrationResponse:
    success: bool = True
    return_value: Any = None


logger = init_logger(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def convert_bytes(bytes_size):
    """Convert bytes to KB, MB, GB, etc."""
    if bytes_size < 0:
        raise ValueError("Size must be a non-negative integer.")

    size_suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    index = 0

    while bytes_size >= 1024 and index < len(size_suffixes) - 1:
        bytes_size /= 1024.0
        index += 1

    return f"{bytes_size:.2f} {size_suffixes[index]}"

def run_coroutine_in_new_thread(coro: Coroutine, blocking: bool):
    def run_coroutine():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_task(coro)
        loop.run_until_complete(future)
        loop.close()
    thread = threading.Thread(target=run_coroutine)
    thread.start()
    if blocking:
        thread.join()

def make_async(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)

    return _async_wrapper

def get_service_resouces(service_name: str, num_gpus: int) -> Dict[str, float]:
    assert service_name in ["prefill", "decode", "no_constraints", None], \
        "Only support prefill, decode, no_constraints, and None service name currently."
    if service_name == "prefill":
        resources = {"PREFILL_GPU": num_gpus}
    elif service_name == "decode":
        resources = {"DECODE_GPU": num_gpus}
    else: # service_name == "no_constraints", service_name is None
        resources = {}
    return resources

def get_llumnix_env_vars():
    llumnix_env_vars = {}
    env_vars = dict(os.environ)
    llumnix_env_vars_keys = list(llumnix_envs.environment_variables.keys())
    try:
        # pylint: disable=import-outside-toplevel
        from vllm import envs as vllm_envs
        llumnix_env_vars_keys.extend(list(vllm_envs.environment_variables.keys()))
    except ImportError:
        pass
    for key, value in env_vars.items():
        if key in llumnix_env_vars_keys:
            llumnix_env_vars[key] = value

    return llumnix_env_vars

def get_service_instance_type(service_name: str) -> "InstanceType":
    # pylint: disable=import-outside-toplevel
    from llumnix.instance_info import InstanceType
    assert service_name in ["prefill", "decode"], \
        "Only specify instance type when the service is prefill or decode."
    if service_name == "prefill":
        instance_type = InstanceType.PREFILL
    else:
        instance_type = InstanceType.DECODE
    return instance_type

def get_ip_address():
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    # pylint: disable=broad-except
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    # pylint: disable=broad-except
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    # pylint: disable=broad-except
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " VLLM_HOST_IP or HOST_IP.",
        stacklevel=2)
    return "0.0.0.0"

def _bind_and_close_port(port: Optional[int] = None, host: str = '0.0.0.0') -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # the SO_REUSEADDR flag tells the kernel to reuse a local socket in TIME_WAIT state,
        # without waiting for its natural timeout to expire. see https://docs.python.org/3/library/socket.html#example
        # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port or 0))
        return s.getsockname()[1]

def _get_port_by_pid(pid: int, start: int, end: int) -> int:
    assert start < end
    assert end <= _MAX_PORT
    return pid % (end - start) + start

def get_free_port() -> int:
    # try to find a free port based on pid to avoid port conflict between multiple processes
    base_port = os.getpid()
    for i in range(10000, 60000, 2000):
        port = _get_port_by_pid(base_port, i, i + 2000)
        if check_free_port(port=port) and check_free_port(port=port + 1):
            return port
    # fallback to random port if pid based port in all segments are occupied
    return _bind_and_close_port()

def check_free_port(host='0.0.0.0', port=8081):
    try:
        _bind_and_close_port(port=port, host=host)
        return True
    except socket.error as e:
        # pylint: disable=no-else-return
        if e.errno == socket.errno.EADDRINUSE:
            return False
        else:
            raise

def wait_port_free(port: int, max_retries: int = 5, force: bool = False):
    retries = 0
    history_pid = None

    while retries < max_retries: # pylint: disable=too-many-nested-blocks
        if check_free_port(port=port):
            return

        start_time = time.time()
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                logger.info("Port {} connection detail: {}".format(port, conn))
                if conn.pid and history_pid != conn.pid:
                    history_pid = conn.pid
                    try:
                        proc = psutil.Process(conn.pid)
                        logger.info("Port {} is in use by process {}, status {}: {}.".format(
                            port, conn.pid, proc.status(), ' '.join(proc.cmdline())))
                        if force:
                            proc.kill()
                            proc.wait(timeout=5)
                    except psutil.NoSuchProcess:
                        continue

                if conn.status == 'TIME_WAIT':
                    time.sleep(60)

        gc.collect()
        time.sleep(3)
        retries += 1

        cost_time = time.time() - start_time
        logger.info("Waiting for port {} to be free for {} seconds...".format(port, cost_time))

    raise RuntimeError(f"Port {port} is still in use after {max_retries} retries.")

def update_environment_variables(envs: Dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning("Overwriting environment variable {} from '{}' to '{}'".format(k, os.environ[k], v))
        os.environ[k] = v

def ray_get_with_timeout(object_refs, *args, timeout=RAY_RPC_TIMEOUT, **kwargs):
    return ray.get(object_refs, *args, timeout=timeout, **kwargs)

def asyncio_wait_for_with_timeout(fut, *args, timeout=RAY_RPC_TIMEOUT, **kwargs):
    return asyncio.wait_for(fut, *args, timeout=timeout, **kwargs)

async def async_wrapper(ray_call, *args, **kwargs):
    return await ray_call(*args, **kwargs)

def execute_method_with_timeout(method, timeout, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(method, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as e:
            raise TimeoutError(f"Method {method.__name__} timed out after {timeout} seconds") from e
