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

import os
import uuid
import asyncio
import threading
from typing import Any, Callable, Awaitable, TypeVar, Coroutine, Dict, Optional
import socket
from functools import partial
import pickle

from typing_extensions import ParamSpec

from llumnix.logging.logger import init_logger
from llumnix import envs as llumnix_envs
from llumnix.constants import MODEL_PATH, DATASET_PATH

logger = init_logger(__name__)

_MAX_PORT = 65536

P = ParamSpec('P')
T = TypeVar("T")


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

def _get_engine_args_filename(engine_type: str) -> str:
    return f"engine_args_{engine_type}.pkl"

def _get_engine_args_filepath(save_path: str, save_key: str = None) -> str:
    if save_key is not None:
        save_filepath = os.path.join(save_path, save_key)
    else:
        save_filepath = save_path
    return save_filepath

def save_engine_args(engine_type: str, save_path: str, engine_args: Any, save_key: str = None) -> None:
    engine_args_filename = _get_engine_args_filename(engine_type)
    save_filepath = _get_engine_args_filepath(save_path, save_key)
    save_filename = os.path.join(save_filepath, engine_args_filename)
    os.makedirs(save_filepath, exist_ok=True)
    with open(save_filename, 'wb') as file:
        pickle.dump(engine_args, file)
    logger.info("Save engine arguments of {} engine type as file: {}".format(engine_type, save_filename))

def load_engine_args(engine_type: str, load_path: str) -> Any:
    engine_args_filename = _get_engine_args_filename(engine_type)
    load_filename = os.path.join(load_path, engine_args_filename)
    with open(load_filename, 'rb') as file:
        engine_args =  pickle.load(file)
    logger.info("Load engine arguments of {} engine type from path: {}".format(engine_type, load_path))
    return engine_args

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
    llumnix_keys = list(llumnix_envs.environment_variables.keys())
    try:
        # pylint: disable=import-outside-toplevel
        from vllm import envs as vllm_envs
        llumnix_keys.extend(list(vllm_envs.environment_variables.keys()))
    except ImportError:
        pass
    for key, value in env_vars.items():
        if key in llumnix_keys:
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
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def _bind_and_close_port(port: Optional[int] = None, host: str = '0.0.0.0') -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # the SO_REUSEADDR flag tells the kernel to reuse a local socket in TIME_WAIT state,
        # without waiting for its natural timeout to expire. see https://docs.python.org/3/library/socket.html#example
        # NOTE(qzhong): Is it a risk to reuse old port before closing it?
        # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port or 0))
        return s.getsockname()[1]

def _get_port_by_pid(pid: int, start: int, end: int) -> int:
    assert start < end
    assert end <= _MAX_PORT
    return pid % (end - start) + start

def get_free_port() -> int:
    # try to find a free port based on pid in each 10000 length segment
    # to avoid port conflict between multiple processes
    base_port = os.getpid()
    for i in range(10000, 60000, 10000):
        port = _get_port_by_pid(base_port, i, i + 10000)
        if check_free_port(port=port):
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

def as_local(data_path: str) -> str:
    assert "/" in data_path
    base_name = data_path.split("/")[-1]

    local_model_path: str = os.path.join(MODEL_PATH, base_name)
    if os.path.exists(local_model_path):
        return local_model_path
    
    local_dataset_path: str = os.path.join(DATASET_PATH, base_name)
    if os.path.exists(local_dataset_path):
        return local_dataset_path
