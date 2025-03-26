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
import traceback
import threading
from typing import Any, Union, Callable, Awaitable, TypeVar, Coroutine
from functools import partial
import pickle
from typing_extensions import ParamSpec
import ray
import ray.actor
from ray.util.placement_group import PlacementGroup
from ray.experimental.internal_kv import (
    _internal_kv_get,
    _internal_kv_initialized,
    _internal_kv_put,
    _internal_kv_exists
)
from ray.util import placement_group_table

from llumnix.logging.logger import init_logger

logger = init_logger(__name__)

MANAGER_NAME = "manager"
PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"

P = ParamSpec('P')
T = TypeVar("T")


def initialize_placement_group(
    placement_group_name: str,
    num_cpus: int,
    num_gpus: int,
    seperate_gpu_groups: bool = True,
    detached: bool = False,
    block: bool = True
) -> PlacementGroup:
    """Initialize the distributed cluster probably with Ray.

    Args:
        placement_group_name: The name of placement group.
        num_cpus: The number of cpus in placement group.
        num_gpus: The number of gpus in placement group.
        seperate_gpu_groups: Whether to seperate gpu in bundles.
        detached: Whether the lifetime of the placement group being detached.
        block: If True, the function will block until the placement group is ready.

    Returns:
        `placement_group`. `placement_group` includes the specification
        of the resources for each distributed worker.
    """
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed "
            "serving.")

    lifetime = "detached" if detached else None

    num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
    if num_gpus > num_gpus_in_cluster:
        raise ValueError(
            "The number of required GPUs exceeds the total number of "
            "available GPUs in the cluster.")

    # Create a new placement group
    # bundle_0: Llumlet + AsyncPutQueueActor + Worker0, bundle_1-N-1: Worker1...WorkerN-1
    if seperate_gpu_groups:
        placement_group_specs = ([{"CPU": num_cpus, "GPU": 1}] + [{"GPU": 1}] * (num_gpus - 1))
    else:
        spec = {"CPU": num_cpus} if num_gpus == 0 else {"CPU": num_cpus, "GPU": num_gpus}
        placement_group_specs = ([spec])

    current_placement_group = ray.util.placement_group(
        placement_group_specs, "STRICT_PACK", name=placement_group_name, lifetime=lifetime)
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    if block:
        ray.get(current_placement_group.ready(), timeout=1800)

    return current_placement_group

def get_placement_group_infos_by_state(state: str = None):
    if state is None:
        return placement_group_table().values()
    target_placement_group_infos = []
    for placement_group_info in placement_group_table().values():
        if placement_group_info["state"] == state:
            target_placement_group_infos.append(placement_group_info)
    return target_placement_group_infos

def get_placement_group_infos_by_name(name: str):
    target_placement_group_infos = []
    for placement_group_info in placement_group_table().values():
        if placement_group_info["name"] == name:
            target_placement_group_infos.append(placement_group_info)
    return target_placement_group_infos

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

def clear_gloo_backend_ray_resources():
    try:
        # clear gloo migrate backend intermediate state
        ray.kill(ray.get_actor("gloo_queue", "llumnix"))
    # pylint: disable=broad-except
    except Exception:
        # gloo_queue may not have been created yet; just ignore this error.
        pass

def get_manager_name() -> str:
    return MANAGER_NAME

def get_placement_group_name(instance_id: str) -> str:
    return f"{PLACEMENT_GROUP_NAME_PREFIX}{instance_id}"

def get_server_name(instance_id: str) -> str:
    return f"{SERVER_NAME_PREFIX}{instance_id}"

def get_instance_name(instance_id: str) -> str:
    return f"{INSTANCE_NAME_PREFIX}{instance_id}"

def remove_placement_group(instance_id: str, placement_group: PlacementGroup = None) -> bool:
    try:
        if not placement_group:
            placement_group = ray.util.get_placement_group(get_placement_group_name(instance_id))
        # asynchronous api
        ray.util.remove_placement_group(placement_group)
        logger.info("Remove placement group {}.".format(instance_id))
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def kill_server(instance_id: str, server: ray.actor.ActorHandle = None) -> bool:
    try:
        if not server:
            server = ray.get_actor(get_server_name(instance_id), namespace="llumnix")
        ray.kill(server)
        logger.info("Kill server {}.".format(instance_id))
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def kill_instance(instance_id: str, instance: ray.actor.ActorHandle = None) -> bool:
    try:
        if not instance:
            instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
        ray.kill(instance)
        logger.info("Kill instance {}.".format(instance_id))
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

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

def _make_key(actor_name: str):
    return actor_name.encode("ascii")

def _load_value(value: Any):
    if isinstance(value, str):
        value = value.decode()
    else:
        value = pickle.loads(value)
    return value

def _dump_value(value: Any):
    if isinstance(value, str):
        value = f"{value}".encode()
    else:
        value = pickle.dumps(value)
    return value

def get_data_from_ray_internal_kv(data_name: str) -> Union[str, None]:
    assert _internal_kv_initialized(), f"Ray internal key-value storage should be initialized to get data {data_name}."
    key = _make_key(data_name)
    assert _internal_kv_exists(key), f"The given data {data_name} is not exist in the ray internal key-value storage."
    value = _internal_kv_get(key)
    value = _load_value(value)
    logger.info("Get data {} from ray internal key-value storage, value: {}.".format(data_name, value))

    return value

def put_data_to_ray_internal_kv(data_name: str, value: Any) -> None:
    if _internal_kv_initialized():
        logger.info("Put data {} to ray internal key-value storage, value: {}.".format(data_name, value))
        try:
            value = _dump_value(value)
            _internal_kv_put(_make_key(data_name), value, overwrite=True)
        # pylint: disable=W0703
        except Exception as e:
            logger.error("Unexpected exception: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))
    else:
        logger.error("Ray internal key-value storage is not initilized, failed to put the given data {}.".format(data_name))

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
