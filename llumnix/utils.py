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
from typing import Any, Union, Callable, Awaitable, TypeVar, Coroutine, Dict, List
from enum import Enum
from functools import partial
import pickle
import glob
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
from llumnix import envs as llumnix_envs

logger = init_logger(__name__)

MANAGER_NAME = "manager"
PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"

P = ParamSpec('P')
T = TypeVar("T")


class GPUBundlingStrategy(str, Enum):
    SPREAD = "spread"
    PACK = "pack"


# pylint: disable=dangerous-default-value
def initialize_placement_group(
    placement_group_name: str,
    num_cpus: int,
    num_gpus: int,
    gpu_bundling_strategy: GPUBundlingStrategy = GPUBundlingStrategy.SPREAD,
    detached: bool = False,
    block: bool = True,
    resources: Dict[str, float] = {}
) -> PlacementGroup:
    """Initialize the distributed cluster probably with Ray.

    Args:
        placement_group_name: The name of placement group.
        num_cpus: The number of cpus in placement group.
        num_gpus: The number of gpus in placement group.
        gpu_bundling_strategy: GPU bundle st.
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
    if gpu_bundling_strategy == GPUBundlingStrategy.SPREAD:
        placement_group_specs = [{"CPU": num_cpus, "GPU": 1}] + [{"GPU": 1}] * (num_gpus - 1)
    else:  # GPUBundlingStrategy.PACK
        placement_group_specs = [{"CPU": num_cpus}] if num_gpus == 0 else [{"CPU": num_cpus, "GPU": num_gpus}]
    if resources:
        placement_group_specs += [resources]
    # pylint: disable=self-assigning-variable
    placement_group_specs = (placement_group_specs)

    logger.debug("placement_group_specs: {}".format(placement_group_specs))

    current_placement_group = ray.util.placement_group(
        placement_group_specs, "STRICT_PACK", name=placement_group_name, lifetime=lifetime)
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    if block:
        ray.get(current_placement_group.ready(), timeout=1.0)

    return current_placement_group

def get_placement_group_infos_by_state(state: str = None) -> Dict[str, str]:
    if state is None:
        return placement_group_table().values()
    target_placement_group_infos = []
    for placement_group_info in placement_group_table().values():
        if placement_group_info["state"] == state:
            target_placement_group_infos.append(placement_group_info)
    return target_placement_group_infos

def get_placement_group_infos_by_name(name: str) -> Dict[str, str]:
    target_placement_group_infos = []
    for placement_group_info in placement_group_table().values():
        if placement_group_info["name"] == name:
            target_placement_group_infos.append(placement_group_info)
    return target_placement_group_infos

def actor_exists(name: str) -> bool:
    try:
        ray.get_actor(name, namespace="llumnix")
        return True
    except ValueError:
        return False

def get_actor_names_by_name_prefix(name_prefix: str) -> List[str]:
    actor_infos = ray.util.list_named_actors(True)
    target_actor_names = []
    for actor_info in actor_infos:
        if actor_info["name"].startswith(name_prefix):
            target_actor_names.append(actor_info["name"])
    return target_actor_names

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

def log_actor_ray_info(actor_class_name: str) -> None:
    actor_name = ray.get_runtime_context().get_actor_name()
    actor_id = ray.get_runtime_context().get_actor_id()
    placement_group_id = ray.get_runtime_context().get_placement_group_id()
    namespace = ray.get_runtime_context().namespace
    job_id = ray.get_runtime_context().get_job_id()
    worker_id = ray.get_runtime_context().get_worker_id()
    node_id = ray.get_runtime_context().get_node_id()
    # assigned_resources = ray.get_runtime_context().get_assigned_resources()
    logger.info("{}(actor_name={}, actor_id={}, placement_group_id={}, namespace={}, " \
                "job_id={}, worker_id={}, node_id={})".format(
                    actor_class_name, actor_name, actor_id, placement_group_id, namespace,
                    job_id, worker_id, node_id))
    # It seems that the logging directory of ray cannot easily get by codes, so use the indirect way below.
    # pylint: disable=protected-access
    session_dir = ray._private.worker._global_node.get_session_dir_path()
    pattern = os.path.join(session_dir, "logs", f"worker-{worker_id}*")
    logger.info("{}(log_dir={})".format(actor_class_name, glob.glob(pattern)))

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

# Cannot put it to utils.py due to circular import.
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
