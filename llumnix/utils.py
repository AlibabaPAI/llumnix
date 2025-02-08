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

import uuid
import asyncio
import threading
from typing import Any, Union
import ray
from ray.util.placement_group import PlacementGroup
from ray.experimental.internal_kv import (
    _internal_kv_get,
    _internal_kv_initialized,
    _internal_kv_put,
)

from llumnix.logging.logger import init_logger

logger = init_logger(__name__)

MANAGER_NAME = "manager"
PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"


def initialize_placement_group(
    placement_group_name: str,
    num_cpus: int,
    num_gpus: int,
    detached: bool = False,
    block: bool = True
) -> PlacementGroup:
    """Initialize the distributed cluster probably with Ray.

    Args:
        placement_group_name: The name of placement group.
        num_cpus: The number of cpus in placement group.
        num_cpus: The number of cpus in placement group.
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
    # bundle_0: Llumlet + AsyncPutQueueActor, bundle_1: Workers
    placement_group_specs = ([{"CPU": num_cpus}] + [{"GPU": 1}] * num_gpus)
    current_placement_group = ray.util.placement_group(
        placement_group_specs, "STRICT_PACK", name=placement_group_name, lifetime=lifetime)
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    if block:
        ray.get(current_placement_group.ready(), timeout=1800)

    return current_placement_group

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

def clear_gloo_backend_state():
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

def remove_placement_group(instance_id: str) -> bool:
    try:
        placement_group = ray.util.get_placement_group(get_placement_group_name(instance_id))
        # asynchronous api
        ray.util.remove_placement_group(placement_group)
        logger.info("Remove placement group {}.".format(instance_id))
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def kill_server(instance_id: str) -> bool:
    try:
        server = ray.get_actor(get_server_name(instance_id), namespace="llumnix")
        ray.kill(server)
        logger.info("Kill server {}.".format(instance_id))
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def kill_instance(instance_id: str) -> bool:
    try:
        instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
        ray.kill(instance)
        logger.info("Kill instance {}.".format(instance_id))
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def run_async_func_sync(func):
    def run_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_task(func)
        loop.run_until_complete(future)
        loop.close()
    thread = threading.Thread(target=run_task)
    thread.start()
    thread.join()

def _make_key(actor_name: str, data_name: str):
    """Generate a binary key for the given actor name and data.

    Args:
        actor_name: The name of the actor
        data_name: The data member of the actor

    Returns:
        The key to use for storing a the value.
    """
    return (actor_name.encode("ascii") + b"." + data_name.encode("ascii"))

def get_actor_data_from_ray_internal_kv(actor_name: str, data_name: str) -> Union[str, None]:
    value = None
    if _internal_kv_initialized():
        value = _internal_kv_get(_make_key(actor_name, data_name))
    if value is not None:
        value = value.decode()
    logger.info("Get {}.{} from ray internal key-value store, value: {}.".format(actor_name, data_name, value))
    return value

def put_actor_data_to_ray_internal_kv(actor_name: str, data_name: str, value: Any):
    if _internal_kv_initialized():
        _internal_kv_put(_make_key(actor_name, data_name), f"{value}".encode(), overwrite=True)
        logger.debug("Put {}.{} to ray internal key-value store, value: {}.".format(actor_name, data_name, value))
