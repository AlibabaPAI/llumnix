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
import ray
from ray.util.placement_group import PlacementGroup

MANAGER_NAME = "manager"
PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"


def initialize_placement_group(
    instance_id: str,
    num_cpus: int = 1,
    num_gpus: int = 1,
    detached: bool = False
) -> PlacementGroup:
    """Initialize the distributed cluster probably with Ray.

    Args:
        instance_id: The instance id of the instance scheduled to the placement group.
        num_cpus: The number of cpus in placement group.
        num_cpus: The number of cpus in placement group.
        detached: Whether the lifetime of the placement group being detached.

    Returns:
        `placement_group`. `placement_group` includes the specification
        of the resources for each distributed worker.
    """
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed "
            "serving.")

    lifetime = "detached" if detached else None
    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        gpu_bundles = 0
        for bundle in bundles:
            bundle_gpus = bundle.get("GPU", 0)
            if bundle_gpus > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 GPU.")
            if bundle_gpus:
                gpu_bundles += 1
        if num_gpus > gpu_bundles:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the placement group.")
    else:
        num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
        if num_gpus > num_gpus_in_cluster:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the cluster.")
        # Create a new placement group
        # bundle_0: Llumlet + AsyncPutQueueActor + ProxyActor, bundle_1: Workers
        placement_group_specs = ([{"CPU": num_cpus}] + [{"GPU": 1}] * num_gpus)
        current_placement_group = ray.util.placement_group(
            placement_group_specs, "STRICT_PACK", name=get_placement_group_name(instance_id), lifetime=lifetime)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
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
    except ValueError:
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
        if not placement_group:
            return False
        # asynchronous api
        ray.util.remove_placement_group(placement_group)
        logger.info("remove placement group {}".format(instance_id))
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def kill_server(instance_id: str) -> bool:
    try:
        server = ray.get_actor(get_server_name(instance_id), namespace="llumnix")
    except ValueError:
        return False
    try:
        ray.kill(server)
        logger.info("kill server {}".format(instance_id))
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def kill_instance(instance_id: str) -> bool:
    try:
        instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
    except ValueError:
        return False
    try:
        ray.kill(instance)
        print("kill instance {}".format(instance_id))
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
