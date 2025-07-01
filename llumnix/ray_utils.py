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

from typing import Any, Union, Dict, List
import glob
import os
import pickle

import ray
import ray.actor
from ray.util.placement_group import PlacementGroup
from ray.experimental.internal_kv import (
    _internal_kv_get,
    _internal_kv_initialized,
    _internal_kv_put,
    _internal_kv_exists
)
import ray.exceptions

from llumnix.logging.logger import init_logger
from llumnix.constants import WAIT_PLACEMENT_GROUP_TIMEOUT

logger = init_logger(__name__)

MANAGER_NAME = "manager"
SCALER_NAME = "scaler"
PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"
DPMANAGER_NAME = "dpmanager_"


def get_manager_name() -> str:
    return MANAGER_NAME

def get_scaler_name() -> str:
    return SCALER_NAME

def get_placement_group_name(instance_id: str) -> str:
    return f"{PLACEMENT_GROUP_NAME_PREFIX}{instance_id}"

def get_server_name(instance_id: str) -> str:
    return f"{SERVER_NAME_PREFIX}{instance_id}"

def get_instance_name(instance_id: str) -> str:
    return f"{INSTANCE_NAME_PREFIX}{instance_id}"

def get_dpmanager_name(instance_id: str) -> str:
    return f"{DPMANAGER_NAME}{instance_id}"

# pylint: disable=dangerous-default-value
def initialize_placement_group(
    placement_group_name: str,
    num_cpus: int,
    num_gpus: int,
    detached: bool = False,
    block: bool = True,
    node_id: str = None,
    dp_size: int = 1,
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
        node_id: The node id of node. If specified, placement group will be created on the specified node.
        resources: The addtional resources requirements of placement group.

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
            "The number of required GPUs {} exceeds the total number of "
            "available GPUs {} in the cluster.".format(num_gpus, num_gpus_in_cluster))

    try:
        # Create a new placement group
        if dp_size > 1:
            # bundle_N: DP rank N, CPU Actors + Worker * world_size
            num_cpu_per_dp = num_cpus // dp_size
            world_size = num_gpus // dp_size
            placement_group_specs_per_dp = [{"CPU": num_cpu_per_dp, "GPU": world_size}]
            placement_group_specs = []
            for _ in range(dp_size):
                placement_group_specs += placement_group_specs_per_dp
            logger.info(f"[sjr] {placement_group_specs=}")
        elif num_gpus >= 1:
            # bundle_0: All CPU Actors + Worker_0, bundle_1-N-1: Worker_1...Worker_N-1
            placement_group_specs = [{"CPU": num_cpus, "GPU": 1}] + [{"GPU": 1}] * (num_gpus - 1)
        else:
            placement_group_specs = [{"CPU": num_cpus}]
        if resources:
            placement_group_specs += [resources]
        # pylint: disable=self-assigning-variable
        placement_group_specs = (placement_group_specs)

        logger.debug("placement_group_specs: {}".format(placement_group_specs))

        # PACK (not STRICT_PACK) to support multi-node placement group.
        if node_id is None:
            current_placement_group = ray.util.placement_group(
                placement_group_specs, "PACK", name=placement_group_name, lifetime=lifetime)
        else:
            current_placement_group = ray.util.placement_group(
                placement_group_specs, "STRICT_PACK", name=placement_group_name, lifetime=lifetime, _soft_target_node_id=node_id)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        if block:
            try:
                ray.get(current_placement_group.ready(), timeout=WAIT_PLACEMENT_GROUP_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                logger.warning("Waiting for new placement group {} ready timeout.".format(placement_group_name))
                return None
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in initialize_placement_group (placement_group_name: {})".format(placement_group_name))
        return None

    return current_placement_group

# Merge to initialize_placement_group
def initialize_placement_group_dp(
    placement_group_name: str,
    num_cpus: int,
    num_gpus: int,
    dp_size: int,
    detached: bool = False,
    block: bool = True,
    node_id: str = None,
    resources: Dict[str, float] = {}
) -> PlacementGroup:
    """Initialize the distributed cluster for data parallelism probably with Ray.
    Currently only support vLLM V1.
    """
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed "
            "serving.")

    lifetime = "detached" if detached else None

    num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
    if num_gpus > num_gpus_in_cluster:
        raise ValueError(
            "The number of required GPUs {} exceeds the total number of "
            "available GPUs {} in the cluster.".format(num_gpus, num_gpus_in_cluster))
    
    try:
        # Create a new placement group
        # bundle_N: DP rank N, APIServer + Llumlet + Worker * world_size
        num_cpu_per_dp = num_cpus // dp_size
        world_size = num_gpus // dp_size
        placement_group_specs_per_dp = [{"CPU": num_cpu_per_dp, "GPU": world_size}]
        placement_group_specs = []
        for _ in range(dp_size):
            placement_group_specs += placement_group_specs_per_dp
        if resources:
            placement_group_specs += [resources]
        logger.info(f"[sjr] {placement_group_specs=}")
        # pylint: disable=self-assigning-variable
        placement_group_specs = (placement_group_specs)

        logger.debug("placement_group_specs: {}".format(placement_group_specs))

        # PACK (not STRICT_PACK) to support multi-node placement group.
        if node_id is None:
            current_placement_group = ray.util.placement_group(
                placement_group_specs, "PACK", name=placement_group_name, lifetime=lifetime)
        else:
            current_placement_group = ray.util.placement_group(
                placement_group_specs, "STRICT_PACK", name=placement_group_name, lifetime=lifetime, _soft_target_node_id=node_id)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        if block:
            try:
                ray.get(current_placement_group.ready(), timeout=WAIT_PLACEMENT_GROUP_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                logger.warning("Waiting for new placement group {} ready timeout.".format(placement_group_name))
                return None
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in initialize_placement_group (placement_group_name: {})".format(placement_group_name))
        return None

    return current_placement_group

def get_placement_group_infos_by_state(state: str = None) -> List[PlacementGroup]:
    if state is None:
        return ray.util.placement_group_table().values()
    target_placement_group_infos = []
    for placement_group_info in ray.util.placement_group_table().values():
        if placement_group_info["state"] == state:
            target_placement_group_infos.append(placement_group_info)
    return target_placement_group_infos

def get_placement_group_infos_by_name(name: str) -> List[PlacementGroup]:
    target_placement_group_infos = []
    for placement_group_info in ray.util.placement_group_table().values():
        if placement_group_info["name"] == name:
            target_placement_group_infos.append(placement_group_info)
    return target_placement_group_infos

def actor_exists(name: str) -> bool:
    try:
        ray.get_actor(name, namespace="llumnix")
        return True
    except ValueError:
        return False
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in actor_exists (actor_name: {})".format(name))
        return False

def get_actor_names_by_name_prefix(name_prefix: str) -> List[str]:
    actor_infos = ray.util.list_named_actors(True)
    target_actor_names = []
    for actor_info in actor_infos:
        if actor_info["name"].startswith(name_prefix):
            target_actor_names.append(actor_info["name"])
    return target_actor_names

def clear_gloo_backend_ray_resources():
    try:
        # clear gloo migrate backend intermediate state
        ray.kill(ray.get_actor("gloo_queue", "llumnix"))
    except ValueError:
        # gloo_queue may not have been created yet; just ignore this error.
        pass
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in clear_gloo_backend_ray_resources")

def remove_placement_group(instance_id: str, placement_group: PlacementGroup = None) -> bool:
    try:
        if not placement_group:
            placement_group = ray.util.get_placement_group(get_placement_group_name(instance_id))
        # asynchronous api
        ray.util.remove_placement_group(placement_group)
        logger.info("Remove placement group {}.".format(instance_id))
    except ValueError:
        return False
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in remove_placement_group (instance_id: {})".format(instance_id))
        return False
    return True

async def kill_server(instance_id: str, server: ray.actor.ActorHandle = None) -> bool:
    try:
        if not server:
            server = ray.get_actor(get_server_name(instance_id), namespace="llumnix")
        try:
            await server.stop.remote()
        # pylint: disable=bare-except
        except:
            pass
        ray.kill(server)
        logger.info("Kill server {}.".format(instance_id))
    except ValueError:
        return False
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in kill_server (instance_id: {})".format(instance_id))
        return False
    return True

async def kill_instance(instance_id: str, instance: ray.actor.ActorHandle = None) -> bool:
    try:
        if not instance:
            instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
        try:
            await instance.stop.remote()
        # pylint: disable=bare-except
        except:
            pass
        ray.kill(instance)
        logger.info("Kill instance {}.".format(instance_id))
    except ValueError:
        return False
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in kill_instance (instance_id: {})".format(instance_id))
        return False
    return True

def get_instance(instance_id: str) -> ray.actor.ActorHandle:
    try:
        instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
        return instance
    except ValueError:
        return None

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
    assert _internal_kv_initialized(), \
        f"Ray internal key-value storage should be initialized to get data {data_name}."
    key = _make_key(data_name)
    assert _internal_kv_exists(key), \
        f"The given data {data_name} is not exist in the ray internal key-value storage."
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
        except Exception:
            logger.exception("Error in put_data_to_ray_internal_kv")
    else:
        logger.error("Ray internal key-value storage is not initilized, "
                     "failed to put the given data {}.".format(data_name))

def log_actor_ray_info(actor_class_name: str) -> None:
    try:
        actor_name = ray.get_runtime_context().get_actor_name()
        actor_id = ray.get_runtime_context().get_actor_id()
        placement_group_id = ray.get_runtime_context().get_placement_group_id()
        namespace = ray.get_runtime_context().namespace
        job_id = ray.get_runtime_context().get_job_id()
        worker_id = ray.get_runtime_context().get_worker_id()
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_runtime_context().get_accelerator_ids()["GPU"]
        # assigned_resources = ray.get_runtime_context().get_assigned_resources()
        logger.info("{}(actor_name={}, actor_id={}, placement_group_id={}, namespace={}, " \
                    "job_id={}, worker_id={}, node_id={}, gpu_ids={})".format(
                        actor_class_name, actor_name, actor_id, placement_group_id, namespace,
                        job_id, worker_id, node_id, gpu_ids))
        # It seems that the logging directory of ray cannot easily get by codes, so use the indirect way below.
        # pylint: disable=protected-access
        session_dir = ray._private.worker._global_node.get_session_dir_path()
        pattern = os.path.join(session_dir, "logs", f"worker-{worker_id}*")
        logger.info("{}(log_dir={})".format(actor_class_name, glob.glob(pattern)))
    # pylint: disable=broad-except
    except Exception:
        logger.exception("Error in log_actor_ray_info (actor_class_name: {})".format(actor_class_name))
