from typing import Any, Union, Dict, List
from enum import Enum
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
from ray.util import placement_group_table

from llumnix.logging.logger import init_logger

logger = init_logger(__name__)

MANAGER_NAME = "manager"
SCALER_NAME = "scaler"
PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"


class GPUBundlingStrategy(str, Enum):
    SPREAD = "spread"
    PACK = "pack"


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
            "The number of required GPUs {} exceeds the total number of "
            "available GPUs {} in the cluster.".format(num_gpus, num_gpus_in_cluster))

    # Create a new placement group
    # bundle_0: Llumlet + AsyncPutQueueActor + Worker0, bundle_1-N-1: Worker1...WorkerN-1
    if gpu_bundling_strategy == GPUBundlingStrategy.SPREAD:
        if num_gpus >= 1:
            placement_group_specs = [{"CPU": num_cpus, "GPU": 1}] + [{"GPU": 1}] * (num_gpus - 1)
        else:
            placement_group_specs = [{"CPU": num_cpus}]
    else:  # GPUBundlingStrategy.PACK
        placement_group_specs = [{"CPU": num_cpus}] if num_gpus == 0 else [{"CPU": num_cpus, "GPU": num_gpus}]
    if resources:
        placement_group_specs += [resources]
    # pylint: disable=self-assigning-variable
    placement_group_specs = (placement_group_specs)

    logger.debug("placement_group_specs: {}".format(placement_group_specs))

    # PACK (not STRICT_PACK) to support multi-node placement group.
    current_placement_group = ray.util.placement_group(
        placement_group_specs, "PACK", name=placement_group_name, lifetime=lifetime)
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    if block:
        ray.get(current_placement_group.ready(), timeout=3.0)

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

def clear_gloo_backend_ray_resources():
    try:
        # clear gloo migrate backend intermediate state
        ray.kill(ray.get_actor("gloo_queue", "llumnix"))
    # pylint: disable=broad-except
    except Exception:
        # gloo_queue may not have been created yet; just ignore this error.
        pass

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
            logger.exception("Unexpected exception: {}".format(e))
    else:
        logger.error("Ray internal key-value storage is not initilized, failed to put the given data {}.".format(data_name))

def log_actor_ray_info(actor_class_name: str) -> None:
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
