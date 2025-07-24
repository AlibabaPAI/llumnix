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

from typing import Any, Union, Dict, List, Tuple, Optional
import glob
import os
import pickle
import asyncio
from functools import partial
from enum import Enum

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
from llumnix.utils import (InstanceType, log_instance_exception, FailoverMigrationStatus,
                           asyncio_wait_for_ray_remote_call_with_timeout)

logger = init_logger(__name__)

# ================== placement_group/actor naming api ==================

MANAGER_NAME = "manager"
SCALER_NAME = "scaler"
PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"
DP_MANAGER_NAME_PREFIX = "dp_manager_"

def get_placement_group_name(unit_id: str) -> str:
    return f"{PLACEMENT_GROUP_NAME_PREFIX}{unit_id}"

def get_placement_group_unit_id(placement_group_name: str) -> str:
    return placement_group_name[len(PLACEMENT_GROUP_NAME_PREFIX):]


_ACTOR_NAME_PREFIX_MAPPING: Dict[str, str] = {
    "scaler": SCALER_NAME,
    "manager": MANAGER_NAME,
    "dp_manager": DP_MANAGER_NAME_PREFIX,
    "instance": INSTANCE_NAME_PREFIX,
    "server": SERVER_NAME_PREFIX,
}

class LlumnixActor(str, Enum):
    SCALER = "scaler"
    MANAGER = "manager"
    DP_MANAGER = "dp_manager"
    INSTANCE = "instance"
    SERVER = "server"

    @property
    def name_prefix(self) -> str:
        return _ACTOR_NAME_PREFIX_MAPPING[self.value]

    def get_actor_name(self, actor_id: str = None) -> str:
        return f"{self.name_prefix}{actor_id}" if actor_id is not None else self.name_prefix

    def get_actor_id(self, actor_name: str):
        assert self in [LlumnixActor.DP_MANAGER, LlumnixActor.INSTANCE, LlumnixActor.SERVER], \
            "Only dp manager, instance and server actors have actor name in its actor name in Llumnix actor naming rules."
        return actor_name[len(self.name_prefix):]

def get_llumnix_actor_name(actor_type: LlumnixActor, actor_id: str = None) -> str:
    return actor_type.get_actor_name(actor_id)

def get_llumnix_actor_id(actor_type: LlumnixActor, actor_name: str) -> str:
    assert actor_type in [LlumnixActor.DP_MANAGER, LlumnixActor.INSTANCE, LlumnixActor.SERVER], \
        "Only dp manager, instance and server actors have actor id in its actor name in Llumnix actor naming rules."
    return actor_type.get_actor_id(actor_name)

def get_llumnix_actor_handle(
    actor_type: LlumnixActor, actor_id: str = None, raise_exc: bool = True
) -> Optional[ray.actor.ActorHandle]:
    try:
        return ray.get_actor(actor_type.get_actor_name(actor_id), namespace="llumnix")
    except ValueError as e:
        if raise_exc:
            raise e
        return None

def actor_exists(name: str) -> bool:
    try:
        ray.get_actor(name, namespace="llumnix")
        return True
    except ValueError:
        return False
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in actor_exists (actor_name: {})".format(name))
        return False

def get_actor_handle(name: str) -> ray.actor.ActorHandle:
    try:
        return ray.get_actor(name, namespace="llumnix")
    except ValueError:
        return None

# ================== list placement_group/actor api ==================

def list_placement_group_infos_by_state(state: str = None) -> List[PlacementGroup]:
    if state is None:
        return ray.util.placement_group_table().values()
    curr_pg_infos = []
    for pg_info in ray.util.placement_group_table().values():
        if pg_info["state"] == state:
            curr_pg_infos.append(pg_info)
    return curr_pg_infos

def list_placement_group_infos_by_name(name: str) -> List[PlacementGroup]:
    curr_pg_infos = []
    for pg_info in ray.util.placement_group_table().values():
        if pg_info["name"] == name:
            curr_pg_infos.append(pg_info)
    return curr_pg_infos

def list_actor_names_by_name_prefix(name_prefix: str) -> List[str]:
    actor_infos = ray.util.list_named_actors(True)
    curr_actor_names = []
    for actor_info in actor_infos:
        if actor_info["name"].startswith(name_prefix):
            curr_actor_names.append(actor_info["name"])
    return curr_actor_names

def list_actor_names_by_actor_type(actor_type: LlumnixActor) -> List[str]:
    return list_actor_names_by_name_prefix(actor_type.name_prefix)

# ================== creation/deletion placement_group/actor api ==================

class BundlingStrategy(str, Enum):
    INSTANCE = "instance"
    WORKER = "worker"

# pylint: disable=dangerous-default-value
def initialize_placement_group(
    placement_group_name: str,
    num_cpus: int,
    num_gpus: int,
    detached: bool = False,
    block: bool = True,
    node_id: str = None,
    dp_size: int = 1,
    bundling_strategy: BundlingStrategy = BundlingStrategy.WORKER,
    resources: Dict[str, float] = {}
) -> PlacementGroup:
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
        if bundling_strategy == BundlingStrategy.INSTANCE:
            num_cpu_per_dp = (num_cpus - 1) // dp_size
            world_size = num_gpus // dp_size
            # bundle_0: DPManager + CPU Actors + Worker * world_size
            first_bundle = {"CPU": num_cpu_per_dp + 1, "GPU": world_size}
            # bundle_1~bundle_N: CPU Actors + Worker * world_size
            regular_bundle = {"CPU": num_cpu_per_dp, "GPU": world_size}
            placement_group_specs = [first_bundle]
            for _ in range(dp_size - 1):
                placement_group_specs.append(regular_bundle)
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

def clear_gloo_backend_ray_resources():
    try:
        # clear gloo migrate backend intermediate state
        ray.kill(ray.get_actor("gloo_queue", "llumnix"))
    except ValueError:
        # gloo_queue may not have been created yet; just ignore this error.
        pass
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in clear_gloo_backend_ray_resources")

def remove_placement_group(unit_id: str, placement_group: PlacementGroup = None) -> bool:
    try:
        if not placement_group:
            placement_group = ray.util.get_placement_group(get_placement_group_name(unit_id))
        # asynchronous api
        ray.util.remove_placement_group(placement_group)
        logger.info("Remove placement group {}.".format(unit_id))
    except ValueError:
        return False
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in remove_placement_group (unit_id: {})".format(unit_id))
        return False
    return True

async def kill_server(instance_id: str, server: ray.actor.ActorHandle = None) -> bool:
    try:
        if not server:
            server = get_llumnix_actor_handle(LlumnixActor.SERVER, instance_id, raise_exc=True)
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
            instance = get_llumnix_actor_handle(LlumnixActor.INSTANCE, instance_id, raise_exc=True)
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

async def kill_dp_manager(unit_id: str, dp_manager: ray.actor.ActorHandle = None) -> bool:
    try:
        if not dp_manager:
            dp_manager = get_llumnix_actor_handle(LlumnixActor.DP_MANAGER, unit_id, raise_exc=True)
        try:
            await dp_manager.stop.remote()
        # pylint: disable=bare-except
        except:
            pass
        ray.kill(dp_manager)
        logger.info("Kill dp manager {}.".format(unit_id))
    except ValueError:
        return False
    except Exception: # pylint: disable=broad-except
        logger.exception("Error in kill_dp_manager (unit_id: {})".format(unit_id))
        return False
    return True

# ================== ray internal kv api ==================

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

# ================== actor utility api ==================

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

async def connect_to_actors_with_instance_type(
    actor_type: LlumnixActor
) -> Tuple[List[str], List[ray.actor.ActorHandle], List[InstanceType]]:
    def connect_to_actor_done_callback(actor_id: str, actor_handle: ray.actor.ActorHandle, fut):
        ret = fut.result()[0]
        if not isinstance(ret, Exception):
            try:
                actor_ids.append(actor_id)
                actor_handles.append(actor_handle)
                instance_types.append(ret)
                logger.info("Connect to {} {}".format(actor_type, actor_id))
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ValueError):
                    logger.warning(
                        "Failed to connect to {} {}, placement group not found.".format(
                            actor_type, actor_id
                        )
                    )
                else:
                    logger.exception(
                        "Error in _connect_to_actors get_placement_group (actor_type: {}, actor_id: {})".format(
                            actor_type, actor_id
                        )
                    )
        else:
            log_instance_exception(ret, actor_id, "_connect_to_actors")

    actor_names = list_actor_names_by_actor_type(actor_type)
    available_actor_names = []
    available_actor_handles: List[ray.actor.ActorHandle] = []
    for actor_name in actor_names:
        try:
            actor_handle = ray.get_actor(actor_name, namespace='llumnix')
            available_actor_names.append(actor_name)
            available_actor_handles.append(actor_handle)
        except Exception as e: # pylint: disable=broad-except
            actor_id = get_llumnix_actor_id(actor_type, actor_name)
            if isinstance(e, ValueError):
                logger.warning(
                    "Failed to connect to {} {}, actor not found.".format(
                        actor_type, actor_id
                    )
                )
            else:
                logger.exception("Error in _connect_to_actors get_actor (actor_id: {})".format(actor_id))

    actor_ids = []
    actor_handles = []
    instance_types = []
    tasks = []
    for actor_name, actor_handle in \
        zip(available_actor_names, available_actor_handles):
        actor_id = get_llumnix_actor_id(actor_type, actor_name)
        task = asyncio.gather(
            asyncio_wait_for_ray_remote_call_with_timeout(actor_handle.get_instance_type),
            return_exceptions=True
        )
        task.add_done_callback(
            partial(connect_to_actor_done_callback, actor_id, actor_handle)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)

    return actor_ids, actor_handles, instance_types

def update_cluster_actor_handles(
    actor_type: LlumnixActor,
    cached_cluster_actors: List[ray.actor.ActorHandle],
) -> List[ray.actor.ActorHandle]:
    curr_actor_names = list_actor_names_by_actor_type(actor_type)
    curr_actor_ids = [get_llumnix_actor_id(actor_type, actor_name) for actor_name in curr_actor_names]
    new_cluster_actors = {}
    for actor_id in curr_actor_ids:
        if actor_id in cached_cluster_actors:
            new_cluster_actors[actor_id] = cached_cluster_actors[actor_id]
        else:
            actor = get_llumnix_actor_handle(actor_type, actor_id, raise_exc=False)
            if actor is not None:
                new_cluster_actors[actor_id] = actor
    return new_cluster_actors

async def check_actors_health(actors: Dict[str, ray.actor.ActorHandle]) -> List[str]:
    def check_actor_health_done_callback(actor_id: str, fut):
        ret = fut.result()[0]
        if isinstance(ret, Exception):
            log_instance_exception(ret, actor_id, "check_actor_health")
            dead_actor_ids.append(actor_id)

    tasks = []
    dead_actor_ids = []
    for actor_id, actor in actors.items():
        task = asyncio.gather(
            asyncio_wait_for_ray_remote_call_with_timeout(actor.is_ready),
            return_exceptions=True
        )
        task.add_done_callback(
            partial(check_actor_health_done_callback, actor_id)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)

    return dead_actor_ids

async def check_instance_ready_to_die(instances: Dict[str, ray.actor.ActorHandle]) -> List[FailoverMigrationStatus]:
    def check_instance_ready_to_die_done_callback(actor_id: str, fut):
        ret = fut.result()[0]
        if isinstance(ret, Exception):
            log_instance_exception(ret, actor_id, "check_instance_ready_to_die")
        else:
            instance_status.append(ret)

    tasks = []
    instance_status = []
    for instance_id, instance in instances.items():
        task = asyncio.gather(
            asyncio_wait_for_ray_remote_call_with_timeout(instance.get_failover_migration_status),
            return_exceptions=True
        )
        task.add_done_callback(
            partial(check_instance_ready_to_die_done_callback, instance_id)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)

    return instance_status
