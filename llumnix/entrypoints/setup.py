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

import subprocess
import sys
import os
import time
from typing import Dict, Optional, List, Tuple

import ray
import ray.exceptions

from llumnix.manager import Manager
from llumnix.llumlet.llumlet import Llumlet
from llumnix.logging.logger import init_logger
from llumnix.utils import random_uuid, get_llumnix_env_vars
from llumnix.ray_utils import get_manager_name, get_scaler_name
from llumnix.arg_utils import ManagerArgs, EntrypointsArgs, LaunchArgs, InstanceArgs, LlumnixEngineArgs
from llumnix.queue.queue_type import QueueType
from llumnix.server_info import ServerInfo
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.entrypoints.utils import LaunchMode, EntrypointsContext
from llumnix.utils import get_ip_address, log_instance_exception, ray_get_with_timeout
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.constants import MAX_RAY_RESTART_TIMES, RAY_RESTART_INTERVAL, SUBPROCESS_RUN_TIMEOUT
from llumnix import envs as llumnix_envs
from llumnix.scaler import Scaler


logger = init_logger(__name__)


def launch_ray_cluster(port: int) -> subprocess.CompletedProcess:
    head_node_ip = llumnix_envs.HEAD_NODE_IP
    node_ip_address = get_ip_address()
    try:
        # Stop the existing ray processes on the node first.
        subprocess.run(['ray', 'stop'], check=True, text=True, capture_output=True, timeout=SUBPROCESS_RUN_TIMEOUT)
    except Exception as e: # pylint: disable=broad-except
        if isinstance(e, subprocess.CalledProcessError):
            logger.error("'ray stop' failed with: \n{}".format(e.stderr))
        else:
            logger.error("'ray stop' failed, unexpected exeption: {}.".format(e))
        sys.exit(1)
    # Need to specify the head node ip through environment variable currently.
    if head_node_ip is None:
        logger.error("Environment variable 'HEAD_NODE_IP' should be set for ray cluster launch.")
        sys.exit(1)
    ray_start_command = None
    if llumnix_envs.HEAD_NODE:
        ray_start_command = f"ray start --head --node-ip-address={node_ip_address} --port={port}"
        try:
            result = subprocess.run(['ray', 'start', '--head', f'--port={port}'],
                                    check=True, text=True, capture_output=True, timeout=SUBPROCESS_RUN_TIMEOUT)
        except Exception as e: # pylint: disable=broad-except
            if isinstance(e, subprocess.CalledProcessError):
                logger.error("'{}' failed with: \n{}".format(ray_start_command, e.stderr))
            else:
                logger.error("'ray stop' failed, unexpected exeption: {}.".format(e))
            sys.exit(1)
    else:
        ray_start_command = f"ray start --address={head_node_ip}:{port} --node-ip-address={node_ip_address}"
        for attempt in range(MAX_RAY_RESTART_TIMES):
            try:
                # wait about 2 mins by default
                result = subprocess.run(['ray', 'start', f'--address={head_node_ip}:{port}'],
                                        check=True, text=True, capture_output=True, timeout=SUBPROCESS_RUN_TIMEOUT)
                break
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, subprocess.CalledProcessError):
                    if attempt < MAX_RAY_RESTART_TIMES:
                        logger.warning("Execute '{}' repeatedly until the head node starts.".format(ray_start_command))
                        time.sleep(RAY_RESTART_INTERVAL)
                    else:
                        logger.error("'{}' failed after {} attempts with: \n{}".format(ray_start_command, attempt, e.stderr))
                        sys.exit(1)
                else:
                    logger.error("'ray stop' failed, unexpected exeption: {}.".format(e))
                sys.exit(1)
    logger.info("'{}' succeeed with: \n{}".format(ray_start_command, result.stdout))
    return result

def connect_to_ray_cluster(head_node_ip: str = None,
                           port: int = None,
                           namespace: str ="llumnix",
                           log_to_driver: bool=True) -> None:
    # env_vars of runtime_env can only set once in ray.init or ray actor initialization, otherwise will get ray error.
    if head_node_ip is not None and port is not None:
        ray.init(address=f"{head_node_ip}:{port}", ignore_reinit_error=True, namespace=namespace, log_to_driver=log_to_driver,
                 runtime_env={"env_vars": get_llumnix_env_vars()})
    else:
        ray.init(ignore_reinit_error=True, namespace=namespace, log_to_driver=log_to_driver,
                 runtime_env={"env_vars": get_llumnix_env_vars()})

def setup_ray_cluster(entrypoints_args) -> None:
    if entrypoints_args.launch_ray_cluster:
        launch_ray_cluster(entrypoints_args.ray_cluster_port)
    connect_to_ray_cluster(head_node_ip=os.getenv('HEAD_NODE_IP'),
                           port=entrypoints_args.ray_cluster_port,
                           namespace="llumnix",
                           log_to_driver=not entrypoints_args.disable_log_to_driver)

def init_scaler(
    manager_args: ManagerArgs,
    instance_args: InstanceArgs,
    entrypoints_args: EntrypointsArgs,
    engine_args: LlumnixEngineArgs,
    launch_args: LaunchArgs,
) -> Scaler:
    # Only one instance create the manager actor, the other instances get the existing manager actor through ray.
    try:
        scaler = Scaler.from_args(
            entrypoints_args=entrypoints_args,
            manager_args=manager_args,
            instance_args=instance_args,
            engine_args=engine_args,
            launch_args=launch_args,
        )
        logger.info("Init Scaler on current node.")
    except ValueError:
        scaler = ray.get_actor(get_scaler_name(), namespace='llumnix')
        logger.info("Get existing Scaler.")
    return scaler

def init_llumnix_components(entrypoints_args: EntrypointsArgs,
                            manager_args: ManagerArgs,
                            instance_args: InstanceArgs,
                            engine_args: LlumnixEngineArgs,
                            launch_args: LaunchArgs,
                            ) -> Tuple[Manager, List[str], List[Llumlet], QueueServerBase]:
    scaler: Scaler = init_scaler(manager_args, instance_args, entrypoints_args, engine_args, launch_args)
    ray.get(scaler.is_ready.remote())
    manager: Manager = ray.get_actor(get_manager_name(), namespace='llumnix')

    request_output_queue_type: QueueType = QueueType(entrypoints_args.request_output_queue_type)
    node_id = ray.get_runtime_context().get_node_id()
    instance_ids, instances = ray_get_with_timeout(
        scaler.init_instances.remote(request_output_queue_type, instance_args, engine_args, node_id)
    )

    available_instance_ids = []
    available_instances = []
    for instance_id, instance in zip(instance_ids, instances):
        try:
            ray.get(instance.is_ready.remote(), timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT))
            available_instance_ids.append(instance_id)
            available_instances.append(instance)
        # pylint: disable=broad-except
        except Exception as e:
            log_instance_exception(e, instance_id, "init_llumnix_components")
            ray_get_with_timeout(manager.scale_down.remote(instance_id))

    if len(available_instance_ids) > 0:
        logger.info("Init Llumnix components done, {} instances are ready, instance_ids: {}."
                    .format(len(available_instance_ids), available_instance_ids))

    ip = get_ip_address()
    if request_output_queue_type == QueueType.RAYQUEUE:
        # Init rayqueue in manager to ensure the job id of all actors are the same as manager.
        # We found that when the job id of rayqueue is inherited from driver process, it may raise job id unequal error sometimes.
        request_output_queue = ray_get_with_timeout(scaler.init_request_output_queue_server.remote(ip, request_output_queue_type))
    else:
        # zmq context cannot be serialized, so init zmq queue server in driver.
        request_output_queue = init_request_output_queue_server(ip, request_output_queue_type)

    return scaler, manager, available_instance_ids, available_instances, request_output_queue

def setup_entrypoints_context(entrypoints_args, scaler, manager, instance_ids, instances,
                              request_output_queue, server=None) -> EntrypointsContext:
    instances_dict: Dict[str, Llumlet] = {}
    for idx, ins_id in enumerate(instance_ids):
        instances_dict[ins_id] = instances[idx]

    server_id = random_uuid()
    ip = get_ip_address()
    port = request_output_queue.port
    server_info = ServerInfo(
        server_id,
        QueueType(entrypoints_args.request_output_queue_type),
        request_output_queue,
        ip,
        port,
    )

    log_requests = not entrypoints_args.disable_log_requests_server
    entrypoints_context = EntrypointsContext(
        scaler,
        manager,
        instances_dict,
        request_output_queue,
        server,
        server_info,
        log_requests,
    )

    return entrypoints_context

def _setup_llumnix_local(entrypoints_args, manager_args, instance_args, engine_args, launch_args) -> EntrypointsContext:
    scaler, manager, instance_ids, instances, request_output_queue = \
        init_llumnix_components(entrypoints_args, manager_args, instance_args, engine_args, launch_args)

    return setup_entrypoints_context(entrypoints_args, scaler, manager, instance_ids, instances, request_output_queue)

def _setup_llumnix_global(entrypoints_args, manager_args, instance_args, engine_args, launch_args) -> None:
    _ = init_scaler(manager_args, instance_args, entrypoints_args, engine_args, launch_args)

def setup_llumnix(entrypoints_args, manager_args, instance_args, engine_args, launch_args) -> Optional[EntrypointsContext]:
    if launch_args.launch_mode == LaunchMode.LOCAL:
        return _setup_llumnix_local(entrypoints_args, manager_args, instance_args, engine_args, launch_args)

    return _setup_llumnix_global(entrypoints_args, manager_args, instance_args, engine_args, launch_args)
