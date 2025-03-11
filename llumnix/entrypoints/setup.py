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
import traceback
import sys
import os
import time
from typing import Dict, Optional, List, Tuple
import ray

from llumnix.manager import Manager
from llumnix.llumlet.llumlet import Llumlet
from llumnix.logging.logger import init_logger
from llumnix.utils import random_uuid, get_manager_name
from llumnix.arg_utils import ManagerArgs, EntrypointsArgs, LaunchArgs, InstanceArgs
from llumnix.queue.queue_type import QueueType
from llumnix.server_info import ServerInfo
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.entrypoints.utils import (LaunchMode, EntrypointsContext, get_ip_address,
                                       retry_manager_method_sync)
from llumnix.backends.backend_interface import BackendType
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.constants import MAX_RAY_RESTART_TIMES, RAY_RESTART_INTERVAL


logger = init_logger(__name__)


def launch_ray_cluster(port: int) -> subprocess.CompletedProcess:
    head_node_ip = os.getenv('HEAD_NODE_IP')
    node_ip_address = get_ip_address()
    try:
        # Stop the existing ray processes on the node first.
        subprocess.run(['ray', 'stop'], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error("'ray stop' failed with: \n{}".format(e.stderr))
        sys.exit(1)
    # Need to specify the head node ip through environment variable currently.
    if head_node_ip is None:
        logger.error("Environment variable 'HEAD_NODE_IP' should be set for ray cluster launch.")
        sys.exit(1)
    ray_start_command = None
    if 'HEAD_NODE' in os.environ:
        ray_start_command = f"ray start --head --node-ip-address={node_ip_address} --port={port}"
        try:
            result = subprocess.run(['ray', 'start', '--head', f'--port={port}'], check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error("'{}' failed with: \n{}".format(ray_start_command, e.stderr))
            sys.exit(1)
    else:
        ray_start_command = f"ray start --address={head_node_ip}:{port} --node-ip-address={node_ip_address}"
        for attempt in range(MAX_RAY_RESTART_TIMES):
            try:
                # wait about 2 mins by default
                result = subprocess.run(['ray', 'start', f'--address={head_node_ip}:{port}'], check=True, text=True, capture_output=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt < MAX_RAY_RESTART_TIMES:
                    logger.warning("Execute '{}' repeatedly until the head node starts.".format(ray_start_command))
                    time.sleep(RAY_RESTART_INTERVAL)
                else:
                    logger.error("'{}' failed after {} attempts with: \n{}".format(ray_start_command, attempt, e.stderr))
                    sys.exit(1)
    logger.info("'{}' succeeed with: \n{}".format(ray_start_command, result.stdout))
    return result

def connect_to_ray_cluster(head_node_ip: str = None,
                           port: int = None,
                           namespace: str ="llumnix",
                           log_to_driver: bool=True) -> None:
    if head_node_ip is not None and port is not None:
        ray.init(address=f"{head_node_ip}:{port}", ignore_reinit_error=True, namespace=namespace, log_to_driver=log_to_driver)
    else:
        ray.init(ignore_reinit_error=True, namespace=namespace, log_to_driver=log_to_driver)

def setup_ray_cluster(entrypoints_args) -> None:
    if entrypoints_args.launch_ray_cluster:
        launch_ray_cluster(entrypoints_args.ray_cluster_port)
    connect_to_ray_cluster(head_node_ip=os.getenv('HEAD_NODE_IP'),
                           port=entrypoints_args.ray_cluster_port,
                           namespace="llumnix",
                           log_to_driver=not entrypoints_args.disable_log_to_driver)

def init_manager(manager_args: ManagerArgs,
                 instance_args: InstanceArgs = None,
                 entrypoints_args: EntrypointsArgs = None,
                 engine_args = None,
                 launch_args: LaunchArgs = None,
                 ) -> Manager:
    # Only one instance create the manager actor, the other instances get the existing manager actor through ray.
    try:
        manager = Manager.from_args(
            entrypoints_args=entrypoints_args,
            manager_args=manager_args,
            instance_args=instance_args,
            engine_args=engine_args,
            launch_args=launch_args)
        logger.info("Init Manager on current node.")
    except ValueError:
        manager = ray.get_actor(get_manager_name(), namespace='llumnix')
        logger.info("Get existing Manager.")
    return manager

def init_llumnix_components(entrypoints_args: EntrypointsArgs,
                            manager_args: ManagerArgs,
                            instance_args: InstanceArgs,
                            engine_args,
                            launch_args: LaunchArgs,
                            ) -> Tuple[Manager, List[str], List[Llumlet], QueueServerBase]:
    manager = init_manager(manager_args)

    backend_type: BackendType = launch_args.backend_type
    request_output_queue_type: QueueType = QueueType(entrypoints_args.request_output_queue_type)
    instance_ids, instances = retry_manager_method_sync(
        manager.init_instances.remote, 'init_instances', request_output_queue_type,
        backend_type, instance_args, engine_args)

    available_instance_ids = []
    available_instances = []
    for instance_id, instance in zip(instance_ids, instances):
        try:
            ray.get(instance.is_ready.remote())
            available_instance_ids.append(instance_id)
            available_instances.append(instance)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Instance {} is dead.".format(instance_id))
            logger.error("Unexpected exception occurs: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))
            retry_manager_method_sync(manager.scale_down.remote, 'scale_down', instance_id)

    ip = get_ip_address()
    request_output_queue_port: str = entrypoints_args.request_output_queue_port
    request_output_queue = init_request_output_queue_server(ip, request_output_queue_port, request_output_queue_type)

    return manager, available_instance_ids, available_instances, request_output_queue

def setup_entrypoints_context(entrypoints_args, manager, instance_ids, instances, request_output_queue) -> EntrypointsContext:
    instances_dict: Dict[str, Llumlet] = {}
    for idx, ins_id in enumerate(instance_ids):
        instances_dict[ins_id] = instances[idx]

    server_id = random_uuid()
    ip = get_ip_address()
    server_info = ServerInfo(server_id,
                             QueueType(entrypoints_args.request_output_queue_type),
                             request_output_queue,
                             ip,
                             entrypoints_args.request_output_queue_port)

    log_requests = not entrypoints_args.disable_log_requests_server
    log_request_timestamps = entrypoints_args.log_request_timestamps
    entrypoints_context = EntrypointsContext(manager,
                                             instances_dict,
                                             request_output_queue,
                                             server_info,
                                             log_requests,
                                             log_request_timestamps)
    return entrypoints_context

def _setup_llumnix_local(entrypoints_args, manager_args, instance_args, engine_args, launch_args) -> EntrypointsContext:
    manager, instance_ids, instances, request_output_queue = \
        init_llumnix_components(entrypoints_args, manager_args, instance_args, engine_args, launch_args)

    return setup_entrypoints_context(entrypoints_args, manager, instance_ids, instances, request_output_queue)

def _setup_llumnix_global(entrypoints_args, manager_args, instance_args, engine_args, launch_args) -> None:
    _ = init_manager(manager_args, instance_args, entrypoints_args, engine_args, launch_args)

def setup_llumnix(entrypoints_args, manager_args, instance_args, engine_args, launch_args) -> Optional[EntrypointsContext]:
    if launch_args.launch_mode == LaunchMode.LOCAL:
        return _setup_llumnix_local(entrypoints_args, manager_args, instance_args, engine_args, launch_args)

    return _setup_llumnix_global(entrypoints_args, manager_args, instance_args, engine_args, launch_args)
