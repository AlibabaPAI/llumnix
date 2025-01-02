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
from typing import Dict, Optional
import ray

from llumnix.manager import Manager
from llumnix.llumlet.llumlet import Llumlet
from llumnix.logger import init_logger
from llumnix.utils import random_uuid, MANAGER_NAME
from llumnix.arg_utils import ManagerArgs, EntrypointsArgs, DeploymentArgs
from llumnix.queue.queue_type import QueueType
from llumnix.server_info import ServerInfo
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.entrypoints.utils import (EntrypointsContext, get_ip_address,
                                       retry_manager_method_sync)
from llumnix.entrypoints.utils import DeploymentMode
from llumnix.backends.backend_interface import BackendType

MAX_RAY_RESTARTS = 5
RAY_RESTART_INTERVALS = 10

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
        for attempt in range(MAX_RAY_RESTARTS):
            try:
                # wait about 2 mins by default
                result = subprocess.run(['ray', 'start', f'--address={head_node_ip}:{port}'], check=True, text=True, capture_output=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt < MAX_RAY_RESTARTS:
                    logger.warning("execute '{}' repeatedly until the head node starts".format(ray_start_command))
                    time.sleep(RAY_RESTART_INTERVALS)
                else:
                    logger.error("'{}' failed after {} attempts with: \n{}".format(ray_start_command, attempt, e.stderr))
                    sys.exit(1)
    logger.info("'{}' succeeed with: \n{}".format(ray_start_command, result.stdout))
    return result

def connect_to_ray_cluster(head_node_ip: str = None, port: int = None, namespace="llumnix") -> None:
    if head_node_ip is not None and port is not None:
        ray.init(address=f"{head_node_ip}:{port}", ignore_reinit_error=True, namespace=namespace)
    else:
        ray.init(ignore_reinit_error=True, namespace=namespace)

def setup_ray_cluster(entrypoints_args) -> None:
    if entrypoints_args.launch_ray_cluster:
        launch_ray_cluster(entrypoints_args.ray_cluster_port)
    connect_to_ray_cluster(head_node_ip=os.getenv('HEAD_NODE_IP'), port=entrypoints_args.ray_cluster_port, namespace="llumnix")

def init_manager(manager_args: ManagerArgs,
                 entrypoints_args: EntrypointsArgs = None,
                 engine_args = None,
                 deployment_args: DeploymentArgs = None,
                 ) -> Manager:
    # Only one instance create the manager actor, the other instances get the existing manager actor through ray.
    try:
        manager = Manager.from_args(manager_args, entrypoints_args, engine_args, deployment_args)
        logger.info("Init Manager on current node.")
    except ValueError:
        manager = ray.get_actor(MANAGER_NAME, namespace='llumnix')
        logger.info("Get existing Manager.")
    return manager

def init_llumnix_components(manager_args: ManagerArgs,
                            engine_args,
                            request_output_queue_type: QueueType,
                            ip: str,
                            request_output_queue_port: str,
                            backend_type: BackendType,
                            *args,
                            **kwargs
                            ):
    manager = init_manager(manager_args)

    instance_ids, llumlets = retry_manager_method_sync(
        manager.init_llumlets.remote, 'init_llumlets', engine_args, request_output_queue_type, backend_type, *args, **kwargs)

    available_instance_ids = []
    dead_instance_ids = []
    available_llumlets = []
    ready_tasks = [llumlet.is_ready.remote() for llumlet in llumlets]
    for idx, task in enumerate(ready_tasks):
        try:
            ray.get(task)
            available_instance_ids.append(instance_ids[idx])
            available_llumlets.append(llumlets[idx])
        except ray.exceptions.RayActorError:
            dead_instance_ids.append(instance_ids[idx])
    if len(dead_instance_ids) > 0:
        retry_manager_method_sync(manager.scale_down.remote, 'scale_down', dead_instance_ids)
    if len(available_instance_ids) > 0:
        retry_manager_method_sync(manager.scale_up.remote, 'scale_up',
                                  available_instance_ids, available_llumlets)
        logger.info("Init Llumnix components done, {} instances are ready, instance_ids: {}."
                    .format(len(available_instance_ids), available_instance_ids))

    request_output_queue = init_request_output_queue_server(ip, request_output_queue_port, request_output_queue_type)

    return manager, available_instance_ids, available_llumlets, request_output_queue

def _setup_llumnix_local(manager_args, entrypoints_args, engine_args, deployment_args,
                         *args, **kwargs) -> EntrypointsContext:
    ip = get_ip_address()
    request_output_queue_type = entrypoints_args.request_output_queue_type
    request_output_queue_port = entrypoints_args.request_output_queue_port
    backend_type = deployment_args.backend_type

    manager, instance_ids, llumlets, request_output_queue = \
        init_llumnix_components(manager_args,
                                engine_args,
                                request_output_queue_type,
                                ip,
                                request_output_queue_port,
                                backend_type,
                                *args,
                                **kwargs)

    server_id = random_uuid()
    server_info = ServerInfo(server_id,
                             request_output_queue_type,
                             request_output_queue,
                             ip,
                             request_output_queue_port)

    instances: Dict[str, Llumlet] = {}
    for idx, ins_id in enumerate(instance_ids):
        instances[ins_id] = llumlets[idx]

    log_requests = not manager_args.disable_log_requests_manager
    log_request_timestamps = entrypoints_args.log_request_timestamps
    logger.info("log_requests: {}, log_request_timestamps: {}".format(log_requests, log_request_timestamps))

    entrypoints_context = EntrypointsContext(manager,
                                             instances,
                                             request_output_queue,
                                             server_info,
                                             deployment_args.deployment_mode,
                                             log_requests,
                                             log_request_timestamps)

    return entrypoints_context

def _setup_llumnix_global(manager_args, entrypoints_args, engine_args, deployment_args) -> None:
    _ = init_manager(manager_args, entrypoints_args, engine_args, deployment_args)

def setup_llumnix(manager_args, entrypoints_args, engine_args, deployment_args,
                  *args, **kwargs) -> Optional[EntrypointsContext]:
    if deployment_args.deployment_mode == DeploymentMode.LOCAL:
        return _setup_llumnix_local(manager_args, entrypoints_args, engine_args, deployment_args, *args, **kwargs)

    return _setup_llumnix_global(manager_args, entrypoints_args, engine_args, deployment_args)
