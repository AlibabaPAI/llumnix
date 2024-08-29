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
from typing import List, Tuple
import asyncio
import ray

from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from llumnix.llm_engine_manager import LLMEngineManager, MANAGER_ACTOR_NAME
from llumnix.llumlet.llumlet import Llumlet
from llumnix.backends.backend_interface import BackendType
from llumnix.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.arg_utils import EngineManagerArgs


logger = init_logger(__name__)

# TODO(s5u13b): Set the values through tests.
MAX_RESTARTS = 30
RESTART_INTERVALS = 1
MAX_TASK_RETRIES = 300
RETRIES_INTERVALS = 0.1


def get_ip_address():
    result = subprocess.run(['hostname', '-i'], stdout=subprocess.PIPE, check=True)
    ip_address = result.stdout.decode('utf-8').strip()
    return ip_address

def launch_ray_cluster(ray_cluster_port: int) -> subprocess.CompletedProcess:
    head_node_ip = os.getenv('HEAD_NODE_IP')
    node_ip_address = get_ip_address()
    try:
        # Stop the existing ray processes on the node first.
        subprocess.run(['ray', 'stop', '--force'], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.info("'ray stop' failed with: \n{}".format(e.stderr))
        sys.exit(1)
    # Need to specify the head node ip through environment variable currently.
    if head_node_ip is None:
        logger.info("Environment variable 'HEAD_NODE_IP' should be set for ray cluster launch.")
        sys.exit(1)
    ray_start_command = None
    if 'HEAD_NODE' in os.environ:
        ray_start_command = f"ray start --head --node-ip-address={node_ip_address} --port={ray_cluster_port}"
        try:
            result = subprocess.run(['ray', 'start', '--head', f'--port={ray_cluster_port}'], check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.info("'{}' failed with: \n{}".format(ray_start_command, e.stderr))
            sys.exit(1)
    else:
        ray_start_command = f"ray start --address={head_node_ip}:{ray_cluster_port} --node-ip-address={node_ip_address}"
        for attempt in range(MAX_RESTARTS):
            try:
                # wait about 2 mins by default
                result = subprocess.run(['ray', 'start', f'--address={head_node_ip}:{ray_cluster_port}'], check=True, text=True, capture_output=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt < MAX_RESTARTS:
                    print("Execute '{}' repeatedly until the head node starts...".format(ray_start_command))
                    time.sleep(RESTART_INTERVALS)
                else:
                    logger.info("'{}' failed after {} attempts with: \n{}".format(ray_start_command, attempt, e.stderr))
                    sys.exit(1)
    logger.info("'{}' succeeed with: \n{}".format(ray_start_command, result.stdout))
    return result

def connect_to_ray_cluster(port: int, namespace="llumnix") -> None:
    head_node_ip = os.getenv('HEAD_NODE_IP')
    ray.init(address=f"{head_node_ip}:{port}", ignore_reinit_error=True, namespace=namespace)

def is_gpu_available() -> bool:
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def retry_manager_method_sync(ray_call, method_name, *args, **kwargs):
    for attempt in range(MAX_TASK_RETRIES):
        try:
            ret = ray.get(ray_call(*args, **kwargs))
            break
        except ray.exceptions.RayActorError:
            if attempt < MAX_TASK_RETRIES - 1:
                logger.info("Manager is unavailable, sleep {}s, and retry {} again...".format(RETRIES_INTERVALS, method_name))
                time.sleep(RETRIES_INTERVALS)
            else:
                logger.info("After {} times retries, manager is still unavailable".format(MAX_TASK_RETRIES))
                raise
    return ret

async def retry_manager_method_async(ray_call, method_name, *args, **kwargs):
    for attempt in range(MAX_TASK_RETRIES):
        try:
            ret = await ray_call(*args, **kwargs)
            break
        except ray.exceptions.RayActorError:
            if attempt < MAX_TASK_RETRIES - 1:
                logger.info("Manager is unavailable, sleep {}s, and retry {} again...".format(RETRIES_INTERVALS, method_name))
                await asyncio.sleep(RETRIES_INTERVALS)
            else:
                logger.info("After {} times retries, manager is still unavailable".format(MAX_TASK_RETRIES))
                raise
    return ret

def init_manager(engine_manager_args: EngineManagerArgs) -> LLMEngineManager:
    # Only one instance create the manager actor, the other instances get the existing manager actor through ray.
    try:
        engine_manager = LLMEngineManager.from_args(engine_manager_args, None)
        logger.info("Init LLMEngineManager on current node")
    except ValueError:
        engine_manager = ray.get_actor(MANAGER_ACTOR_NAME, namespace='llumnix')
        logger.info("Get existing LLMEngineManager")
    return engine_manager

def init_llumlets(engine_manager_args: EngineManagerArgs,
                  engine_args,
                  node_id: str) -> Tuple[List[str], List[Llumlet]]:
    engine_config = engine_args.create_engine_config()
    parallel_config = engine_config.parallel_config
    instance_ids: List[str] = []
    llumlets: List[Llumlet] = []

    instance_ids = [random_uuid() for _ in range(engine_manager_args.initial_instances)]
    migration_configs = engine_manager_args.create_migration_config()

    for idx in range(engine_manager_args.initial_instances):
        instance_id = instance_ids[idx]
        if not engine_manager_args.profiling_result_file_path:
            llumlet = Llumlet.from_args(
                engine_manager_args.disable_fixed_node_init_instance,
                False,
                node_id,
                instance_id,
                BackendType.VLLM,
                parallel_config.world_size,
                migration_configs,
                engine_args,
            )
        else:
            llumlet = Llumlet.from_args(
                engine_manager_args.disable_fixed_node_init_instance,
                False,
                node_id,
                instance_id,
                BackendType.SIM_VLLM,
                parallel_config.world_size,
                migration_configs,
                engine_manager_args.profiling_result_file_path,
                engine_manager_args.gpu_type,
                engine_args,
            )
        llumlets.append(llumlet)
    return instance_ids, llumlets

def init_request_output_queue() -> RayQueue:
    # request_output_queue should be placed in the same node as the api server.
    request_output_queue = RayQueue(actor_options={
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,)
    })
    return request_output_queue

def init_llumnix_components(engine_manager_args: EngineManagerArgs,
                            engine_args,
                            node_id: str) -> Tuple[LLMEngineManager, List[Llumlet], RayQueue]:
    engine_manager = init_manager(engine_manager_args)
    if engine_manager_args.disable_init_instance_by_manager:
        instance_ids, llumlets = init_llumlets(engine_manager_args, engine_args, node_id)
    else:
        instance_ids, llumlets = retry_manager_method_sync(engine_manager.init_llumlets.remote, 'init_llumlets', engine_args, node_id)

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
        retry_manager_method_sync(engine_manager.scale_down.remote, 'scale_down', dead_instance_ids)

    if len(available_instance_ids) > 0:
        retry_manager_method_sync(engine_manager.scale_up.remote, 'scale_up',
                                  available_instance_ids, available_llumlets)
        logger.info("Init Llumnix components done, {} instances are ready, instance_ids: {}."
                    .format(len(available_instance_ids), available_instance_ids))

    request_output_queue = init_request_output_queue()

    return engine_manager, available_instance_ids, available_llumlets, request_output_queue
