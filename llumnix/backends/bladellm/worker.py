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

import threading
import os
import time
from multiprocessing import Process, set_start_method
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from collections import defaultdict
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray.actor

from blade_llm.service.args import ServingArgs
from blade_llm.service.worker import worker_main, WorkerProcesses

from llumnix.logging.logger import init_logger
from llumnix.utils import (get_ip_address, update_environment_variables,
                           ray_get_with_timeout)
from llumnix.constants import RAY_REMOTE_CALL_TIMEOUT
from llumnix.utils import random_uuid
from llumnix.ray_utils import log_actor_ray_info
from llumnix.constants import NUM_GPUS_BLADELLM_GPU_ACTOR

logger = init_logger(__name__)


@dataclass
class RayWorkerMetaData:
    """
    Metadata for a Ray worker.
    The order of ray worker creation can be random,
    and we need to reset the rank after creating all workers.
    """
    worker: ray.actor.ActorHandle
    created_rank: int
    adjusted_rank: int = -1
    ip: str = ""


class WorkerProcessActor:
    def __init__(self, rank: int, instance_id: str, worker_ray_name: str):
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.rank = rank
        self.instance_id = instance_id
        self.worker_ray_name = worker_ray_name
        self.grpc_migration_server_port = None

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]}, rank={self.rank})"

    def init_worker_process(self, rank: int, serving_args: ServingArgs, args, kwargs) -> None:
        self._setup_dist_node_rank(serving_args)
        logger.debug("Init worker process, rank: {}, dist_inference_options: {}".format(
            self.rank, serving_args.dist_inference_options))
        set_start_method("spawn", force=True)
        kwargs["worker_ray_name"] = self.worker_ray_name
        p = Process(
                target=worker_main,
                args=(rank, serving_args, *args),
                kwargs=kwargs,
                name=f"Worker_{self.rank}",
                daemon=True,
            )
        p.start()
        self._proc = p

    def get_node_ip(self) -> str:
        return get_ip_address()

    def set_worker_port(self, port):
        logger.debug("Worker process actor (rank: {}) set grpc migration server port: {}.".format(self.rank, port))
        self.grpc_migration_server_port = port

    def get_grpc_migration_back_port(self) -> Tuple[int, int]:
        assert self.grpc_migration_server_port is not None
        return self.grpc_migration_server_port

    def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_gpu_ids()
        return node_id, gpu_ids

    def get_worker_process_pid(self) -> int:
        return self._proc.pid

    def kill_worker_process(self) -> None:
        self._proc.kill()

    def stop_worker_process(self) -> None:
        self._proc.terminate()
        self._proc.join()

    def adjust_rank(self, rank_mapping: Dict[int, int]) -> None:
        """
        Adjust the rank based on the given mapping.
        It is only used during the initialization of the executor,
        to adjust the rank of workers after we create all workers.
        """
        if self.rank in rank_mapping:
            self.rank = rank_mapping[self.rank]

    def update_environment_variables(self, envs_list: List[Dict[str, str]]) -> None:
        envs = envs_list[self.rank]
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    def execute_method(self, method, *args, **kwargs):
        try:
            executor = getattr(self, method)
            return executor(*args, **kwargs)
        except Exception as e:
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    def _setup_dist_node_rank(self, serving_args: ServingArgs):
        tp_size_per_node = serving_args.tensor_parallel_size * serving_args.pipeline_parallel_size \
            // serving_args.dist_inference_options.nnodes
        serving_args.dist_inference_options.node_rank = self.rank // tp_size_per_node


class WorkerProcessesRay(WorkerProcesses):
    def __init__(self, placement_group: PlacementGroup, serving_args: ServingArgs, instance_id: str, *args, **kwargs):
        super().__init__(serving_args, instance_id, *args, **kwargs)
        self.placement_group = placement_group
        self.instance_id = instance_id
        self.workers: List[WorkerProcessActor] = [None] * self._args.tensor_parallel_size
        self.worker_node_and_gpu_ids: List[Tuple[int, List[int]]] = [(None, None)] * len(self.workers)
        self.worker_pids: List[int] = [None] * len(self.workers)

    def _worker_watch_dog(self):
        while self._running:
            time.sleep(5)
            has_dead = False
            for rank, worker in enumerate(self.workers):
                try:
                    has_dead = not ray_get_with_timeout(worker.is_alive.remote())
                except: # pylint: disable=bare-except
                    has_dead = True
                if has_dead:
                    logger.exception("Worker {} is dead (pid {}, node_id: {}, gpu_ids: {}).".format(
                        rank, self.worker_pids[rank],
                        self.worker_node_and_gpu_ids[rank][0], self.worker_node_and_gpu_ids[rank][1]))
            if self.remote_watch_dog:
                has_dead = has_dead or self.remote_watch_dog.worker_watch_dog()
            if has_dead:
                self._exit_server()
        logger.info("watch dog exit.")

    def _exit_server(self):
        for rank, worker in enumerate(self.workers):
            try:
                if ray_get_with_timeout(worker.is_alive.remote()):
                    logger.critical("Kill alive worker {} (pid {}, node_id: {}, gpu_ids: {}).".format(
                        rank, self.worker_pids[rank],
                        self.worker_node_and_gpu_ids[rank][0], self.worker_node_and_gpu_ids[rank][1]))
                    ray_get_with_timeout(worker.kill_worker_process.remote())
            except: # pylint: disable=bare-except
                pass
            ray.kill(worker)
        # Not using sys.exit() since it do cleanup work which may still hang server process.
        logger.critical("Server {} exit.".format(os.getpid()))
        os._exit(255) # pylint: disable=protected-access

    def _init_workers_ray(self, placement_group: PlacementGroup, instance_id: str):
        # Create the workers.
        tp_size = self._args.tensor_parallel_size
        workers: List[WorkerProcessActor] = []
        worker_metadata: List[RayWorkerMetaData] = []
        driver_ip = get_ip_address()
        # TODO(s5u13b): Not support pipeline parallelism currently.
        # Suppose that all bundles are created for tp workers.
        for bundle_id in range(tp_size):
            bundle = placement_group.bundle_specs[bundle_id]
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            num_gpus = NUM_GPUS_BLADELLM_GPU_ACTOR if bundle_id == 0 else 1
            worker_ray_name = f"worker_{self.instance_id}_"+random_uuid()
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                name=worker_ray_name,
                namespace='llumnix',
                scheduling_strategy=scheduling_strategy
            )(WorkerProcessActor).remote(rank=bundle_id, instance_id=instance_id, worker_ray_name=worker_ray_name)
            workers.append(worker)
            worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=bundle_id))
        self.workers = workers

        # Longer timeout because the first remote call to worker requires waiting for the initialization of the worker.
        worker_ips = self._run_workers("get_node_ip", timeout=600.0)

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (bladellm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item.ip
            return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be close to each other.
        sorted_worker_metadata = sorted(worker_metadata, key=sort_by_driver_then_worker_ip)
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i
        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank
            for item in sorted_worker_metadata
        }
        self._run_workers("adjust_rank", rerank_mapping)

        # Get the set of GPU IDs used on each node.
        self.worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids")
        self.worker_ips = self._run_workers("get_node_ip")

        logger.debug("worker_node_and_gpu_ids: {}".format(self.worker_node_and_gpu_ids))

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(self.worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP` or "
                "`HOST_IP` environment variable, make sure it is unique for"
                " each node.")

        # Set environment variables for workers.
        all_args_to_update_environment_variables = [{
            "CUDA_VISIBLE_DEVICES":
            ",".join(map(str, node_gpus[node_id]))
        } for (node_id, _) in self.worker_node_and_gpu_ids]

        # Environment variables to copy from driver to workers
        env_vars_to_copy = [
            'MASTER_ADDR',
            'MASTER_PORT',
            'ACCL_MAX_USER_MR_GB',
            'ACCL_SOFT_TX_DEPTH',
            'ACCL_SET_ERDMA',
            'BLLM_KVTRANS_CACHE_SHAPE'
        ]

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        # pylint: disable=self-assigning-variable
        all_args_to_update_environment_variables = (all_args_to_update_environment_variables)

        logger.debug(
            "Copying the following environment variables to workers: %s",
            [v for v in env_vars_to_copy if v in os.environ])

        self._run_workers("update_environment_variables",
                          all_args_to_update_environment_variables)

    def _spawn_workers_ray(self, serving_args: ServingArgs, args, kwargs):
        # Initialize the worker process inside worker process actor.
        ray_get_with_timeout(
            [worker.init_worker_process.remote(rank, serving_args, args, kwargs)
                for rank, worker in enumerate(self.workers)]
        )
        self.worker_pids = self._run_workers("get_worker_process_pid")

    def _run_workers(
            self,
            method: str,
            *args,
            timeout: float = RAY_REMOTE_CALL_TIMEOUT,
            **kwargs,
        ) -> Any:
        ray_worker_outputs = [
            worker.execute_method.remote(method, *args, **kwargs)
            for worker in self.workers
        ]
        return ray.get(ray_worker_outputs, timeout=timeout)

    def get_all_workers_grpc_migration_server_port(self):
        return self._run_workers("get_grpc_migration_back_port")

    def start(self):
        try:
            self._init_workers_ray(self.placement_group, self.instance_id)
            self._spawn_workers_ray(self._args, self.addition_args, self.addition_kwargs)
        except: # pylint: disable=bare-except
            self._exit_server()
        if self.remote_watch_dog:
            self.remote_watch_dog.start()
        self._watchdog = threading.Thread(target=self._worker_watch_dog, args=(), daemon=True, name="worker_watchdog")
        self._watchdog.start()

    def stop(self):
        self._running = False
        # Follow BladeLLM.
        # self._watchdog.join()
        for rank, worker in enumerate(self.workers):
            try:
                ray_get_with_timeout(worker.stop_worker_process.remote())
            # pylint: disable=broad-except
            except Exception as e:
                if isinstance(e, TimeoutError):
                    logger.error("Worker is hang (instance_id: {}, rank: {}), please check the cause.")
                else:
                    logger.exception("Failed to stop worker process (instance_id: {}, rank: {}), "
                                     "unexpected exception: {}.".format(self.instance_id, rank, e))
            try:
                ray.kill(worker)
            # pylint: disable=broad-except
            except Exception as e:
                logger.exception("Failed to kill worker (instance_id: {}, rank: {}), "
                                 "unexpected exception: {}.".format(self.instance_id, rank, e))
        if self.remote_watch_dog:
            self.remote_watch_dog.stop()
