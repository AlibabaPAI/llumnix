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

import time
from collections import defaultdict
from itertools import islice, repeat
from typing import Callable, Dict, List, Optional, Tuple, Type, Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# pylint: disable=unused-import
from ray.util.placement_group import PlacementGroup

from vllm.executor.ray_gpu_executor import (RayGPUExecutorAsync, RayWorkerWrapper, envs,
                                            get_ip, get_vllm_instance_id, get_distributed_init_method, get_open_port)
from vllm.worker.worker_base import WorkerBase
from vllm.sequence import ExecuteModelRequest
from vllm.model_executor.layers.sampler import SamplerOutput

from llumnix.internal_config import MigrationConfig
from llumnix.logging.logger import init_logger
from llumnix.utils import random_uuid, ray_get_with_timeout
from llumnix.constants import NUM_GPUS_VLLM_GPU_ACTOR
import llumnix.envs as llumnix_envs

logger = init_logger(__name__)


class LlumnixRayGPUExecutor(RayGPUExecutorAsync):
    instance_id: str = None
    migration_config: MigrationConfig = None
    last_inference_latency: int = 0

    def _init_workers_ray(self, placement_group: PlacementGroup,
                          **ray_remote_kwargs):
        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: List[List[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        logger.info("use_ray_spmd_worker: %s", self.use_ray_spmd_worker)

        # Create the workers.
        driver_ip = get_ip()
        worker_wrapper_kwargs = self._get_worker_wrapper_args()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if bundle.get("PREFILL_GPU") or bundle.get("DECODE_GPU"):
                continue
            if not bundle.get("GPU", 0):
                raise Exception("GPU resource cannot be 0.")
            num_gpus = NUM_GPUS_VLLM_GPU_ACTOR if bundle_id == 0 else 1
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id
            )
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                max_concurrency=2,
                name=f"worker_{self.instance_id}_{random_uuid()}",
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(**worker_wrapper_kwargs)

            if self.use_ray_spmd_worker:
                self.workers.append(worker)
            else:
                worker_ip = ray_get_with_timeout(worker.get_node_ip.remote())
                if worker_ip == driver_ip and self.driver_dummy_worker is None:
                    # If the worker is on the same node as the driver, we use it
                    # as the resource holder for the driver process.
                    self.driver_dummy_worker = worker
                    self.driver_worker = RayWorkerWrapper(
                        **worker_wrapper_kwargs)
                else:
                    # Else, added to the list of workers.
                    self.workers.append(worker)

        logger.debug("workers: %s", self.workers)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)
        if not self.use_ray_spmd_worker and self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        worker_ips = [
            ray_get_with_timeout(worker.get_node_ip.remote())  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(worker):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = ray_get_with_timeout(worker.get_node_ip.remote())
            return (ip != driver_ip, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        self.workers = sorted(self.workers, key=sort_by_driver_then_worker_ip)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                                                    use_dummy_driver=True)

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
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
                f"{n_ips} unique IP addresses {all_ips}. please check the cause your"
                " network configuration. If you set `VLLM_HOST_IP` or "
                "`HOST_IP` environment variable, make sure it is unique for"
                " each node.")

        # pylint: disable=invalid-name
        VLLM_INSTANCE_ID = get_vllm_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [({
            "CUDA_VISIBLE_DEVICES":
            ",".join(map(str, node_gpus[node_id])),
            "VLLM_INSTANCE_ID":
            VLLM_INSTANCE_ID,
            "VLLM_TRACE_FUNCTION":
            str(envs.VLLM_TRACE_FUNCTION),
            **({
                "VLLM_ATTENTION_BACKEND": envs.VLLM_ATTENTION_BACKEND
            } if envs.VLLM_ATTENTION_BACKEND is not None else {})
        }, ) for (node_id, _) in worker_node_and_gpu_ids]

        self._env_vars_for_all_workers = (
            all_args_to_update_environment_variables)

        self._run_workers("update_environment_variables",
                          all_args=self._get_env_vars_to_be_updated())

        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        init_worker_all_kwargs = [
            self._get_worker_kwargs(
                local_rank=node_workers[node_id].index(rank),
                rank=rank,
                distributed_init_method=distributed_init_method,
            ) for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids)
        ]
        self._run_workers("init_worker", all_kwargs=init_worker_all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)
        if self.migration_config and self.migration_config.enable_migration:
            self._run_workers("reserve_memory_for_migration",
                              migration_config=self.migration_config,
                              model_config=self.model_config,
                              cache_config=self.cache_config,
                              parallel_config=self.parallel_config)

        if self.use_ray_spmd_worker:
            for pp_rank in range(self.parallel_config.pipeline_parallel_size):
                self.pp_tp_workers.append([])
                for tp_rank in range(
                        self.parallel_config.tensor_parallel_size):
                    # PP=2, TP=4
                    # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                    rank = (pp_rank * self.parallel_config.tensor_parallel_size
                            ) + tp_rank
                    assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                    assert pp_rank < len(self.pp_tp_workers)
                    self.pp_tp_workers[pp_rank].append(self.workers[rank])

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[RayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[RayWorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for index, worker in enumerate(self.workers):
            # The driver worker is rank 0 and not in self.workers.
            rank = index + 1
            if rank % self.parallel_config.tensor_parallel_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)

    # pylint: disable=arguments-differ
    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        use_dummy_driver: bool = False,
        max_concurrent_workers: Optional[int] = None,
        timeout: float = float(llumnix_envs.INSTANCE_READY_TIMEOUT),
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """
        if self.use_ray_spmd_worker:
            assert not async_run_tensor_parallel_workers_only, (
                "async_run_tensor_parallel_workers_only is not supported for "
                "spmd mode.")

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        count = len(self.workers) if not \
            async_run_tensor_parallel_workers_only \
            else len(self.non_driver_workers)
        # If using SPMD worker, all workers are the same, so we should execute
        # the args on all workers. Otherwise, we skip the first worker's args
        # because those args will go to the driver worker.
        first_worker_args_index: int = 0 if self.use_ray_spmd_worker else 1
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, first_worker_args_index, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, first_worker_args_index, None)

        # Start the ray workers first.
        ray_workers = self.workers
        if async_run_tensor_parallel_workers_only:
            ray_workers = self.non_driver_workers
        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(ray_workers, all_worker_args, all_worker_kwargs)
        ]

        # TODO(s5u13b): Check it.
        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return ray_worker_outputs

        driver_worker_output = []
        # In SPMD mode, the driver worker is the same as any other worker,
        # so we only explicitly execute on the driver worker if using a
        # non-SPMD worker class.
        if not self.use_ray_spmd_worker:
            driver_args = args if all_args is None else all_args[0]
            driver_kwargs = kwargs if all_kwargs is None else all_kwargs[0]

            # Start the driver worker after all the ray workers.
            if not use_dummy_driver:
                driver_worker_output = [
                    self.driver_worker.execute_method(method, *driver_args,
                                                      **driver_kwargs)
                ]
            else:
                assert self.driver_dummy_worker is not None
                driver_worker_output = [
                    ray.get(
                        self.driver_dummy_worker.execute_method.remote(
                            method, *driver_args, **driver_kwargs),
                        timeout=timeout)
                ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs, timeout=timeout)

        return driver_worker_output + ray_worker_outputs

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        assert not (self.scheduler_config.is_multi_step or self.speculative_config), \
                "Llumnix does not support mult_step_worker and spec_decode_worker"
        worker_module_name = "llumnix.backends.vllm.worker"
        worker_class_name = "MigrationWorker"
        return (worker_module_name, worker_class_name, worker_class_fn)

    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        t0 = time.time()
        if not self.use_ray_spmd_worker:
            return await super().execute_model_async(execute_model_req)

        # pylint: disable=access-member-before-definition
        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=True)

        serialized_data = self.input_encoder.encode(execute_model_req)
        dag_future = await self.forward_dag.execute_async(serialized_data)
        outputs = await dag_future[0]
        request_outputs = self.output_decoder.decode(outputs)

        t1 = time.time()
        self.last_inference_latency = (t1 - t0) * 1000
        return request_outputs

    # _check_ray_adag_installation in vllm-v0.6.3.post1 requires ray version == 2.35.
    # _check_ray_adag_installation here (follows vllm-v0.7.2) requires ray version >= 2.40.
    # Llumnix requires ray version == 3.0.0.dev0, so we override the `_check_ray_adag_installation` method.
    def _check_ray_adag_installation(self):
        # pylint: disable=import-outside-toplevel
        import pkg_resources
        # pylint: disable=import-outside-toplevel
        from packaging import version

        required_version = version.parse("2.40")
        current_version = version.parse(
            pkg_resources.get_distribution("ray").version)
        if current_version < required_version:
            raise ValueError(f"Ray version {required_version} is "
                             f"required, but found {current_version}")

        # pylint: disable=import-outside-toplevel
        import importlib.util
        adag_spec = importlib.util.find_spec(
            "ray.experimental.compiled_dag_ref")
        if adag_spec is None:
            raise ValueError("Ray accelerated DAG is not installed. "
                             "Run `pip install ray[adag]` to install it.")

        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is None and envs.VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL:
            raise ValueError(
                "cupy is not installed but required since "
                "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL is set."
                "Run `pip install ray[adag]` and check cupy installation.")
