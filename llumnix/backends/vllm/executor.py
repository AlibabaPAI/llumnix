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
from typing import List, Optional
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# pylint: disable=unused-import
from ray.util.placement_group import PlacementGroup

from vllm.executor.executor_base import ExecutorBase
from vllm.executor.ray_gpu_executor import RayGPUExecutor, RayGPUExecutorAsync, RayWorkerWrapper,\
                                           get_distributed_init_method, get_ip, get_vllm_instance_id, get_open_port

from vllm import envs
from vllm.sequence import Logprob, SequenceOutput, SequenceGroupOutput, SamplerOutput, ExecuteModelRequest
from vllm.config import _GB

from llumnix.internal_config import MigrationConfig
from llumnix.logging.logger import init_logger
from llumnix.backends.vllm.utils import get_cache_block_size
from llumnix.backends.profiling import LatencyMemData, SimCacheConfig, model_prefill, model_decode, _pad_to_alignment

logger = init_logger(__name__)


class LlumnixRayGPUExecutor(RayGPUExecutorAsync):
    migration_config: MigrationConfig = None

    def _init_workers_ray(self, placement_group: PlacementGroup,
                          **ray_remote_kwargs):
        self.last_inference_latency = 0
        if self.parallel_config.tensor_parallel_size == 1:
            # For single GPU case, we use a ray worker with constrained memory.
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            # Otherwise, the ray workers are allocated with a full GPU.
            num_gpus = 1

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        # Create the workers.
        driver_ip = get_ip()
        for rank in range(self.parallel_config.world_size):
            bundle = placement_group.bundle_specs[rank + 1]
            if not bundle.get("GPU", 0):
                raise Exception("GPU resource cannot be 0.")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
            )
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                max_concurrency=2,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(
                worker_module_name="llumnix.backends.vllm.worker",
                worker_class_name="MigrationWorker",
                trust_remote_code=self.model_config.trust_remote_code,
            )
            worker_ip = ray.get(worker.get_node_ip.remote())
            if worker_ip == driver_ip and self.driver_dummy_worker is None:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker
                self.driver_worker = RayWorkerWrapper(
                    worker_module_name="llumnix.backends.vllm.worker",
                    worker_class_name="MigrationWorker",
                    trust_remote_code=self.model_config.trust_remote_code,
                )
            else:
                # Else, added to the list of workers.
                self.workers.append(worker)

        if self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids",
                                                    use_dummy_driver=True)

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

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
        }, ) for (node_id, _) in worker_node_and_gpu_ids]
        self._run_workers("update_environment_variables",
                          all_args=all_args_to_update_environment_variables)

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
        self._run_workers("reserve_memory_for_migration",
                          migration_config=self.migration_config,
                          model_config=self.model_config,
                          cache_config=self.cache_config,
                          parallel_config=self.parallel_config)

    async def execute_model_async(self, *args, **kwargs):
        t0 = time.time()
        outputs = await super().execute_model_async(*args, **kwargs)
        t1 = time.time()
        self.last_inference_latency = (t1 - t0) * 1000
        return outputs
