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
from typing import List, Optional, Tuple
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
# pylint: disable=unused-import
from ray.util.placement_group import PlacementGroup

from vllm.executor.gpu_executor import GPUExecutor
from vllm.executor.ray_gpu_executor import RayGPUExecutor, RayWorkerWrapper, get_distributed_init_method,\
                                           get_ip, get_vllm_instance_id, get_open_port

from vllm import envs
from vllm.sequence import Logprob, SequenceOutput, SequenceGroupOutput, SamplerOutput, ExecuteModelRequest
from vllm.config import _GB

from llumnix.config import MigrationConfig
from llumnix.logger import init_logger
from llumnix.backends.vllm.utils import get_cache_block_size
from llumnix.backends.profiling import LatencyMemData, SimCacheConfig, model_prefill, model_decode, _pad_to_alignment

logger = init_logger(__name__)

class LlumnixRayGPUExecutor(RayGPUExecutor):
    node_id: str = None
    migration_config: MigrationConfig = None

    def _init_workers_ray(self, placement_group: "PlacementGroup",
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
        node_id = self.node_id
        for rank in range(self.parallel_config.world_size):
            if placement_group:
                bundle = placement_group.bundle_specs[rank+1]
                if not bundle.get("GPU", 0):
                    raise Exception("GPU resource cannot be 0.")
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True,
                )
            else:
                scheduling_strategy = NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
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

    def execute_model(self, *args, **kwargs):
        t0 = time.time()
        outputs = super().execute_model(*args, **kwargs)
        t1 = time.time()
        self.last_inference_latency = (t1 - t0) * 1000
        return outputs

class SimGPUExecutor(GPUExecutor):
    latency_mem: LatencyMemData = None
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_inference_latency = 0
        self.migration_bandwidth = self.latency_mem.migration_bandwidth
        # TODO(ZeldaHuang): add swap bandwidth

        self.cache_block_size = get_cache_block_size(
            self.cache_config.block_size, self.model_config, self.parallel_config)
        self.cache_block_size /= _GB
        self.sim_cache_config = SimCacheConfig(self.cache_config.gpu_memory_utilization,
                                               self.cache_config.block_size,
                                               self.scheduler_config.max_num_batched_tokens)

    def _init_executor(self) -> None:
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_gpu_blocks = self.latency_mem.cache_dict.get(self.sim_cache_config, 880)
        num_cpu_blocks = 2048
        return (num_gpu_blocks, num_cpu_blocks)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks, num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        prefill_seq_len = 0
        decode_seq_len = 0
        decode_bs = 0
        for meta_data in execute_model_req.seq_group_metadata_list:
            if meta_data.is_prompt:
                prefill_seq_len += meta_data.token_chunk_size
            else:
                decode_bs += meta_data.token_chunk_size
                decode_seq_len += list(meta_data.seq_data.values())[0].get_len()
        decode_bs = _pad_to_alignment(decode_bs, 8)
        latency = 0
        if prefill_seq_len:
            latency += self.latency_mem.prefill_latency[prefill_seq_len][0] if prefill_seq_len in self.latency_mem.prefill_latency \
                       else model_prefill(prefill_seq_len, *self.latency_mem.prefill_model_params)
        if decode_bs:
            decode_meta_data = (decode_bs, decode_seq_len)
            latency += self.latency_mem.decode_latency[decode_meta_data][0] if decode_meta_data in self.latency_mem.decode_latency \
                       else model_decode((decode_bs, decode_seq_len), *self.latency_mem.decode_model_params)
        time.sleep(latency/1000)
        sampler_outputs = []
        for meta_data in execute_model_req.seq_group_metadata_list:
            samples = []
            for seq_id in meta_data.seq_data.keys():
                dummy_sample_output = SequenceOutput(seq_id, 20, {20: Logprob(1.0)})
                samples.append(dummy_sample_output)
            if samples:
                output = SequenceGroupOutput(samples, None)
                sampler_outputs.append(output)
        return [SamplerOutput(outputs=sampler_outputs)]

    def send_blocks(self, blocks_len) -> None:
        migration_latency = (self.cache_block_size * blocks_len) / self.migration_bandwidth
        time.sleep(migration_latency)
