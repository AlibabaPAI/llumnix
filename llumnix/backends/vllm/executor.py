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
import asyncio

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Type
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# pylint: disable=unused-import
from ray.util.placement_group import PlacementGroup

from vllm.executor.executor_base import ExecutorBase
from vllm.model_executor.layers.sampler import SamplerOutput, CompletionSequenceGroupOutput
from vllm.executor.ray_gpu_executor import RayGPUExecutor, RayGPUExecutorAsync, RayWorkerWrapper, envs, \
                                           get_ip, get_vllm_instance_id, get_distributed_init_method, get_open_port
from vllm.worker.worker_base import WorkerBase

from vllm.sequence import Logprob, SequenceOutput, ExecuteModelRequest
from vllm.utils import GiB_bytes

from llumnix.internal_config import MigrationConfig
from llumnix.logger import init_logger
from llumnix.backends.vllm.utils import get_cache_block_size
from llumnix.backends.profiling import LatencyMemData, SimCacheConfig, model_prefill, model_decode, _pad_to_alignment

logger = init_logger(__name__)


class LlumnixRayGPUExecutor(RayGPUExecutorAsync):
    migration_config: MigrationConfig = None
    last_inference_latency:int = 0

    def _init_workers_ray(self, placement_group: PlacementGroup,
                          **ray_remote_kwargs):
        if (self.parallel_config.tensor_parallel_size == 1
                and self.parallel_config.pipeline_parallel_size == 1):
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
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(**worker_wrapper_kwargs)

            if self.use_ray_spmd_worker:
                self.workers.append(worker)
            else:
                worker_ip = ray.get(worker.get_node_ip.remote())
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
            ray.get(worker.get_node_ip.remote())  # type: ignore[attr-defined]
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
            ip = ray.get(worker.get_node_ip.remote())
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
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
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

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        assert not (self.scheduler_config.is_multi_step or self.speculative_config), \
                "Llumnix does not support mult_step_worker and spec_decode_worker"
        worker_module_name = "llumnix.backends.vllm.worker"
        worker_class_name = "MigrationWorker"
        return (worker_module_name, worker_class_name, worker_class_fn)

    async def execute_model_async(self, *args, **kwargs):
        t0 = time.time()
        outputs = await super().execute_model_async(*args, **kwargs)
        t1 = time.time()
        self.last_inference_latency = (t1 - t0) * 1000
        return outputs

class SimGPUExecutor(RayGPUExecutor):
    latency_mem: LatencyMemData = None
    def __init__(self, *args, **kwargs) -> None:
        RayGPUExecutor.__init__(self, *args, **kwargs)
        self.last_inference_latency = 0
        self.migration_bandwidth = self.latency_mem.migration_bandwidth
        # TODO(ZeldaHuang): add swap bandwidth

        self.cache_block_size = get_cache_block_size(
            self.cache_config.block_size, self.model_config, self.parallel_config)
        self.cache_block_size /= GiB_bytes
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

    async def execute_model_async(
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
        prefill_seq_len = _pad_to_alignment(prefill_seq_len, 8)
        latency = 0
        if prefill_seq_len:
            latency += self.latency_mem.prefill_latency[prefill_seq_len][0] if prefill_seq_len in self.latency_mem.prefill_latency \
                       else model_prefill(prefill_seq_len, *self.latency_mem.prefill_model_params)
        if decode_bs:
            decode_meta_data = (decode_bs, decode_seq_len)
            latency += self.latency_mem.decode_latency[decode_meta_data][0] if decode_meta_data in self.latency_mem.decode_latency \
                       else model_decode((decode_bs, decode_seq_len), *self.latency_mem.decode_model_params)
        await asyncio.sleep(latency/1000)
        sampler_outputs = []
        for meta_data in execute_model_req.seq_group_metadata_list:
            samples = []
            for seq_id in meta_data.seq_data.keys():
                dummy_sample_output = SequenceOutput(seq_id, 20, {20: Logprob(1.0)})
                samples.append(dummy_sample_output)
            if samples:
                output = CompletionSequenceGroupOutput(samples, None)
                sampler_outputs.append(output)
        return [SamplerOutput(outputs=sampler_outputs)]

    async def send_blocks(self, blocks_len) -> None:
        migration_latency = (self.cache_block_size * blocks_len) / self.migration_bandwidth
        await asyncio.sleep(migration_latency)
