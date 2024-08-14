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

from typing import Dict, List
import math
import os
import ray
import torch

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from vllm.utils import is_pin_memory_available
from vllm.worker.worker import Worker
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.worker.cache_engine import CacheEngine

from llumnix.logger import init_logger
from llumnix.backends.vllm.utils import _sample_with_torch

from llumnix.backends.vllm.migration_backend import MigrationBackendBase, get_migrate_backend
from llumnix.config import MigrationConfig

logger = init_logger(__name__)

class MigrationWorker(Worker):
    def __init__(self, *args, **kwargs) -> None:
        # replace sampler
        # pylint: disable=import-outside-toplevel
        import vllm.model_executor.layers.sampler
        vllm.model_executor.layers.sampler._sample_with_torch = _sample_with_torch

        backend = os.environ.get("MIGRATE_BACKEND", "rpc")
        migrate_cache_size = int(os.environ.get("MIGRATE_CACHE_SIZE", 32))
        migrate_num_layers =int(os.environ.get("MIGRATE_NUM_LAYERS", 1))

        parallel_config: ParallelConfig = kwargs["parallel_config"]
        model_parallelism_enabled = parallel_config.world_size > 1

        if backend == "nccl" and (not model_parallelism_enabled):
            model_config: ModelConfig = kwargs["model_config"]
            cache_config: CacheConfig = kwargs["cache_config"]
            # for nccl backend gpu cache
            total_size = migrate_num_layers * migrate_cache_size * CacheEngine.get_cache_block_size(
                cache_config, model_config, parallel_config) // model_config.get_num_layers(parallel_config)

            device = torch.device(f"cuda:{kwargs['local_rank']}")
            _, total_memory = torch.cuda.mem_get_info(device)
            migrate_ratio = math.ceil(total_size / total_memory * 10000) / 10000
            cache_config.gpu_memory_utilization -= migrate_ratio

            if cache_config.gpu_memory_utilization < 0:
                raise RuntimeError("nccl migration backend take {} gpu memory, which is greater than gpu_memory_utilization {}. " \
                                    "try to increase gpu_memory_utilization or reduce migration-cache-blocks."
                                    .format(migrate_ratio, cache_config.gpu_memory_utilization))

            logger.info("nccl migration backend take {} gpu memory, left gpu_memory_utilization {} for kv cache." \
                        .format(migrate_ratio, cache_config.gpu_memory_utilization))

        super().__init__(*args, **kwargs)

    def load_model(self):
        torch.cuda.set_device(self.device)
        return super().load_model()

    def get_ray_rank(self):
        return self.ray_rank

    def init_migration(self, instance_id: str, migration_config: MigrationConfig, src_worker_handle_list, placement_group=None) -> None:
        if placement_group:
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
            )
        else:
            scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )

        pin_memory = is_pin_memory_available()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")

        self.instance_id = instance_id
        self.migrate_backend: MigrationBackendBase = get_migrate_backend(migrate_config=migration_config,
                                                  cache_engine=self.cache_engine,
                                                  worker_handle_list=src_worker_handle_list,
                                                  scheduling_strategy=scheduling_strategy,\
                                                  is_driver_worker=self.is_driver_worker,
                                                  gpu_cache=self.gpu_cache,
                                                  worker_rank=self.rank,
                                                  local_rank=self.local_rank)

    def migrate_cache(self, src_worker_handle_list, src_blocks: List[int], dst_blocks: List[int]):
        try:
            src_worker_handle = src_worker_handle_list[self.rank]
            self.migrate_backend.migrate_cache(src_worker_handle, src_blocks, dst_blocks)
        except ray.exceptions.RayActorError:
            logger.info("[migrate_cache] self.rank: {}, src_worker_handle {} is dead".format(self.rank, src_worker_handle))

    def do_recv(self, *args, **kwargs):
        return self.migrate_backend.do_recv(*args, **kwargs)

    def do_send(self, *args, **kwargs):
        return self.migrate_backend.do_send(*args, **kwargs)

    def rebuild_migrate_backend(self, instance_rank: Dict[str, int], group_name: str):
        self.migrate_backend.destory_col()

        ret = True
        if group_name is not None:
            num_instance = len(instance_rank)
            self.ray_world_size = num_instance * self.parallel_config.world_size
            self.ray_rank = self.rank + instance_rank[self.instance_id] * self.parallel_config.world_size
            ret = self.migrate_backend.init_col(group_name, self.ray_world_size, self.ray_rank)
        
        return ret

    def warmup(self):
        return self.migrate_backend.warmup()

    def shutdown(self) -> None:
        torch.cuda.synchronize()
        del self.model_runner
        del self.cache_engine
        del self.gpu_cache
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    def restart(self) -> None:
        self.init_model()
        self.init_cache_engine(self.cache_config)
