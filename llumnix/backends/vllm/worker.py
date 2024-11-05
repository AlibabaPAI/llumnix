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
from typing import Dict, List
import ray
import torch

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from vllm.utils import is_pin_memory_available
from vllm.worker.worker import Worker
from vllm.worker.cache_engine import CacheEngine
from vllm.config import _GB

from llumnix.logger import init_logger
from llumnix.backends.vllm.utils import _sample_with_torch
from llumnix.backends.vllm.migration_backend import MigrationBackendBase, get_migration_backend
from llumnix.internal_config import MigrationConfig
from llumnix.utils import convert_bytes

logger = init_logger(__name__)

class MigrationWorker(Worker):
    def __init__(self, *args, **kwargs) -> None:
        # replace sampler
        # pylint: disable=import-outside-toplevel
        import vllm.model_executor.layers.sampler
        vllm.model_executor.layers.sampler._sample_with_torch = _sample_with_torch

        super().__init__(*args, **kwargs)

    def load_model(self):
        torch.cuda.set_device(self.device)
        return super().load_model()

    def get_global_rank(self):
        return self.global_rank

    def init_migration(self, instance_id: str, migration_config: MigrationConfig, src_worker_handle_list,
                       placement_group=None, node_id=None) -> None:
        if placement_group:
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
            )
        else:
            scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )

        pin_memory = is_pin_memory_available()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")

        self.instance_id = instance_id
        self.global_world_size = 0
        self.global_rank = -1
        self.migration_backend: MigrationBackendBase = get_migration_backend(migration_config=migration_config,
                                                                             cache_engine=self.cache_engine,
                                                                             worker_handle_list=src_worker_handle_list,
                                                                             scheduling_strategy=scheduling_strategy,
                                                                             is_driver_worker=self.is_driver_worker,
                                                                             gpu_cache=self.gpu_cache,
                                                                             worker_rank=self.rank,
                                                                             local_rank=self.local_rank)

    def migrate_cache(self, src_worker_handle_list, src_blocks: List[int], dst_blocks: List[int]) -> None:
        src_worker_handle = src_worker_handle_list[self.rank]

        start_time = time.time()
        try:
            self.migration_backend.migrate_cache(src_worker_handle, src_blocks, dst_blocks)
        except ray.exceptions.RayActorError:
            logger.info("[migrate_cache] self.rank: {}, src_worker_handle {} is dead".format(self.rank, src_worker_handle))
        end_time = time.time()

        total_kv_cache_size = len(src_blocks) * CacheEngine.get_cache_block_size(
            self.cache_config, self.model_config, self.parallel_config)
        speed = total_kv_cache_size/_GB/(end_time - start_time)
        logger.info("[migration_cache] blocks_num: {}, total_kv_cache_size: {}, time: {}s, speed: {}GB/s."
                    .format(len(src_blocks), convert_bytes(total_kv_cache_size), end_time-start_time, speed))

    def do_recv(self, *args, **kwargs):
        return self.migration_backend.do_recv(*args, **kwargs)

    def do_send(self, *args, **kwargs):
        return self.migration_backend.do_send(*args, **kwargs)

    def rebuild_migration_backend(self, instance_rank: Dict[str, int], group_name: str) -> bool:
        self.migration_backend.destory_backend()

        ret = True
        if group_name is not None:
            num_instance = len(instance_rank)
            self.global_world_size = num_instance * self.parallel_config.world_size
            self.global_rank = self.rank + instance_rank[self.instance_id] * self.parallel_config.world_size
            ret = self.migration_backend.init_backend(group_name, self.global_world_size, self.global_rank)

        return ret

    def warmup(self) -> bool:
        return self.migration_backend.warmup()

    def shutdown(self) -> None:
        torch.cuda.synchronize()
        del self.model_runner
        del self.cache_engine
        del self.gpu_cache
        del self.migration_backend
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
