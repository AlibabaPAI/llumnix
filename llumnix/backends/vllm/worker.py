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
from typing import Dict, List, Any, Union
import math
import ray
import torch

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

from vllm.utils import is_pin_memory_available
from vllm.worker.worker import Worker
from vllm.config import CacheConfig,  ModelConfig, ParallelConfig
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import GiB_bytes
from vllm.sequence import SequenceGroupMetadata, SequenceGroupMetadataDelta

from llumnix.logging.logger import init_logger
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

    def reserve_memory_for_migration(self,
                                     migration_config: MigrationConfig,
                                     model_config: ModelConfig,
                                     cache_config: CacheConfig,
                                     parallel_config: ParallelConfig) -> int:
        migrate_cache_blocks_size = migration_config.migration_buffer_blocks
        migration_num_layers = migration_config.migration_num_layers
        dummy_cache_size = migration_num_layers * migrate_cache_blocks_size * CacheEngine.get_cache_block_size(
            cache_config, model_config, parallel_config) // model_config.get_num_layers(parallel_config)

        # For nccl migration backend, reserve gpu memory for dummy cache in migration backend. For other backends,
        # CPU memory is used for the dummy cache, which is almost unlimited, so no special action is needed.
        if migration_config.migration_backend == "nccl" and parallel_config.world_size == 1:
            device = torch.device(f"cuda:{self.local_rank}")
            _, total_memory = torch.cuda.mem_get_info(device)
            migration_memory_ratio = math.ceil(dummy_cache_size / total_memory * 10000) / 10000
            cache_config.gpu_memory_utilization -= migration_memory_ratio

            if cache_config.gpu_memory_utilization <= 0:
                raise ValueError("Nccl migration backend take {:.4f} gpu memory, which is greater than gpu_memory_utilization {:.4f}. "
                                 "try to increase gpu-memory-utilization or reduce migration-buffer-blocks."
                                 .format(migration_memory_ratio, cache_config.gpu_memory_utilization))

            logger.info("Nccl migration backend take {:.4f} gpu memory, left gpu_memory_utilization {:.4f} for kv cache."
                        .format(migration_memory_ratio, cache_config.gpu_memory_utilization))

        return dummy_cache_size

    def init_migration(self,
                       instance_id: str,
                       migration_config: MigrationConfig,
                       src_worker_handle_list: List["ray.actor.ActorHandle"],
                       placement_group: PlacementGroup) -> None:
        # for proxy actor
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
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
                                                                             local_rank=self.local_rank,
                                                                             worker_put_data_dst_callback=self.put_data_dst)

    def migrate_cache(self,
                      src_worker_handle_list: List["ray.actor.ActorHandle"],
                      src_blocks: List[int],
                      dst_blocks: List[int],
                      request_id: str,
                      recv_worker_data: bool = False) -> None:
        src_worker_handle = src_worker_handle_list[self.rank]

        start_time = time.time()
        try:
            self.migration_backend.migrate_cache(src_worker_handle, src_blocks, dst_blocks, request_id, recv_worker_data)
        except ray.exceptions.RayActorError:
            logger.info("rank: {}, src_worker_handle {} is dead".format(self.rank, src_worker_handle))
        end_time = time.time()

        total_kv_cache_size = len(src_blocks) * CacheEngine.get_cache_block_size(
            self.cache_config, self.model_config, self.parallel_config)
        speed = total_kv_cache_size/GiB_bytes/(end_time - start_time)
        logger.info("Migrate kv cache done, blocks_num: {}, total_kv_cache_size: {}, time: {}s, speed: {}GB/s."
                    .format(len(src_blocks), convert_bytes(total_kv_cache_size), end_time-start_time, speed))

    def do_recv(self, *args, **kwargs):
        return self.migration_backend.do_recv(*args, **kwargs)

    def do_send(self, *args, request_id: str = None, send_worker_data: bool = False, **kwargs):
        if not send_worker_data:
            return self.migration_backend.do_send(*args, **kwargs)
        return self.migration_backend.do_send(*args, **kwargs), self.get_data_src(request_id)

    def get_data_src(self, request_id: str) -> Dict[str, Any]:
        src_data = {
            "_seq_group_metadata_cache": {}
        }
        # the request_id is in self._seq_group_metadata_cache when the request is in running states.
        if request_id in self._seq_group_metadata_cache:
            src_data["_seq_group_metadata_cache"][request_id] = self._seq_group_metadata_cache[request_id]
            del self._seq_group_metadata_cache[request_id]

        return src_data

    def put_data_dst(self, src_data: Dict[str, Any]) -> None:
        if "_seq_group_metadata_cache" in src_data:
            # Only support migrate one request at once currently.
            if src_data["_seq_group_metadata_cache"].keys():
                request_id = list(src_data["_seq_group_metadata_cache"].keys())[0]
                self._seq_group_metadata_cache[request_id] = src_data["_seq_group_metadata_cache"][request_id]

    def _get_cached_seq_group_metadata(
            self,
            seq_group_metadata_list: List[Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]],
            finished_request_ids: List[str]) -> List[SequenceGroupMetadata]:
        # Change seq_data for cached seq_grou_metadata in worker (the src seq_id is different from the dst seq_id).
        for metadata_or_delta in seq_group_metadata_list:
            request_id = metadata_or_delta.request_id
            if request_id in self._seq_group_metadata_cache:
                seq_group_metadata = self._seq_group_metadata_cache[request_id]
                if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
                    # TODO(s5u13b): Only support one sequence.
                    for new_seq_id, old_seq_id in zip(metadata_or_delta.seq_data_delta.keys(), \
                                                      seq_group_metadata.seq_data.keys()):
                        if new_seq_id != old_seq_id:
                            seq_group_metadata.seq_data[new_seq_id] = seq_group_metadata.seq_data[old_seq_id]
                            del seq_group_metadata.seq_data[old_seq_id]

        return super()._get_cached_seq_group_metadata(seq_group_metadata_list, finished_request_ids)

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
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
