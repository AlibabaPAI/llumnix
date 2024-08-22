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

from typing import List
import ray
import torch

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from vllm.utils import is_pin_memory_available
from vllm.worker.worker import Worker

from llumnix.logger import init_logger
from llumnix.backends.vllm.utils import _sample_with_torch

logger = init_logger(__name__)
NUMPY_SUPPORT_DTYPES = [torch.float32, torch.float16]

@ray.remote(num_cpus=0)
class RecvActor:
    def recv_cpu_cache(self, src_worker_handle, src_blocks, is_driver_worker):
        """
        Args:
            src_worker_handle: src worker actor handle
            blocks: block to send
        """
        try:
            if is_driver_worker:
                migration_cache = ray.get(src_worker_handle.execute_engine_method.remote("send_cpu_cache", src_blocks))
            else:
                migration_cache = ray.get(src_worker_handle.execute_method.remote("send_cpu_cache", src_blocks))
        # pylint: disable=try-except-raise
        except:
            raise
        return migration_cache

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

    def init_migration(self, num_migration_cache_blocks: int, src_worker_handle_list, placement_group=None, node_id=None) -> None:
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
        self.recv_actor = RecvActor.options(scheduling_strategy=scheduling_strategy).remote()

        self.migration_stream = torch.cuda.Stream()
        self.default_stream = torch.cuda.current_stream()
        self.num_migration_cache_blocks = num_migration_cache_blocks
        assert self.migration_stream != self.default_stream
        pin_memory = is_pin_memory_available()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        migration_cache_size = self.cache_engine.block_size * self.cache_engine.num_heads * self.cache_engine.head_size
        self.rpc_dtype = self.cache_engine.dtype
        if self.cache_engine.dtype in NUMPY_SUPPORT_DTYPES:
            self.rpc_dtype = self.cache_engine.dtype
        else:
            self.rpc_dtype = torch.float32
            logger.warning("Detecting numpy unsupported dtype: {}. Using torch.float32.".format(self.cache_engine.dtype))
        # self.migration_cache = torch.zeros(
        #     size=(self.cache_engine.num_layers, 2, self.num_migration_cache_blocks, migration_cache_size),
        #     dtype=self.cache_engine.dtype,
        #     pin_memory=pin_memory,
        # )
        self.migration_key_cache = torch.empty(
            size=(self.num_migration_cache_blocks, self.cache_engine.num_layers, migration_cache_size),
            dtype=self.cache_engine.dtype,
            pin_memory=pin_memory,
        )
        self.migration_value_cache = torch.empty(
            size=(self.num_migration_cache_blocks, self.cache_engine.num_layers, migration_cache_size),
            dtype=self.cache_engine.dtype,
            pin_memory=pin_memory,
        )
        # do dummy rpc
        src_worker_handle = src_worker_handle_list[self.rank]
        self.recv_actor.recv_cpu_cache.remote(src_worker_handle, [0], self.is_driver_worker)

    def send_cpu_cache(self, blocks: List[int]):
        num_blocks = len(blocks)
        dummy_key_cpu = self.migration_key_cache[:num_blocks]
        dummy_value_cpu = self.migration_value_cache[:num_blocks]
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.cache_engine.num_layers):
                for idx, block_num in enumerate(blocks):
                    dummy_key_cpu[idx][layer_idx].copy_(self.gpu_cache[layer_idx][0][block_num])
                    dummy_value_cpu[idx][layer_idx].copy_(self.gpu_cache[layer_idx][1][block_num])
        torch.cuda.Stream.synchronize(self.migration_stream)
        return (dummy_key_cpu.to(self.rpc_dtype).numpy(), dummy_value_cpu.to(self.rpc_dtype).numpy())

    def recv_cpu_cache(self, blocks: List[int], rpc_numpy_cache):
        num_blocks = len(blocks)
        dummy_key = self.migration_key_cache[:num_blocks]
        dummy_value = self.migration_value_cache[:num_blocks]
        k = rpc_numpy_cache[0]
        v = rpc_numpy_cache[1]
        dummy_key.copy_(torch.from_numpy(k))
        dummy_value.copy_(torch.from_numpy(v))
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.cache_engine.num_layers):
                for idx, block_num in enumerate(blocks):
                    self.gpu_cache[layer_idx][0][block_num].copy_(dummy_key[idx][layer_idx])
                    self.gpu_cache[layer_idx][1][block_num].copy_(dummy_value[idx][layer_idx])
        torch.cuda.Stream.synchronize(self.migration_stream)

    def send_cpu_cache_v2(self, blocks: List[int]):
        src_to_dst = dict(enumerate(blocks))
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.cache_engine.num_layers):
                self.cache_engine.attn_backend.swap_blocks(self.gpu_cache[layer_idx], self.migration_cache[layer_idx], src_to_dst)
        torch.cuda.Stream.synchronize(self.migration_stream)
        return self.migration_cache.to(self.rpc_dtype).numpy()

    def recv_cpu_cache_v2(self, blocks: List[int], rpc_numpy_cache):
        with torch.cuda.stream(self.migration_stream):
            self.migration_cache.copy_(torch.from_numpy(rpc_numpy_cache))
            src_to_dst = dict(enumerate(blocks))
            for layer_idx in range(self.cache_engine.num_layers):
                self.cache_engine.attn_backend.swap_blocks(self.migration_cache[layer_idx], self.gpu_cache[layer_idx],src_to_dst)
        torch.cuda.Stream.synchronize(self.migration_stream)

    def migrate_gpu_cache_ray_rpc(self, src_worker_handle_list, src_blocks: List[int], dst_blocks: List[int]):
        # TODO(s5u13b): Raise exception here.
        try:
            src_worker_handle = src_worker_handle_list[self.rank]
            tot_blocks = len(src_blocks)
            rpc_numpy_cache = None
            for start_idx in range(0, tot_blocks, self.num_migration_cache_blocks):
                # send/recv num_migration_cache_blocks per iter
                offset = min(self.num_migration_cache_blocks, tot_blocks - start_idx)
                send_blocks = src_blocks[start_idx:start_idx+offset]
                ray_obj = self.recv_actor.recv_cpu_cache.remote(src_worker_handle, send_blocks, self.is_driver_worker)
                if rpc_numpy_cache is not None:
                    self.recv_cpu_cache(recv_blocks, rpc_numpy_cache)
                rpc_numpy_cache = ray.get(ray_obj)
                recv_blocks = dst_blocks[start_idx:start_idx+offset]
            self.recv_cpu_cache(recv_blocks, rpc_numpy_cache)
        except ray.exceptions.RayActorError:
            logger.info("[migrate_gpu_cache_ray_rpc] self.rank: {}, src_worker_handle {} is dead".format(self.rank, src_worker_handle))

    # def send_gpu_cache_ray(self,rank_offset:int, blocks:List[int]):
    #     with torch.cuda.stream(self.migration_stream):
    #         dst_rank = self.ray_rank + rank_offset
    #         num_blocks = len(blocks)
    #         dummy_key_cpu = self.dummy_key_cpu[:num_blocks]
    #         dummy_value_cpu = self.dummy_value_cpu[:num_blocks]
    #         with torch.cuda.stream(self.migration_stream):
    #             for i in range(self.cache_engine.num_layers):
    #                 for idx,block_num in enumerate(blocks):
    #                     dummy_key_cpu[idx].copy_(self.gpu_cache[i][0][block_num])
    #                     dummy_value_cpu[idx].copy_(self.gpu_cache[i][1][block_num])
    #                 col.send(dummy_key_cpu, dst_rank)
    #                 col.send(dummy_value_cpu, dst_rank)

    #     torch.cuda.Stream.synchronize(self.migration_stream)

    # def recv_gpu_cache_ray(self,rank_offset:int, blocks):
    #     with torch.cuda.stream(self.migration_stream):
    #         src_rank = self.ray_rank + rank_offset
    #         num_blocks = len(blocks)
    #         dummy_key = self.dummy_key_cpu[:num_blocks]
    #         dummy_value = self.dummy_value_cpu[:num_blocks]
    #         for i in range(self.cache_engine.num_layers):
    #             col.recv(dummy_key, src_rank)
    #             col.recv(dummy_value, src_rank)
    #             for idx,block_num in enumerate(blocks):
    #                 self.gpu_cache[i][0][block_num].copy_(dummy_key[idx])
    #                 self.gpu_cache[i][1][block_num].copy_(dummy_value[idx])
    #     torch.cuda.Stream.synchronize(self.migration_stream)

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

    # instance_id is changed from int to str, this function should be modified if used
    # def init_migration_dist_ray(self, num_instances, instance_id):
    #     self.ray_world_size = num_instances * self.parallel_config.world_size
    #     self.ray_rank = self.rank + instance_id * self.parallel_config.world_size
    #     logger.info(f"{self.ray_world_size, self.ray_rank}")
    #     # col.init_collective_group(world_size=self.ray_world_size, rank=self.ray_rank , backend="gloo")
    #     # rpc.init_rpc(f"worker_{self.ray_rank}", rank=self.ray_rank, world_size=self.ray_world_size)


    # def run_migration_warmup(self):
    #     if self.ray_world_size > 1:
    #         if self.ray_rank % 2:
    #             self.recv_gpu_cache_ray(1 if self.ray_rank + 1 < self.ray_world_size else 1-self.ray_world_size,[0])
    #             self.send_gpu_cache_ray(-1 if self.ray_rank > 0 else self.ray_world_size-1,[0])
    #         else:
    #             self.send_gpu_cache_ray(-1 if self.ray_rank > 0 else self.ray_world_size-1,[0])
    #             self.recv_gpu_cache_ray(1 if self.ray_rank + 1 < self.ray_world_size else 1-self.ray_world_size,[0])
