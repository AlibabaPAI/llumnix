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

from typing import List, Tuple
import torch
from func_timeout import func_set_timeout, FunctionTimedOut
import cupy
from cupy.cuda import nccl
import ray
import ray.util.collective as col
from ray.util.collective.collective_group import nccl_util
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.worker.cache_engine import CacheEngine

from llumnix.internal_config import MigrationConfig
from llumnix.backends.migration_backend_interface import MigrationBackendBase, MigrationBackendWithBuffer
from llumnix.logging.logger import init_logger
from llumnix import envs as llumnix_envs

logger = init_logger(__name__)


class ProxyActor:
    def exec_method(self, is_driver_worker, handle, *args, **kwargs):
        try:
            if is_driver_worker:
                ret = ray.get(handle.execute_engine_method_async.remote("execute_worker_method_async", *args, **kwargs))
            else:
                ret = ray.get(handle.execute_method.remote(*args, **kwargs))
        # pylint: disable=try-except-raise
        except:
            raise

        return ret

NUMPY_SUPPORTED_DTYPES = [torch.float32, torch.float16]


class RayRpcMigrationBackend(MigrationBackendWithBuffer):
    def __init__(self,
                 migration_config: MigrationConfig,
                 cache_engine: List[CacheEngine],
                 worker_rank: int,
                 worker_handle_list: List["ray.actor.ActorHandle"], \
                 scheduling_strategy: PlacementGroupSchedulingStrategy,
                 is_driver_worker: bool,
                 gpu_cache: List[torch.Tensor]) -> None:
        self.migration_config = migration_config
        self.cache_engine = cache_engine

        self.worker_rank = worker_rank
        self.worker_handle_list = worker_handle_list
        worker_max_concurrency = llumnix_envs.LLUMNIX_WORKER_MAX_CONCURRENCY
        self.actor = ray.remote(
            num_cpus=0,
            max_concurrency=worker_max_concurrency,
            scheduling_strategy=scheduling_strategy,
        )(ProxyActor).remote()
        self.migration_stream = torch.cuda.Stream()

        if self.cache_engine[0].dtype in NUMPY_SUPPORTED_DTYPES:
            self.rpc_dtype = self.cache_engine[0].dtype
        else:
            self.rpc_dtype = torch.float32
            logger.warning("Detect numpy unsupported dtype: {}. Using torch.float32.".format(self.cache_engine[0].dtype))

        self.is_driver_worker = is_driver_worker
        self.gpu_cache = gpu_cache
        self.cache_device = "cpu"
        self.num_migration_buffer_blocks = self.migration_config.migration_buffer_blocks
        self.migration_num_layers = self.cache_engine[0].num_attention_layers
        self.migration_cache_size = self.cache_engine[0].block_size * self.cache_engine[0].num_kv_heads * self.cache_engine[0].head_size
        buffer_shape = (self.num_migration_buffer_blocks, self.migration_num_layers, 2, self.migration_cache_size)

        super().__init__(buffer_shape, self.cache_engine[0].dtype, self.cache_device,
                         pin_memory=True, num_buffers=migration_config.migration_num_buffers)

    # pylint: disable=unused-argument
    def init_backend(self, group_name: str, world_size: int, rank: int) -> bool:
        logger.info("Create rayrpc migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        # rpc migration backend does not need to be destroyed as there is no group.
        # It uses ray actor handle to migration cache blocks.
        pass

    def warmup(self) -> bool:
        self.actor.exec_method.remote(self.is_driver_worker, self.worker_handle_list[self.worker_rank], "do_send", None, [0])
        logger.info("Rayrpc migration backend warmup successfully.")
        return True

    # The src actor will pack the kv-cache data layer by layer. Specifically, NumPy is used for the transfer
    # because, for a single node, Ray RPC can transfer NumPy arrays via shared memory. Then, the recv actor
    # first copies the data to a pinned-memory dummy cache before transferring it to the GPU to accelerate data transfer.
    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        tot_blocks = len(src_blocks)
        rpc_numpy_cache = None
        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            ray_obj = self.actor.exec_method.remote(self.is_driver_worker, src_handle, "do_send", None, send_blocks)
            if rpc_numpy_cache is not None:
                self.do_recv(rpc_numpy_cache, recv_blocks)
            rpc_numpy_cache_ref = ray.get(ray_obj)
            rpc_numpy_cache = ray.get(rpc_numpy_cache_ref)
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
        self.do_recv(rpc_numpy_cache, recv_blocks)

    def do_send(self, _, blocks: List[int], virtuel_engine: int=0):
        num_blocks = len(blocks)
        dummy_cache_idx = self.get_available_cache()
        send_cache = self.dummy_buffer[dummy_cache_idx][:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
        # src_to_dst = {block_num: idx for idx, block_num in enumerate(blocks)}
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((blocks[idx], idx))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=True).view(-1, 2)
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.migration_num_layers):
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(self.gpu_cache[virtuel_engine][layer_idx], send_cache[layer_idx], block_mapping_tensor)
        torch.cuda.Stream.synchronize(self.migration_stream)
        # Here, we use ray.put to store data and finally return the object reference so that we can release the internal buffer.
        # This might seem like an anti-pattern, but it's okay since the kv-cache transferred is in the MB range and won't utilize
        # Ray's optimization for returning small objects (<100KB).
        data = ray.put(send_cache.to(self.rpc_dtype).numpy())
        self.put_back_cache(dummy_cache_idx)
        return data

    def do_recv(self, src_handle, blocks: List[int], virtuel_engine: int=0):
        num_blocks = len(blocks)
        # src_to_dst = dict(enumerate(blocks))
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((idx, blocks[idx]))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=True).view(-1, 2)
        dummy_cache_idx = self.get_available_cache()
        recv_cache = self.dummy_buffer[dummy_cache_idx][:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
        # use pin memory dummy_cache to speed up data transfer
        recv_cache.copy_(torch.from_numpy(src_handle))

        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.migration_num_layers):
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(recv_cache[layer_idx], self.gpu_cache[virtuel_engine][layer_idx], block_mapping_tensor)
        torch.cuda.Stream.synchronize(self.migration_stream)
        self.put_back_cache(dummy_cache_idx)

def try_import_gloo():
    try:
        # pylint: disable=C0415
        from ray.util.collective.collective_group import gloo_util
        import pygloo

        # Add support for bf16 type in Gloo. Now bf16 and fp16 both map to glooFloat16, but this is okay because
        # Gloo only uses the data type size for transmission.
        # pylint: disable=W0221,I1101
        gloo_util.TORCH_GLOO_DTYPE_MAP[torch.bfloat16] = pygloo.glooDataType_t.glooFloat16
    except ImportError as e:
        raise ImportError("Gloo is not installed. Please install it first.") from e


class RayColMigrationBackend(MigrationBackendWithBuffer):
    def __init__(self,
                 migration_config: MigrationConfig,
                 cache_engine: List[CacheEngine],
                 local_rank: int,
                 scheduling_strategy: PlacementGroupSchedulingStrategy,
                 is_driver_worker: bool,
                 gpu_cache: List[torch.Tensor]) -> None:
        self.migration_config = migration_config
        self.cache_engine = cache_engine
        self.backend = migration_config.migration_backend
        self.migration_num_layers = min(migration_config.migration_num_layers, self.cache_engine[0].num_attention_layers)
        self.num_migration_buffer_blocks = migration_config.migration_buffer_blocks

        self.backend = migration_config.migration_backend
        self.global_world_size = -1
        self.global_rank = -1
        self.group_name = None

        self.local_rank = local_rank
        worker_max_concurrency = llumnix_envs.LLUMNIX_WORKER_MAX_CONCURRENCY
        self.actor = ray.remote(
            num_cpus=0,
            max_concurrency=worker_max_concurrency,
            scheduling_strategy=scheduling_strategy,
        )(ProxyActor).remote()
        self.is_driver_worker = is_driver_worker
        self.gpu_cache = gpu_cache
        self.migration_stream = cupy.cuda.Stream()

        self.migration_cache_size = self.cache_engine[0].block_size * self.cache_engine[0].num_kv_heads * self.cache_engine[0].head_size

        if self.backend == 'gloo':
            try_import_gloo()
            self.cache_device = "cpu"
            self.pin_memory = True
        else:
            nccl_util.TORCH_NCCL_DTYPE_MAP[torch.bfloat16] = nccl.NCCL_FLOAT16
            self.cache_device = torch.device(f"cuda:{self.local_rank}")
            self.pin_memory = False

        buffer_shape = (self.num_migration_buffer_blocks, self.migration_num_layers, 2, self.migration_cache_size)
        super().__init__(buffer_shape, self.cache_engine[0].dtype, self.cache_device,
                         pin_memory=self.pin_memory, num_buffers=migration_config.migration_num_buffers)

    def init_backend(self, group_name, world_size, rank) -> bool:
        @func_set_timeout(self.migration_config.migration_backend_init_timeout)
        def init_group(world_size, rank, backend, group_name):
            col.init_collective_group(world_size, rank, backend, group_name)

        try:
            init_group(world_size, rank, self.backend, group_name)
        except FunctionTimedOut:
            logger.info("Create migration backend failed (group_name: {}, world_size: {}, rank: {}, backbend: {})."
                .format(group_name, world_size, rank, self.backend))
            return False

        self.group_name = group_name
        self.global_world_size = world_size
        self.global_rank = rank

        logger.info("Create migration backend group successfully (group_name: {}, world_size: {}, rank: {}, backbend: {})."
                    .format(self.group_name, self.global_world_size, self.global_rank, self.backend))
        return True

    def destory_backend(self) -> None:
        if self.group_name is None:
            return

        err_info = None
        try:
            col.destroy_collective_group(self.group_name)
        # pylint: disable=W0703
        except Exception as e:
            err_info = e

        if err_info is not None:
            logger.info("Destory migration backend successfully (group_name: {}, backbend: {}), error: {}."
                    .format(self.group_name, self.backend, err_info))
        else:
            logger.info("Destory migration backend successfully (group_name: {}, backbend: {})."
                    .format(self.group_name, self.backend))

        self.group_name = None

    def warmup(self) -> bool:
        if self.global_world_size > 1:
            try:
                col.allreduce(self.dummy_buffer[0][0], self.group_name)
            # pylint: disable=W0703
            except Exception as e:
                logger.error("Migration backend warmup failed (group_name: {}, world_size: {}, rank: {}, backbend: {}), err: {}."
                    .format(self.group_name, self.global_world_size, self.global_rank, self.backend, e))
                return False

        logger.info("Migration backend warmup successfully (group_name: {}, world_size: {}, rank: {}, backbend: {})."
                    .format(self.group_name, self.global_world_size, self.global_rank, self.backend))
        return True

    # Ray.collective is used to construct the gloo and nccl backends. The do_send/do_recv functions will transmit
    # data layer by layer. Take into consideration that col.send/recv are blocking operations.
    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        tot_blocks = len(src_blocks)
        src_rank = ray.get(self.actor.exec_method.remote(self.is_driver_worker, src_handle, "get_global_rank"))

        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
            self.actor.exec_method.remote(self.is_driver_worker, src_handle, "do_send", self.global_rank, send_blocks)
            self.do_recv(src_rank, recv_blocks)

    def do_send(self, dst_handle, blocks: List[int], virtuel_engine: int=0):
        num_blocks = len(blocks)
        dummy_cache_idx = self.get_available_cache()
        send_cache = self.dummy_buffer[dummy_cache_idx][:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((blocks[idx], idx))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=True).view(-1, 2)
        with self.migration_stream:
            for layer_idx in range(self.cache_engine[0].num_attention_layers):
                cache_idx = layer_idx % self.migration_num_layers
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(self.gpu_cache[virtuel_engine][layer_idx], send_cache[cache_idx], block_mapping_tensor)
                if cache_idx + 1 == self.migration_num_layers or layer_idx + 1 == self.cache_engine[0].num_attention_layers:
                    # TODO(KuilongCui): check the error code if peer is dead
                    col.send(send_cache, dst_handle, self.group_name)
        self.migration_stream.synchronize()
        self.put_back_cache(dummy_cache_idx)

    def do_recv(self, src_handle, blocks: List[int], virtuel_engine: int=0):
        num_blocks = len(blocks)
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((idx, blocks[idx]))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=True).view(-1, 2)
        dummy_cache_idx = self.get_available_cache()
        recv_cache = self.dummy_buffer[dummy_cache_idx][:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)

        with self.migration_stream:
            for layer_idx in range(self.cache_engine[0].num_attention_layers):
                cache_idx = layer_idx % self.migration_num_layers
                if cache_idx == 0:
                    col.recv(recv_cache, src_handle, self.group_name)
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(recv_cache[cache_idx], self.gpu_cache[virtuel_engine][layer_idx], block_mapping_tensor)
        self.migration_stream.synchronize()
        self.put_back_cache(dummy_cache_idx)

def get_migration_backend(migration_config: MigrationConfig,
                          cache_engine: List[CacheEngine],
                          worker_handle_list: List["ray.Actor.ActorHanle"],
                          scheduling_strategy: PlacementGroupSchedulingStrategy,
                          is_driver_worker: bool,
                          gpu_cache: List[torch.Tensor],
                          worker_rank: int,
                          local_rank: int) -> MigrationBackendBase:
    if cache_engine[0].num_gpu_blocks < migration_config.migration_buffer_blocks:
        logger.warning("migration_cache_blocks({}) is larger than num_gpu_blocks({}), reducing it to num_gpu_blocks."
                       .format(migration_config.migration_buffer_blocks, cache_engine[0].num_gpu_blocks))
        migration_config.migration_buffer_blocks = cache_engine[0].num_gpu_blocks

    target_migration_backend = None
    backend = migration_config.migration_backend

    assert backend in ['nccl', 'rayrpc', 'gloo'], "Unsupported migration backend for vLLM: {}".format(backend)

    if backend in ['nccl', 'gloo']:
        target_migration_backend = RayColMigrationBackend(migration_config, cache_engine, local_rank, scheduling_strategy,
                                                          is_driver_worker, gpu_cache)
    else:
        target_migration_backend = RayRpcMigrationBackend(migration_config, cache_engine, worker_rank, worker_handle_list,
                                                          scheduling_strategy, is_driver_worker, gpu_cache)

    return target_migration_backend
