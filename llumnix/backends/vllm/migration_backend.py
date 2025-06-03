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

from typing import List, Tuple, Optional, Callable, Any

import torch
from func_timeout import func_set_timeout
import ray
import ray.util.collective as col
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray.actor

from vllm.worker.cache_engine import CacheEngine

from llumnix.internal_config import MigrationConfig
from llumnix.backends.migration_backend_interface import MigrationBackendBase
from llumnix.logging.logger import init_logger
from llumnix.constants import (
    NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION,
    RAYRPC_MIGRATION_TIMEOUT,
)
from llumnix.utils import random_uuid, ray_get_with_timeout

logger = init_logger(__name__)


# Once worker died, proxy actor will not restart.
@ray.remote(num_cpus=0, max_concurrency=2, max_restarts=-1)
class ProxyActor:
    def __init__(self, is_driver_worker: bool, use_ray_spmd_worker: bool):
        self.is_driver_worker = is_driver_worker
        self.use_ray_spmd_worker = use_ray_spmd_worker

    def exec_method(self, handle: ray.actor.ActorHandle, *args, **kwargs) -> Any:
        if self.is_driver_worker and not self.use_ray_spmd_worker:
            ret = ray_get_with_timeout(
                handle.execute_engine_method_async.remote(
                    "execute_driver_worker_method_async", *args, **kwargs
                )
            )
        else:
            ret = ray_get_with_timeout(
                handle.execute_method.remote(*args, **kwargs)
            )

        return ret


NUMPY_SUPPORTED_DTYPES = [torch.float32, torch.float16]


class RayRpcMigrationBackend(MigrationBackendBase):
    def __init__(self,
                 instance_id: str,
                 migration_config: MigrationConfig,
                 cache_engine: List[CacheEngine],
                 worker_rank: int,
                 worker_handle_list: List[ray.actor.ActorHandle],
                 scheduling_strategy: PlacementGroupSchedulingStrategy,
                 is_driver_worker: bool,
                 gpu_cache: Optional[List[List[torch.Tensor]]],
                 use_ray_spmd_worker: bool,
                 worker_stage_seq_group_metadata_callback: Callable) -> None:
        super().__init__()

        self.instance_id = instance_id
        self.migration_config = migration_config
        self.cache_engine = cache_engine

        self.worker_rank = worker_rank
        self.worker_handle_list = worker_handle_list
        self.proxy_actor = ProxyActor.options(
            scheduling_strategy=scheduling_strategy,
            name=f"ProxyActor_{self.instance_id}_{random_uuid()}").remote(
                is_driver_worker, use_ray_spmd_worker
            )

        if self.cache_engine[0].dtype in NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION:
            self.rpc_dtype = self.cache_engine[0].dtype
        else:
            self.rpc_dtype = torch.float32
            logger.warning("Detect numpy unsupported dtype: {}. Using torch.float32.".format(self.cache_engine[0].dtype))

        self.gpu_cache = gpu_cache
        self.use_ray_spmd_worker = use_ray_spmd_worker
        self.worker_stage_seq_group_metadata_callback = worker_stage_seq_group_metadata_callback

        self.cache_device = "cpu"
        self.num_migration_buffer_blocks = self.migration_config.migration_buffer_blocks
        self.num_layers = self.cache_engine[0].num_attention_layers
        self.migration_cache_size = self.cache_engine[0].block_size * self.cache_engine[0].num_kv_heads * self.cache_engine[0].head_size

        self.dummy_cache = torch.empty(
            size=(self.num_migration_buffer_blocks, self.num_layers, 2, self.migration_cache_size),
            dtype=self.cache_engine[0].dtype,
            device=self.cache_device,
            pin_memory=True
        )
        self.migration_stream = torch.cuda.Stream()

    def init_backend(self, group_name: str, world_size: int, rank: int) -> bool:
        logger.info("Create rayrpc migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        # rpc migration backend does not need to be destroyed as there is no group.
        # It uses ray actor handle to migration cache blocks.
        pass

    def warmup(self) -> bool:
        ray_get_with_timeout(
            self.proxy_actor.exec_method.remote(
                self.worker_handle_list[self.worker_rank], "do_send", None, [0]
            )
        )
        logger.info("Rayrpc migration backend warmup successfully.")
        return True

    # The src actor will pack the kv-cache data layer by layer. Specifically, NumPy is used for the transfer
    # because, for a single node, Ray RPC can transfer NumPy arrays via shared memory. Then, the recv actor
    # first copies the data to a pinned-memory dummy cache before transferring it to the GPU to accelerate data transfer.
    def recv_cache(self,
                   request_id: str,
                   src_worker_handle: ray.actor.ActorHandle,
                   src_blocks: List[int],
                   dst_blocks: List[int],
                   is_last_stage: bool) -> None:
        tot_blocks = len(src_blocks)
        rpc_numpy_cache = None
        src_seq_group_metadata = None
        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            is_last_comm = (tot_blocks - start_idx <= self.num_migration_buffer_blocks)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            send_worker_metadata = self.use_ray_spmd_worker and is_last_stage and is_last_comm
            ray_obj = self.proxy_actor.exec_method.remote(
                src_worker_handle,
                "do_send",
                None,
                send_blocks,
                request_id=request_id,
                send_worker_metadata=send_worker_metadata
            )
            if rpc_numpy_cache is not None:
                self.do_recv(rpc_numpy_cache, recv_blocks)
            recv_blocks = dst_blocks[start_idx:start_idx+offset]

            if send_worker_metadata:
                rpc_numpy_cache, src_seq_group_metadata = ray_get_with_timeout(ray_obj, timeout=RAYRPC_MIGRATION_TIMEOUT)
            else:
                rpc_numpy_cache = ray_get_with_timeout(ray_obj)

        self.do_recv(rpc_numpy_cache, recv_blocks)
        if src_seq_group_metadata:
            self.worker_stage_seq_group_metadata_callback(request_id, src_seq_group_metadata)

    def do_send(self, dst_worker_handle: ray.actor.ActorHandle, blocks: List[int], virtuel_engine: int=0) -> None:
        num_blocks = len(blocks)
        send_cache = self.dummy_cache[:num_blocks].view(self.num_layers, 2, num_blocks, self.migration_cache_size)
        # src_to_dst = {block_num: idx for idx, block_num in enumerate(blocks)}
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((blocks[idx], idx))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu", pin_memory=True).view(-1, 2)
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.num_layers):
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(self.gpu_cache[virtuel_engine][layer_idx], send_cache[layer_idx], block_mapping_tensor)
        torch.cuda.Stream.synchronize(self.migration_stream)
        return send_cache.to(self.rpc_dtype).numpy()

    # pylint: disable=arguments-differ
    def do_recv(self, src_worker_handle: ray.actor.ActorHandle, blocks: List[int], virtuel_engine: int=0) -> None:
        num_blocks = len(blocks)
        # src_to_dst = dict(enumerate(blocks))
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((idx, blocks[idx]))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu", pin_memory=True).view(-1, 2)
        recv_cache = self.dummy_cache[:num_blocks].view(self.num_layers, 2, num_blocks, self.migration_cache_size)
        # use pin memory dummy_cache to speed up data transfer
        recv_cache.copy_(torch.from_numpy(src_worker_handle))

        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.num_layers):
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(recv_cache[layer_idx], self.gpu_cache[virtuel_engine][layer_idx], block_mapping_tensor)
        torch.cuda.Stream.synchronize(self.migration_stream)


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


class RayColMigrationBackend(MigrationBackendBase):
    def __init__(self,
                 instance_id: str,
                 migration_config: MigrationConfig,
                 cache_engine: List[CacheEngine],
                 local_rank: int,
                 scheduling_strategy: PlacementGroupSchedulingStrategy,
                 is_driver_worker: bool,
                 gpu_cache: Optional[List[List[torch.Tensor]]],
                 use_ray_spmd_worker: bool,
                 worker_stage_seq_group_metadata_callback: Callable) -> None:
        super().__init__()

        # pylint: disable=C0415
        import cupy

        self.instance_id = instance_id
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
        self.proxy_actor = ProxyActor.options(
            scheduling_strategy=scheduling_strategy,
            name=f"ProxyActor_{self.instance_id}_{random_uuid()}").remote(
                is_driver_worker, use_ray_spmd_worker
            )
        self.gpu_cache = gpu_cache
        self.use_ray_spmd_worker = use_ray_spmd_worker
        self.worker_stage_seq_group_metadata_callback = worker_stage_seq_group_metadata_callback

        self.migration_cache_size = self.cache_engine[0].block_size * self.cache_engine[0].num_kv_heads * self.cache_engine[0].head_size

        if self.backend == 'gloo':
            try_import_gloo()
            self.cache_device = "cpu"
        else:
            self.cache_device = torch.device(f"cuda:{self.local_rank}")

        pin_memory = (self.backend == 'gloo')
        self.dummy_cache = torch.empty(
            size=(self.num_migration_buffer_blocks, self.migration_num_layers, 2, self.migration_cache_size),
            dtype=self.cache_engine[0].dtype,
            device=self.cache_device,
            pin_memory=pin_memory
        )

        self.migration_stream = cupy.cuda.Stream()

    def init_backend(self, group_name: str, world_size: int, rank: int) -> bool:
        @func_set_timeout(self.migration_config.migration_backend_init_timeout)
        def init_group(world_size, rank, backend, group_name):
            col.init_collective_group(world_size, rank, backend, group_name)

        try:
            init_group(world_size, rank, self.backend, group_name)
        # pylint: disable=broad-except
        except Exception as e:
            logger.exception("Failed to create migration backend "
                             "(group_name: {}, world_size: {}, rank: {}, backbend: {}), "
                             "unexpected exception: {}".format(group_name, world_size, rank, self.backend, e))
            return False

        self.group_name = group_name
        self.global_world_size = world_size
        self.global_rank = rank

        logger.info("Create migration backend group successfully "
                    "(group_name: {}, world_size: {}, rank: {}, backbend: {})."
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
            logger.exception("Failed to destory migration backend (group_name: {}, backbend: {}), "
                             "unexpected exception: {}".format(self.group_name, self.backend, err_info))
        else:
            logger.info("Destory migration backend successfully "
                        "(group_name: {}, backbend: {}).".format(self.group_name, self.backend))

        self.group_name = None

    def warmup(self) -> bool:
        if self.global_world_size > 1:
            try:
                col.allreduce(self.dummy_cache[0], self.group_name)
            # pylint: disable=W0703
            except Exception as e:
                logger.exception("Failed to warmup migration backend "
                                 "(group_name: {}, world_size: {}, rank: {}, backbend: {}), "
                                 "unexpected exception: {}".format(
                                     self.group_name, self.global_world_size, self.global_rank, self.backend, e))
                return False

        logger.info("Migration backend warmup successfully (group_name: {}, world_size: {}, rank: {}, backbend: {})."
                    .format(self.group_name, self.global_world_size, self.global_rank, self.backend))
        return True

    # Ray.collective is used to construct the gloo and nccl backends. The do_send/do_recv functions will transmit
    # data layer by layer. Take into consideration that col.send/recv are blocking operations.
    def recv_cache(self,
                   request_id: str,
                   src_worker_handle: ray.actor.ActorHandle,
                   src_blocks: List[int],
                   dst_blocks: List[int],
                   is_last_stage: bool) -> None:
        tot_blocks = len(src_blocks)
        src_rank = ray_get_with_timeout(self.proxy_actor.exec_method.remote(src_worker_handle, "get_global_rank"))

        src_seq_group_metadata = None
        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            is_last_comm = (tot_blocks - start_idx <= self.num_migration_buffer_blocks)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
            send_worker_metadata = self.use_ray_spmd_worker and is_last_stage and is_last_comm
            ray_obj = self.proxy_actor.exec_method.remote(
                src_worker_handle,
                "do_send",
                self.global_rank,
                send_blocks,
                request_id=request_id,
                send_worker_metadata=send_worker_metadata
            )
            # Ray collective communication does not have timeout parameters,
            # and run this method in another thread to set timeout will also cause cuda stream device mismatch error,
            # so recv cache does not have timeout only when the migration backend is ray collective.
            self.do_recv(src_rank, recv_blocks)
            if send_worker_metadata:
                _, src_seq_group_metadata = ray_get_with_timeout(ray_obj)
        if src_seq_group_metadata:
            self.worker_stage_seq_group_metadata_callback(request_id, src_seq_group_metadata)

    def do_send(self, dst_worker_handle: ray.actor.ActorHandle, blocks: List[int], virtuel_engine: int=0) -> None:
        num_blocks = len(blocks)
        send_cache = self.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((blocks[idx], idx))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu", pin_memory=True).view(-1, 2)
        with self.migration_stream:
            for layer_idx in range(self.cache_engine[0].num_attention_layers):
                cache_idx = layer_idx % self.migration_num_layers
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(self.gpu_cache[virtuel_engine][layer_idx], send_cache[cache_idx], block_mapping_tensor)
                if cache_idx + 1 == self.migration_num_layers or layer_idx + 1 == self.cache_engine[0].num_attention_layers:
                    # TODO(KuilongCui): check the error code if peer is dead
                    col.send(send_cache, dst_worker_handle, self.group_name)
        self.migration_stream.synchronize()

    def do_recv(self, src_worker_handle: ray.actor.ActorHandle, blocks: List[int], virtuel_engine: int=0) -> None:
        num_blocks = len(blocks)
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((idx, blocks[idx]))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu", pin_memory=True).view(-1, 2)
        recv_cache = self.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)

        with self.migration_stream:
            for layer_idx in range(self.cache_engine[0].num_attention_layers):
                cache_idx = layer_idx % self.migration_num_layers
                if cache_idx == 0:
                    col.recv(recv_cache, src_worker_handle, self.group_name)
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(recv_cache[cache_idx], self.gpu_cache[virtuel_engine][layer_idx], block_mapping_tensor)
        self.migration_stream.synchronize()


# TODO(s5u13b): Remove unnescessary args.
def get_migration_backend(instance_id: str,
                          migration_config: MigrationConfig,
                          cache_engine: List[CacheEngine],
                          worker_handle_list: List[ray.actor.ActorHandle],
                          scheduling_strategy: PlacementGroupSchedulingStrategy,
                          is_driver_worker: bool,
                          gpu_cache: Optional[List[List[torch.Tensor]]],
                          worker_rank: int,
                          local_rank: int,
                          use_ray_spmd_worker: bool,
                          worker_stage_seq_group_metadata_callback: Callable) -> MigrationBackendBase:
    assert migration_config.migration_backend in ['rayrpc', 'gloo', 'nccl'], \
        "Only support rayrpc, gloo and nccl migration backend for vLLM."

    if cache_engine[0].num_gpu_blocks < migration_config.migration_buffer_blocks:
        logger.warning("migration_cache_blocks({}) is larger than num_gpu_blocks({}), reducing it to num_gpu_blocks."
                       .format(migration_config.migration_buffer_blocks, cache_engine[0].num_gpu_blocks))
        migration_config.migration_buffer_blocks = cache_engine[0].num_gpu_blocks

    target_migration_backend = None
    backend = migration_config.migration_backend

    if backend in ['nccl', 'gloo']:
        target_migration_backend = RayColMigrationBackend(instance_id,
                                                          migration_config,
                                                          cache_engine,
                                                          local_rank,
                                                          scheduling_strategy,
                                                          is_driver_worker,
                                                          gpu_cache,
                                                          use_ray_spmd_worker,
                                                          worker_stage_seq_group_metadata_callback)
    else:
        target_migration_backend = RayRpcMigrationBackend(instance_id,
                                                          migration_config,
                                                          cache_engine,
                                                          worker_rank,
                                                          worker_handle_list,
                                                          scheduling_strategy,
                                                          is_driver_worker,
                                                          gpu_cache,
                                                          use_ray_spmd_worker,
                                                          worker_stage_seq_group_metadata_callback)

    return target_migration_backend
