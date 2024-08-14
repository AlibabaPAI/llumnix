from abc import ABC, abstractmethod
from typing import List
import torch
from func_timeout import func_set_timeout, FunctionTimedOut

import ray
import pygloo
import ray.util.collective as col
import ray.util.collective.collective_group.gloo_util as gloo_util
from vllm.logger import init_logger
from vllm.worker.cache_engine import CacheEngine
from llumnix.config import MigrationConfig

logger = init_logger(__name__)

class MigrationBackendBase(ABC):
    @abstractmethod
    def init_col(self, name, world_size, rank) -> bool:
        raise NotImplementedError

    @abstractmethod
    def destory_col(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def do_send(self, dst_handle, blocks: List[int]):
        raise NotImplementedError

    @abstractmethod
    def do_recv(self, src_handle, blocks: List[int]):
        raise NotImplementedError

@ray.remote(num_cpus=0)
class ProxyActor:
    def exec_method(self, is_driver_worker, handle, *args, **kwargs):
        try:
            if is_driver_worker:
                ret = ray.get(handle.execute_engine_method.remote("execute_worker_method", *args, **kwargs))
            else:
                ret = ray.get(handle.execute_method.remote(*args, **kwargs))
        # pylint: disable=try-except-raise
        except:
            raise

        return ret

NUMPY_SUPPORT_DTYPES = [torch.float32, torch.float16]

class RayRpcMigrationBackend(MigrationBackendBase):
    def __init__(self, migration_config: MigrationConfig, cache_engine: CacheEngine,  worker_rank, worker_handle_list, \
                  scheduling_strategy, is_driver_worker, gpu_cache) -> None:
        super().__init__()

        self.migration_config = migration_config
        self.cache_engine = cache_engine

        self.worker_rank = worker_rank
        self.worker_handle_list = worker_handle_list
        self.actor = ProxyActor.options(scheduling_strategy=scheduling_strategy).remote()

        self.rpc_dtype = self.cache_engine.dtype
        if self.cache_engine.dtype in NUMPY_SUPPORT_DTYPES:
            self.rpc_dtype = self.cache_engine.dtype
        else:
            self.rpc_dtype = torch.float32
            logger.warning("Detecting numpy unsupported dtype: {}. Using torch.float32.".format(self.cache_engine.dtype))

        self.is_driver_worker = is_driver_worker
        self.gpu_cache = gpu_cache
        self.cache_device = "cpu"
        self.num_migration_cache_blocks = self.migration_config.migration_cache_blocks
        self.num_layers = self.cache_engine.num_layers
        migration_cache_size = self.cache_engine.block_size * self.cache_engine.num_heads * self.cache_engine.head_size

        self.dummy_cache = torch.empty(
            size=(self.num_layers, 2, self.num_migration_cache_blocks, self.migration_cache_size),
            dtype=self.cache_engine.dtype,
            device=self.cache_device,
            pin_memory=True
        )
        self.migration_stream = torch.cuda.Stream()

    def init_col(self, name, world_size, rank) -> bool:
        logger.info("create rpc migrate backend successfully.")
        return True

    def destory_col(self) -> None:
        logger.info("destory rpc migrate backend successfully.")

    def warmup(self) -> None:
        self.actor.exec_method.remote(self.is_driver_worker, "do_send", [0])

    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        tot_blocks = len(src_blocks)
        rpc_numpy_cache = None
        for start_idx in range(0, tot_blocks, self.num_migration_cache_blocks):
            offset = min(self.num_migration_cache_blocks, tot_blocks - start_idx)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            ray_obj = self.actor.exec_method.remote(self.is_driver_worker, src_handle, "do_send", send_blocks)
            if rpc_numpy_cache is not None:
                self.do_recv(rpc_numpy_cache, recv_blocks)
            rpc_numpy_cache = ray.get(ray_obj)
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
        self.do_recv(rpc_numpy_cache, recv_blocks)

    def do_send(self, blocks: List[int]):
        num_blocks = len(blocks)
        data = self.dummy_cache[:,:,:num_blocks,:]
        dummy_key_cpu = self.dummy_cache[:,0,:num_blocks,:]
        dummy_value_cpu = self.dummy_cache[:,1,:num_blocks,:]
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.num_layers):
                for idx, block_num in enumerate(blocks):
                    dummy_key_cpu[layer_idx][idx].copy_(self.gpu_cache[layer_idx][0][block_num], non_blocking=True)
                    dummy_value_cpu[layer_idx][idx].copy_(self.gpu_cache[layer_idx][1][block_num], non_blocking=True)
        torch.cuda.Stream.synchronize(self.migration_stream)
        return data.to(self.rpc_dtype).numpy()

    def do_recv(self, rpc_numpy_cache, blocks: List[int]):
        num_blocks = len(blocks)
        # use pin memory dummy_cache to speed up data transfer
        self.dummy_cache[:,:,:num_blocks,:].copy_(torch.from_numpy(rpc_numpy_cache))
        dummy_key_cpu = self.dummy_cache[:,0,:num_blocks,:]
        dummy_value_cpu = self.dummy_cache[:,1,:num_blocks,:]

        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.num_layers):
                for idx, block_num in enumerate(blocks):
                    self.gpu_cache[layer_idx][0][block_num].copy_(dummy_key_cpu[layer_idx][idx], non_blocking=True)
                    self.gpu_cache[layer_idx][1][block_num].copy_(dummy_value_cpu[layer_idx][idx], non_blocking=True)
        torch.cuda.Stream.synchronize(self.migration_stream)

class RayColMigrationBackend(MigrationBackendBase):
    def __init__(self, migration_config: MigrationConfig, cache_engine: CacheEngine, local_rank,
                 scheduling_strategy, is_driver_worker, gpu_cache) -> None:
        super().__init__()

        self.migration_config = migration_config
        self.cache_engine = cache_engine
        self.backend = migration_config.migration_backend
        self.num_migration_num_layers = min(migration_config.migration_num_layers, self.cache_engine.num_layers)
        self.num_migration_cache_blocks = migration_config.migration_cache_blocks

        self.backend = migration_config.migration_backend
        self.ray_world_size = -1
        self.ray_rank = -1
        self.group_name = None

        self.local_rank = local_rank
        self.actor = ProxyActor.options(scheduling_strategy=scheduling_strategy).remote()
        self.is_driver_worker = is_driver_worker
        self.gpu_cache = gpu_cache
        # add bf16 type
        gloo_util.TORCH_GLOO_DTYPE_MAP[torch.bfloat16] = pygloo.glooDataType_t.glooFloat16

        migration_cache_size = self.cache_engine.block_size * self.cache_engine.num_heads * self.cache_engine.head_size

        if self.backend == 'gloo':
            self.cache_device = "cpu"
        elif self.backend == 'nccl':
            self.cache_device = torch.device(f"cuda:{self.local_rank}")
        else:
            raise ValueError("backend must be 'gloo' or 'nccl'")

        pin_memory = (self.backend == 'gloo')
        self.dummy_cache= torch.empty(
            size=(self.num_migration_num_layers, 2, self.num_migration_cache_blocks, migration_cache_size),
            dtype=self.cache_engine.dtype,
            device=self.cache_device,
            pin_memory=pin_memory
        )

        self.migration_stream = torch.cuda.Stream()

    def init_col(self, group_name, world_size, rank) -> bool:
        @func_set_timeout(self.migration_config.migration_backend_init_timeout)
        def init_group(world_size, rank, backend, group_name):
            col.init_collective_group(world_size, rank, backend, group_name)

        try:
            init_group(world_size, rank, self.backend, group_name)
        except FunctionTimedOut:
            logger.info("create ray collective group fail (group_name:{}, world_size: {}, rank: {}, backbend: {})."
                .format(group_name, world_size, rank, self.backend))
            return False

        self.group_name = group_name
        self.ray_world_size = world_size
        self.ray_rank = rank

        logger.info("create ray collective group successfully (group_name:{}, world_size: {}, rank: {}, backbend: {})."
                    .format(self.group_name, self.ray_world_size, self.ray_rank, self.backend))
        return True

    def warmup(self):
        if self.ray_world_size > 1 and self.group_name is not None:
            try:
                col.allreduce(self.dummy_cache[0], self.group_name)
            except Exception as e:
                logger.info("warmup collective group failed (group_name:{}, world_size: {}, rank: {}, backbend: {})."
                    .format(self.group_name, self.ray_world_size, self.ray_rank, self.backend))
                return False

        logger.info("ray collective group warmup successfully (group_name:{}, world_size: {}, rank: {}, backbend: {})."
                    .format(self.group_name, self.ray_world_size, self.ray_rank, self.backend))
        return True

    def destory_col(self) -> None:
        if self.group_name is not None:
            err_info = None
            try:
                col.destroy_collective_group(self.group_name)
            except Exception as e:
                err_info = e
            logger.info("destory ray collective group successfully (group_name:{}, backbend: {}), neet_err: {}."
                        .format(self.group_name, self.backend, err_info))
            self.group_name = None

    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        tot_blocks = len(src_blocks)
        src_rank = ray.get(self.actor.exec_method.remote(self.is_driver_worker, src_handle, "get_ray_rank"))

        for start_idx in range(0, tot_blocks, self.num_migration_cache_blocks):
            offset = min(self.num_migration_cache_blocks, tot_blocks - start_idx)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
            self.actor.exec_method.remote(self.is_driver_worker, src_handle, "do_send", self.ray_rank, send_blocks)
            self.do_recv(src_rank, recv_blocks)

    def do_send(self, dst_handle, blocks: List[int]):
        src_to_dst = {block_num: idx for idx, block_num in enumerate(blocks)}
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.cache_engine.num_layers):
                cache_idx = layer_idx % self.num_migration_num_layers
                self.cache_engine.attn_backend.swap_blocks(self.gpu_cache[layer_idx], self.dummy_cache[cache_idx], src_to_dst)
                if cache_idx + 1 == self.num_migration_num_layers or layer_idx + 1 == self.cache_engine.num_layers:
                    col.send(self.dummy_cache, dst_handle, self.group_name)
        torch.cuda.Stream.synchronize(self.migration_stream)

    def do_recv(self, src_handle, blocks: List[int]):
        src_to_dst = dict(enumerate(blocks))
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.cache_engine.num_layers):
                cache_idx = layer_idx % self.num_migration_num_layers
                if cache_idx == 0:
                    col.recv(self.dummy_cache, src_handle, self.group_name)
                self.cache_engine.attn_backend.swap_blocks(self.dummy_cache[cache_idx], self.gpu_cache[layer_idx], src_to_dst)
        torch.cuda.Stream.synchronize(self.migration_stream)

def get_migrate_backend(migration_config: MigrationConfig, cache_engine: CacheEngine, worker_handle_list, scheduling_strategy,
                        is_driver_worker, gpu_cache, worker_rank, local_rank) -> MigrationBackendBase:
    if migration_config.model_parallelism_enabled and migration_config.migration_backend == 'nccl':
        logger.warning("NCCL backend is not supported for PP or TP enabled model, using gloo instead.")
        migration_config.migration_backend = 'gloo'

    if cache_engine.num_gpu_blocks < migration_config.migration_cache_blocks:
        logger.warning("migration_cache_blocks({}) is larger than num_gpu_blocks({}), reducing it to num_gpu_blocks."
                       .format(migration_config.migration_cache_blocks, cache_engine.num_gpu_blocks))
        migration_config.migration_cache_blocks = cache_engine.num_gpu_blocks

    target_col = None
    backend = migration_config.migration_backend
    if backend in ['nccl', 'gloo']:
        target_col = RayColMigrationBackend(migration_config, cache_engine, local_rank, scheduling_strategy,
                                            is_driver_worker, gpu_cache)
    elif backend == 'rpc':
        target_col = RayRpcMigrationBackend(migration_config, cache_engine, worker_rank, worker_handle_list,
                                            scheduling_strategy, is_driver_worker, gpu_cache)
    else:
        raise ValueError(f"Unsupported backend {backend}")

    return target_col
