from abc import ABC, abstractmethod
from typing import List
import torch
from func_timeout import func_set_timeout, FunctionTimedOut

import ray
import ray.util.collective as col
from vllm.logger import init_logger
from vllm.worker.cache_engine import CacheEngine
from llumnix.config import MigrationConfig

logger = init_logger(__name__)

class MigrateBackendBase(ABC):
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

class RPCMigrateBackend(MigrateBackendBase):
    def __init__(self, migrate_config: MigrationConfig, cache_engine: CacheEngine,  worker_rank, worker_handle_list, \
                  scheduling_strategy, is_driver_worker, gpu_cache) -> None:
        super().__init__()

        self.migrate_config = migrate_config
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
        self.num_migration_cache_blocks = self.migrate_config.migration_cache_blocks
        self.num_layers = self.cache_engine.num_layers
        self.migration_cache_size = self.cache_engine.block_size * self.cache_engine.num_heads * self.cache_engine.head_size

        self.dummy_cache = torch.empty(
            size=(2*self.num_migration_cache_blocks, self.num_layers, self.migration_cache_size),
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
        self_handle = self.worker_handle_list[self.worker_rank]
        self.actor.exec_method.remote(self.is_driver_worker, self_handle, "do_send", None, [0])

    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        tot_blocks = len(src_blocks)
        rpc_numpy_cache = None
        for start_idx in range(0, tot_blocks, self.num_migration_cache_blocks):
            offset = min(self.num_migration_cache_blocks, tot_blocks - start_idx)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            ray_obj = self.actor.exec_method.remote(self.is_driver_worker, src_handle, "do_send", None, send_blocks)
            if rpc_numpy_cache is not None:
                self.do_recv(rpc_numpy_cache, recv_blocks)
            rpc_numpy_cache = ray.get(ray_obj)
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
        self.do_recv(rpc_numpy_cache, recv_blocks)

    def do_send(self, dst_handle, blocks: List[int]):
        num_blocks = len(blocks)
        data = self.dummy_cache[:2*num_blocks]
        dummy_key_cpu = self.dummy_cache[:num_blocks]
        dummy_value_cpu = self.dummy_cache[num_blocks:2*num_blocks]
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.num_layers):
                for idx, block_num in enumerate(blocks):
                    dummy_key_cpu[idx][layer_idx].copy_(self.gpu_cache[layer_idx][0][block_num], non_blocking=True)
                    dummy_value_cpu[idx][layer_idx].copy_(self.gpu_cache[layer_idx][1][block_num], non_blocking=True)
        torch.cuda.Stream.synchronize(self.migration_stream)
        return data.to(self.rpc_dtype).numpy()

    def do_recv(self, src_handle, blocks: List[int]):
        num_blocks = len(blocks)

        # use pin memory dummy_cache to speed up data transfer
        data = self.dummy_cache[:2*num_blocks].copy_(torch.from_numpy(src_handle))
        dummy_key = data[:num_blocks]
        dummy_value = data[num_blocks:2*num_blocks]

        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.num_layers):
                for idx, block_num in enumerate(blocks):
                    self.gpu_cache[layer_idx][0][block_num].copy_(dummy_key[idx][layer_idx], non_blocking=True)
                    self.gpu_cache[layer_idx][1][block_num].copy_(dummy_value[idx][layer_idx], non_blocking=True)
        torch.cuda.Stream.synchronize(self.migration_stream)

class RayMigrateBackend(MigrateBackendBase):
    def __init__(self, migrate_config: MigrationConfig, cache_engine: CacheEngine, local_rank, scheduling_strategy, 
                 is_driver_worker, gpu_cache, backend_init_timeout) -> None:
        super().__init__()

        self.migrate_config = migrate_config
        self.cache_engine = cache_engine
        self.num_migration_cache_blocks = migrate_config.migration_cache_blocks

        self.backend = migrate_config.migration_backend
        self.ray_world_size = -1
        self.ray_rank = -1
        self.group_name = None
        self.init_timeout = backend_init_timeout

        self.local_rank = local_rank
        self.actor = ProxyActor.options(scheduling_strategy=scheduling_strategy).remote()
        self.is_driver_worker = is_driver_worker
        self.gpu_cache = gpu_cache

        self.migration_cache_size = self.cache_engine.block_size * self.cache_engine.num_heads * self.cache_engine.head_size

        if self.backend == 'gloo':
            self.cache_device = "cpu"
        elif self.backend == 'nccl':
            self.cache_device = torch.device(f"cuda:{self.local_rank}")
        else:
            raise ValueError("backend must be 'gloo' or 'nccl'")

        pin_memory = self.backend == 'gloo'
        self.dummy_cache = torch.empty(
            size=(2*self.num_migration_cache_blocks, self.cache_engine.num_layers, self.migration_cache_size),
            dtype=self.cache_engine.dtype,
            device=self.cache_device,
            pin_memory=pin_memory
        )

        self.migration_stream = torch.cuda.Stream()

    def init_col(self, group_name, world_size, rank) -> bool:
        @func_set_timeout(self.init_timeout)
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
        num_blocks = len(blocks)
        data = self.dummy_cache[:2*num_blocks]
        dummy_key_cpu = data[:num_blocks]
        dummy_value_cpu = data[num_blocks:2*num_blocks]
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.cache_engine.num_layers):
                for idx, block_num in enumerate(blocks):
                    dummy_key_cpu[idx][layer_idx].copy_(self.gpu_cache[layer_idx][0][block_num], non_blocking=True)
                    dummy_value_cpu[idx][layer_idx].copy_(self.gpu_cache[layer_idx][1][block_num], non_blocking=True)
        torch.cuda.Stream.synchronize(self.migration_stream)
        col.send(data, dst_handle, self.group_name)

    def do_recv(self, src_handle, blocks: List[int]):
        num_blocks = len(blocks)
        data = self.dummy_cache[:2*num_blocks]
        # note that col.recv use ray.collective inner stream, not migration_stream
        col.recv(data, src_handle, self.group_name)

        dummy_key_cpu = data[:num_blocks]
        dummy_value_cpu = data[num_blocks:2*num_blocks]
        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.cache_engine.num_layers):
                for idx, block_num in enumerate(blocks):
                    self.gpu_cache[layer_idx][0][block_num].copy_(dummy_key_cpu[idx][layer_idx], non_blocking=True)
                    self.gpu_cache[layer_idx][1][block_num].copy_(dummy_value_cpu[idx][layer_idx], non_blocking=True)
        torch.cuda.Stream.synchronize(self.migration_stream)

def get_migrate_backend(migrate_config: MigrationConfig, cache_engine: CacheEngine, worker_handle_list, scheduling_strategy, 
                        is_driver_worker, gpu_cache, worker_rank, local_rank) -> MigrateBackendBase:
    if migrate_config.pp_or_tp_enabled and migrate_config.migration_backend == 'nccl':
        logger.warning("NCCL backend is not supported for PP or TP enabled model, using gloo instead.")
        migrate_config.migration_backend = 'gloo'

    if cache_engine.num_gpu_blocks < migrate_config.migration_cache_blocks:
        logger.warning("migration_cache_blocks({}) is larger than num_gpu_blocks({}), reducing it to num_gpu_blocks."
                       .format(migrate_config.migration_cache_blocks, cache_engine.num_gpu_blocks))
        migrate_config.migration_cache_blocks = cache_engine.num_gpu_blocks

    target_col = None
    backend = migrate_config.migration_backend
    if backend in ['nccl', 'gloo']:
        target_col = RayMigrateBackend(migrate_config, cache_engine, local_rank, scheduling_strategy, is_driver_worker, 
                                       gpu_cache, migrate_config.migration_backend_init_timeout)
    elif backend == 'rpc':
        target_col = RPCMigrateBackend(migrate_config, cache_engine, worker_rank, worker_handle_list, scheduling_strategy, 
                                       is_driver_worker, gpu_cache)
    else:
        raise ValueError(f"Unsupported backend {backend}")

    return target_col
