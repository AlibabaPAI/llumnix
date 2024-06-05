"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl
from vllm.model_executor.parallel_utils.parallel_state import get_instance_parallel_group

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
    def send_gpu_cache_batch(self, dst_rank, send_blocks):
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        dummy_key = torch.empty(
            size=(len(send_blocks), *key_block_shape),
            dtype=self.dtype,
            device="cuda")
        dummy_value = torch.empty(
            size=(len(send_blocks), *value_block_shape),
            dtype=self.dtype,
            device="cuda")
        inds_key = torch.zeros_like(dummy_key, dtype=torch.int64)
        inds_value = torch.zeros_like(dummy_value, dtype=torch.int64)
        block_tensor_key = torch.tensor(
            send_blocks, dtype=torch.int64, device="cuda").view(-1, 1, 1, 1, 1)
        block_tensor_value = torch.tensor(
            send_blocks, dtype=torch.int64, device="cuda").view(-1, 1, 1, 1)
        inds_key.add_(block_tensor_key)
        inds_value.add_(block_tensor_value)
        for i in range(self.num_layers):
            torch.gather(self.gpu_cache[i][0], 0, inds_key, out=dummy_key)
            send_key_op = torch.distributed.P2POp(torch.distributed.isend, dummy_key, dst_rank)
            torch.gather(self.gpu_cache[i][1], 0, inds_value, out=dummy_value)
            send_value_op = torch.distributed.P2POp(torch.distributed.isend, dummy_value, dst_rank)
            reqs = torch.distributed.batch_isend_irecv([send_key_op, send_value_op])
            for req in reqs:
                req.wait()
            # torch.distributed.send(dummy_key, dst_rank)
            # torch.distributed.send(dummy_value, dst_rank)
            # torch.cuda.synchronize()
        torch.cuda.synchronize()
        # logger.info(f"total gather time:{tot_gather_time*1000}ms")
    def recv_gpu_cache_batch(self, src_rank, recv_blocks):
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        dummy_key = torch.empty(
            size=(len(recv_blocks), *key_block_shape),
            dtype=self.dtype,
            device="cuda")
        dummy_value = torch.empty(
            size=(len(recv_blocks), *value_block_shape),
            dtype=self.dtype,
            device="cuda")
        inds_key = torch.zeros_like(dummy_key, dtype=torch.int64)
        inds_value = torch.zeros_like(dummy_value, dtype=torch.int64)
        block_tensor_key = torch.tensor(
            recv_blocks, dtype=torch.int64, device="cuda").view(-1, 1, 1, 1, 1)
        block_tensor_value = torch.tensor(
            recv_blocks, dtype=torch.int64, device="cuda").view(-1, 1, 1, 1)
        inds_key.add_(block_tensor_key)
        inds_value.add_(block_tensor_value)
        for i in range(self.num_layers):
            recv_key_op = torch.distributed.P2POp(torch.distributed.irecv, dummy_key, src_rank)
            recv_value_op = torch.distributed.P2POp(torch.distributed.irecv, dummy_value, src_rank)
            # torch.distributed.recv(dummy_key, src_rank)
            # torch.distributed.recv(dummy_value, src_rank)
            reqs = torch.distributed.batch_isend_irecv([recv_key_op, recv_value_op])
            reqs[0].wait()
            self.gpu_cache[i][0].scatter_(0, inds_key, dummy_key)
            reqs[1].wait()
            self.gpu_cache[i][1].scatter_(0, inds_value, dummy_value)
            # if i<1:
            #     logger.info(f"recv layer{i} from rank{src_rank},value{self.gpu_cache[i][1][recv_blocks]}")
        torch.cuda.synchronize()
    def send_gpu_cache(self, dst_rank, send_blocks):
        with torch.cuda.stream(self.cache_stream):
            group = None
            if self.parallel_config.migrate_backend == "gloo":
                group = get_instance_parallel_group()
            key_block_shape = self.get_key_block_shape()
            value_block_shape = self.get_value_block_shape()
            dummy_key = torch.empty(
                size=(len(send_blocks), *key_block_shape),
                dtype=self.dtype,
                device="cuda")
            dummy_value = torch.empty(
                size=(len(send_blocks), *value_block_shape),
                dtype=self.dtype,
                device="cuda")
            inds_key = torch.zeros_like(dummy_key, dtype=torch.int64)
            inds_value = torch.zeros_like(dummy_value, dtype=torch.int64)
            block_tensor_key = torch.tensor(
                send_blocks, dtype=torch.int64, device="cuda").view(-1, 1, 1, 1, 1)
            block_tensor_value = torch.tensor(
                send_blocks, dtype=torch.int64, device="cuda").view(-1, 1, 1, 1)
            inds_key.add_(block_tensor_key)
            inds_value.add_(block_tensor_value)
            for i in range(self.num_layers):
                torch.gather(self.gpu_cache[i][0], 0, inds_key, out=dummy_key)
                torch.gather(self.gpu_cache[i][1], 0, inds_value, out=dummy_value)
                if self.parallel_config.migrate_backend == "nccl":
                    torch.distributed.send(dummy_key, dst_rank, group=group)
                    torch.distributed.send(dummy_value, dst_rank, group=group)
                else:
                    torch.distributed.send(dummy_key.cpu(), dst_rank, group=group)
                    torch.distributed.send(dummy_value.cpu(), dst_rank, group=group)
        torch.cuda.Stream.synchronize(self.cache_stream)

    def recv_gpu_cache(self, src_rank, recv_blocks):
        with torch.cuda.stream(self.cache_stream):
            group = None
            if self.parallel_config.migrate_backend == "gloo":
                group = get_instance_parallel_group()
            key_block_shape = self.get_key_block_shape()
            value_block_shape = self.get_value_block_shape()
            dummy_key = torch.empty(
                size=(len(recv_blocks), *key_block_shape),
                dtype=self.dtype,
                device="cuda" if self.parallel_config.migrate_backend == "nccl" else "cpu")
            dummy_value = torch.empty(
                size=(len(recv_blocks), *value_block_shape),
                dtype=self.dtype,
                device="cuda" if self.parallel_config.migrate_backend == "nccl" else "cpu")
            inds_key = torch.zeros_like(dummy_key, dtype=torch.int64, device="cuda")
            inds_value = torch.zeros_like(dummy_value, dtype=torch.int64, device="cuda")
            block_tensor_key = torch.tensor(
                recv_blocks, dtype=torch.int64, device="cuda").view(-1, 1, 1, 1, 1)
            block_tensor_value = torch.tensor(
                recv_blocks, dtype=torch.int64, device="cuda").view(-1, 1, 1, 1)
            inds_key.add_(block_tensor_key)
            inds_value.add_(block_tensor_value)
            for i in range(self.num_layers):
                torch.distributed.recv(dummy_key, src_rank, group=group)
                torch.distributed.recv(dummy_value, src_rank, group=group)
                if self.parallel_config.migrate_backend == "nccl":
                    self.gpu_cache[i][0].scatter_(0, inds_key, dummy_key)
                    self.gpu_cache[i][1].scatter_(0, inds_value, dummy_value)
                else:
                    self.gpu_cache[i][0].scatter_(0, inds_key, dummy_key.cuda())
                    self.gpu_cache[i][1].scatter_(0, inds_value, dummy_value.cuda())
        torch.cuda.Stream.synchronize(self.cache_stream)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
