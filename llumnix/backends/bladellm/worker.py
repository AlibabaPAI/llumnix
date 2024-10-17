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

from concurrent.futures import ThreadPoolExecutor
import asyncio
import gc
import sys
import torch
import grpc
from google.protobuf import empty_pb2
import loguru

from blade_llm.service.args import ServingArgs
from blade_llm.service.worker import RemoteManager
from blade_llm.utils.network import get_free_port
from blade_llm.service.proto import bladellm_pb2_grpc
from blade_llm.service.worker import Worker

from llumnix.backends.bladellm.proto import (
    llumnix_bladellm_pb2,
    llumnix_bladellm_pb2_grpc,
)
from llumnix.logger import init_logger

logger = init_logger(__name__)
NUMPY_SUPPORT_DTYPES = [torch.float32, torch.float16]


def recv_cpu_cache(src_worker_handle, src_blocks):
    """
    Args:
        src_worker_handle: src worker client handle
        blocks: block to send
    """
    try:
        k, v = src_worker_handle.send_cpu_cache(src_blocks)
    # pylint: disable=try-except-raise
    except:
        raise
    return k, v


class LlumnixWorker(llumnix_bladellm_pb2_grpc.LlumnixWorkerServicer, Worker):
    def __init__(self, *args, **kwargs) -> None:
        # replace sampler
        # pylint: disable=import-outside-toplevel
        super().__init__(*args, **kwargs)

        # too many logs in BladeLLM, redefine the log level
        loguru.logger.remove()
        loguru.logger.add(sys.stderr, level="INFO")

    # pylint: disable=unused-argument
    def allocate_migration_cache(
        self, request: llumnix_bladellm_pb2.AllocRequest, context: grpc.ServicerContext
    ):
        self.migration_stream = torch.cuda.Stream()
        self.default_stream = torch.cuda.current_stream()
        # TODO(ziming) make num_migration_cache_blocks configurable
        self.num_migration_cache_blocks = request.num_migration_cache_blocks
        assert self.migration_stream != self.default_stream
        # pylint: disable=protected-access
        state_manager = self._engine._state_manager
        num_kv_heads = (
            state_manager.model_conf.num_attention_heads // state_manager.tp_size
            if state_manager.model_conf.num_query_group is None
            else max(
                1, state_manager.model_conf.num_query_group // state_manager.tp_size
            )
        )
        migration_cache_key_shape = (
            self.num_migration_cache_blocks,
            state_manager.num_layers,
            num_kv_heads,
            state_manager.model_conf.head_dim // state_manager.x,
            state_manager.block_size,
            state_manager.x,
        )
        migration_cache_value_shape = (
            self.num_migration_cache_blocks,
            state_manager.num_layers,
            num_kv_heads,
            state_manager.block_size,
            state_manager.model_conf.head_dim,
        )
        if state_manager.dtype in NUMPY_SUPPORT_DTYPES:
            migration_cache_dtype = state_manager.dtype
        else:
            migration_cache_dtype = torch.float32
            logger.warning(
                "Detecting numpy unsupported dtype: {}.".format(state_manager.dtype),
                "Using torch.float32.",
            )
        self.migration_key_cache = torch.empty(
            size=migration_cache_key_shape,
            dtype=migration_cache_dtype,
        )
        self.migration_value_cache = torch.empty(
            size=migration_cache_value_shape,
            dtype=migration_cache_dtype,
        )
        return empty_pb2.Empty()

    # pylint: disable=unused-argument
    def send_cpu_cache(self, request: llumnix_bladellm_pb2.SendRequest, context: grpc.ServicerContext):
        num_blocks = len(request.blocks)
        dummy_key_cpu = self.migration_key_cache[:num_blocks]
        dummy_value_cpu = self.migration_value_cache[:num_blocks]
        with torch.cuda.stream(self.migration_stream):
            # pylint: disable=protected-access
            for layer_idx in range(self._engine._state_manager.num_layers):
                for idx, block_num in enumerate(request.blocks):
                    dummy_key_cpu[idx][layer_idx].copy_(
                        self._engine._state_manager._kv_cache[0][layer_idx][block_num]
                    )
                    dummy_value_cpu[idx][layer_idx].copy_(
                        self._engine._state_manager._kv_cache[1][layer_idx][block_num]
                        )
        torch.cuda.Stream.synchronize(self.migration_stream)
        return dummy_key_cpu.numpy(), dummy_value_cpu.numpy()

    # pylint: disable=unused-argument
    def migrate_gpu_cache_grpc(
        self, request: llumnix_bladellm_pb2.MigrateRequest, context: grpc.ServicerContext
    ):
        with torch.cuda.stream(self.migration_stream):
            src_worker_handle = request.src_worker_handle[self._rank]
            tot_blocks = len(request.src_blocks)
            for start_idx in range(0, tot_blocks, self.num_migration_cache_blocks):
                # send/recv num_migration_cache_blocks per iter
                # TODO(ziming): overlap get_numpy_cache with prev idx H2D copy
                offset = min(
                    self.num_migration_cache_blocks, tot_blocks - start_idx
                )
                send_blocks = request.src_blocks[start_idx : start_idx + offset]
                recv_blocks = request.dst_blocks[start_idx : start_idx + offset]
                k, v = self.recv_cpu_cache(src_worker_handle, send_blocks)
                dummy_key = self.migration_key_cache[:offset]
                dummy_value = self.migration_value_cache[:offset]
                dummy_key.copy_(torch.from_numpy(k))
                dummy_value.copy_(torch.from_numpy(v))
                # pylint: disable=protected-access
                for layer_idx in range(self._engine._state_manager.num_layers):
                    for idx, block_num in enumerate(recv_blocks):
                        self._engine._state_manager._kv_cache[0][layer_idx][block_num].copy_(dummy_key[idx][layer_idx])
                        self._engine._state_manager._kv_cache[1][layer_idx][block_num].copy_(dummy_value[idx][layer_idx])
        torch.cuda.Stream.synchronize(self.migration_stream)
        return empty_pb2.Empty()


def worker_main(rank: int, args: ServingArgs, inst_id=None):
    asyncio.run(worker_server(rank, args, inst_id))


async def worker_server(rank: int, args: ServingArgs, inst_id=None):
    if args.server_ip:
        worker_port = int(get_free_port())
        await RemoteManager.start_watch_dog(args, worker_port)
        await RemoteManager.wait_until_all_workers_ready()
    worker = LlumnixWorker(rank, args, inst_id)
    server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=1))
    bladellm_pb2_grpc.add_WorkerServicer_to_server(worker, server)
    llumnix_bladellm_pb2_grpc.add_LlumnixWorkerServicer_to_server(worker, server)
    if inst_id is not None:
        listen_addr = (
            f"0.0.0.0:{worker_port}"
            if args.server_ip
            else f"unix://{args.worker_socket_path}.{inst_id}.{rank}"
        )
    else:
        listen_addr = (
            f"0.0.0.0:{worker_port}"
            if args.server_ip
            else f"unix://{args.worker_socket_path}.{rank}"
        )
    server.add_insecure_port(listen_addr)
    await server.start()
    if args.server_ip:
        await RemoteManager.wait_for_termination(server)
    else:
        await server.wait_for_termination()
    # explicit cleanup
    del server
    del worker
    gc.collect()
