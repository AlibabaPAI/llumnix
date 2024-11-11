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

import os
import time
import signal
from typing import List
import random
import asyncio
from multiprocessing import Process, set_start_method, shared_memory
from concurrent.futures import ThreadPoolExecutor

import pytest
import grpc
from google.protobuf import empty_pb2
import numpy as np
import torch

from blade_llm.service.args import ServingArgs
from blade_llm.utils.load_model_options import LoadModelOptions
from blade_llm.service.proto import bladellm_pb2_grpc
from blade_llm.generation.kvcache.kv_transfer import TransferType
from blade_llm.generation.statemanagers.ragged_flash_state_manager import RaggedFlashStateManager
from blade_llm.generation.statemanagers.paged_state_manager import PagedStateManager

from llumnix.backends.bladellm.remote_worker import MigrationWorker
from llumnix.backends.bladellm.migration_backend import NUMPY_SUPPORTED_DTYPES
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.internal_config import MigrationConfig
from llumnix.arg_utils import EngineManagerArgs

from tests.unit_test.backends.bladellm.proto import mock_migration_worker_pb2_grpc, mock_migration_worker_pb2

class MockMigrationWorker(mock_migration_worker_pb2_grpc.MockMigrationWorkerServicer, MigrationWorker):
    def __init__(self, *args, **kwargs) -> None:
        MigrationWorker.__init__(self, *args, **kwargs)

    def get_kv_cache_meta(self, request, context):
        state_manager = self.migration_backend.state_manager

        if state_manager.dtype in NUMPY_SUPPORTED_DTYPES:
            migration_cache_dtype = state_manager.dtype
        else:
            migration_cache_dtype = torch.float32

        state_manager = self.migration_backend.state_manager
        key_shape, value_shape, kv_shape = [], [], []
        if isinstance(self.migration_backend.state_manager, PagedStateManager):
            key_shape = [len(state_manager._kv_cache[0])] + list(state_manager._kv_cache[0][0].shape)
            value_shape = [len(state_manager._kv_cache[1])] + list(state_manager._kv_cache[1][0].shape)
        elif isinstance(self.migration_backend.state_manager, RaggedFlashStateManager):
            gpu_kv_cache = state_manager.kv_cache_arena.gpu_kv_cache
            kv_shape = [len(gpu_kv_cache)] + list(gpu_kv_cache[0].shape)
        else:
            raise NotImplementedError

        return mock_migration_worker_pb2.KvCacheMeta(
            key_shape=np.array(key_shape, dtype=np.int64).tobytes(),
            value_shape=np.array(value_shape, dtype=np.int64).tobytes(),
            kv_shape=np.array(kv_shape, dtype=np.int64).tobytes(),
            kv_cache_dtype=str(state_manager.dtype),
            migration_cache_dtype=str(migration_cache_dtype)
        )

    def set_gpu_cache(self, request, context):
        state_manager = self.migration_backend.state_manager

        dtype = getattr(np, request.np_dtype, None)
        key_shape = np.frombuffer(request.key_shape, dtype=np.int64)
        value_shape = np.frombuffer(request.value_shape, dtype=np.int64)
        kv_shape = np.frombuffer(request.kv_shape, dtype=np.int64)
        block_idxs = list(request.block_idxs)

        if isinstance(self.migration_backend.state_manager, PagedStateManager):
            key = torch.from_numpy(np.frombuffer(request.key, dtype=dtype).reshape(key_shape))
            value = torch.from_numpy(np.frombuffer(request.value, dtype=dtype).reshape(value_shape))
            gpu_k_cache = self.migration_backend.state_manager._kv_cache[0]
            gpu_v_cache = self.migration_backend.state_manager._kv_cache[1]

            for layer_idx in range(key.shape[0]):
                for src_block_idx, target_block_idx in enumerate(block_idxs):
                    gpu_k_cache[layer_idx][target_block_idx].copy_(key[layer_idx][src_block_idx])
                    gpu_v_cache[layer_idx][target_block_idx].copy_(value[layer_idx][src_block_idx])
        elif isinstance(self.migration_backend.state_manager, RaggedFlashStateManager):
            kv = torch.from_numpy(np.frombuffer(request.kv, dtype=dtype).reshape(kv_shape))
            gpu_kv_cache = state_manager.kv_cache_arena.gpu_kv_cache
            for layer_idx in range(kv.shape[0]):
                for src_block_idx, target_block_idx in enumerate(block_idxs):
                    gpu_kv_cache[layer_idx][target_block_idx].copy_(kv[layer_idx][src_block_idx])
        else:
            raise NotImplementedError

        torch.cuda.synchronize()

        return empty_pb2.Empty()

    def get_gpu_cache(self, request, context):
        state_manager = self.migration_backend.state_manager

        if state_manager.dtype in NUMPY_SUPPORTED_DTYPES:
            migration_cache_dtype = state_manager.dtype
        else:
            migration_cache_dtype = torch.float32

        key, value, kv = bytes(), bytes(), bytes()
        key_tensor, value_tensor, kv_tensor = torch.tensor([]), torch.tensor([]), torch.tensor([])
        if isinstance(self.migration_backend.state_manager, PagedStateManager):
            key = []
            value = []
            gpu_k_cache = self.migration_backend.state_manager._kv_cache[0]
            gpu_v_cache = self.migration_backend.state_manager._kv_cache[1]

            for idx, _ in enumerate(gpu_k_cache):
                key.append(gpu_k_cache[idx][request.block_idxs])
                value.append(gpu_v_cache[idx][request.block_idxs])

            key_tensor = torch.stack(key).type(migration_cache_dtype).cpu()
            value_tensor = torch.stack(value).type(migration_cache_dtype).cpu()

            key = key_tensor.numpy().tobytes()
            value = value_tensor.numpy().tobytes()
            np_dtype=str(key_tensor.numpy().dtype)
        elif isinstance(self.migration_backend.state_manager, RaggedFlashStateManager):
            kv = []
            gpu_kv_cache = state_manager.kv_cache_arena.gpu_kv_cache
            for idx, _ in enumerate(gpu_kv_cache):
                kv.append(gpu_kv_cache[idx][request.block_idxs])
            kv_tensor = torch.stack(kv).type(migration_cache_dtype).cpu()
            kv = kv_tensor.numpy().tobytes()
            np_dtype=str(kv_tensor.numpy().dtype)
        else:
            raise NotImplementedError

        if self.migration_backend.state_manager.dtype in NUMPY_SUPPORTED_DTYPES:
            migration_cache_dtype = self.migration_backend.state_manager.dtype
        else:
            migration_cache_dtype = torch.float32

        torch.cuda.synchronize()

        return mock_migration_worker_pb2.KvCacheData(
            key=key,
            value=value,
            kv=kv,
            np_dtype=np_dtype,
            key_shape=np.array(key_tensor.shape, dtype=np.int64).tobytes(),
            value_shape=np.array(value_tensor.shape, dtype=np.int64).tobytes(),
            kv_shape=np.array(kv_tensor.shape, dtype=np.int64).tobytes(),
            block_idxs=request.block_idxs
        )

MAX_MESSAGE_LENGHT = 1024 * 1024 * 1024
options=[
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGHT),
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGHT),
]

def worker_main(listen_addr: str, rank: int, args: ServingArgs, instance_id: int,
                migration_config: MigrationConfig, naming_url: str, tranfer_type: TransferType):
    asyncio.run(launch_worker(listen_addr, rank, args, instance_id, migration_config, naming_url, tranfer_type))

async def launch_worker(listen_addr: str, rank: int, args: ServingArgs, instance_id: int,
                        migration_config: MigrationConfig, naming_url: str, tranfer_type: TransferType):
    worker = MockMigrationWorker(instance_id, listen_addr, migration_config, naming_url, tranfer_type, rank, args)
    server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=2), options=options)
    bladellm_pb2_grpc.add_WorkerServicer_to_server(worker, server)
    migration_worker_pb2_grpc.add_MigrationWorkerServicer_to_server(worker, server)
    mock_migration_worker_pb2_grpc.add_MockMigrationWorkerServicer_to_server(worker, server)
    server.add_insecure_port(listen_addr)

    async def shutdown(server):
        await server.stop(0)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown(server)))
    loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(shutdown(server)))

    await server.start()
    await server.wait_for_termination()

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU to run the test.")
@pytest.mark.parametrize(
    "backend, attention_type, transfer_type",
    [
        ('kvtransfer', 'ragged_flash', TransferType.CUDA_IPC_DIRECT),
        ('kvtransfer', 'ragged_flash', TransferType.RDMA_DIRECT),
        ('grpc', 'ragged_flash', TransferType.CUDA_IPC_DIRECT),
        ('grpc', 'paged', TransferType.CUDA_IPC_DIRECT),
    ],
)
def test_migrate_cache(backend, attention_type, transfer_type):
    worker_count = 2
    worker_socket_addrs = []
    migration_cache_blocks = 2
    total_migration_cache_blocks = migration_cache_blocks + 1
    migration_config = EngineManagerArgs(migration_backend=backend,
        migration_cache_blocks=migration_cache_blocks).create_migration_config()
    naming_url = 'shm:migrate_cache_test'
    shm = shared_memory.SharedMemory(create=True, size=1024*1024, name='migrate_cache_test')
    # TODO(KuilongCui): check the best value for ACCL_MAX_USER_MR_GB
    os.environ['ACCL_MAX_USER_MR_GB'] = '10'

    set_start_method("spawn", force=True)
    backends: List[Process] = []
    for i in range(worker_count):
        worker_socket_addrs.append(f"localhost:{11234+i}")
        worker_args = ServingArgs(
            enable_remote_worker=True,
            device=i,
            server_ip="127.0.0.1",
            rank=0,
            max_gpu_memory_utilization=0.3,
            load_model_options=LoadModelOptions(
                model='/mnt/workspace/llumnix/opt-125m', attn_cls=attention_type, disable_cuda_graph=True
            )
        )
        p = Process(target=worker_main, daemon=True,
                    args=(worker_socket_addrs[-1], 0, worker_args, i, migration_config, naming_url, transfer_type))
        p.start()
        backends.append(p)

    time.sleep(10)

    worker0_channel = grpc.insecure_channel(worker_socket_addrs[0], options=options)
    worker1_channel = grpc.insecure_channel(worker_socket_addrs[1], options=options)

    worker0_stub = migration_worker_pb2_grpc.MigrationWorkerStub(worker0_channel)
    worker1_stub = migration_worker_pb2_grpc.MigrationWorkerStub(worker1_channel)

    mock_worker0_stub = mock_migration_worker_pb2_grpc.MockMigrationWorkerStub(worker0_channel)
    mock_worker1_stub = mock_migration_worker_pb2_grpc.MockMigrationWorkerStub(worker1_channel)

    assert worker0_stub.warmup(empty_pb2.Empty()).ok
    assert worker1_stub.warmup(empty_pb2.Empty()).ok

    responce = mock_worker0_stub.get_kv_cache_meta(empty_pb2.Empty())
    key_shape = list(np.frombuffer(responce.key_shape, dtype=np.int64))
    value_shape = list(np.frombuffer(responce.value_shape, dtype=np.int64))
    kv_shape = list(np.frombuffer(responce.kv_shape, dtype=np.int64))
    # pylint: disable=eval-used
    kv_cache_dtype = eval(responce.kv_cache_dtype)
    migration_cache_dtype = eval(responce.migration_cache_dtype)

    if attention_type == 'paged':
        key_shape[1] = total_migration_cache_blocks
        value_shape[1] = total_migration_cache_blocks
        dummy_key_data = torch.randn(size=tuple(key_shape), dtype=kv_cache_dtype)
        dummy_value_data = torch.randn(size=tuple(value_shape), dtype=kv_cache_dtype)
        np_key = dummy_key_data.type(migration_cache_dtype).numpy()
        np_value = dummy_value_data.type(migration_cache_dtype).numpy()
        np_dtype = str(np_key.dtype)
        key_shape=np.array(dummy_key_data.shape, dtype=np.int64).tobytes()
        value_shape=np.array(dummy_value_data.shape, dtype=np.int64).tobytes()
        kv_shape, np_kv = bytes(), np.array([], dtype=np.float32)
    elif attention_type == 'ragged_flash':
        kv_shape[1] = total_migration_cache_blocks
        dummy_kv_data = torch.randn(size=tuple(kv_shape), dtype=kv_cache_dtype)
        np_kv = dummy_kv_data.type(migration_cache_dtype).numpy()
        kv_shape = np.array(dummy_kv_data.shape, dtype=np.int64).tobytes()
        np_dtype = str(np_kv.dtype)
        key_shape, value_shape, = bytes(), bytes()
        np_key, np_value = np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    responce = mock_worker0_stub.set_gpu_cache(mock_migration_worker_pb2.KvCacheData(
        key=np_key.tobytes(),
        value=np_value.tobytes(),
        kv=np_kv.tobytes(),
        np_dtype=np_dtype,
        key_shape=key_shape,
        value_shape=value_shape,
        kv_shape=kv_shape,
        block_idxs=list(range(total_migration_cache_blocks)),
    ))

    request = mock_migration_worker_pb2.GetGpuCacheRequest(block_idxs=list(range(total_migration_cache_blocks)))
    responce = mock_worker0_stub.get_gpu_cache(request)
    dtype = getattr(np, responce.np_dtype, None)
    if attention_type == 'paged':
        key_shape = np.frombuffer(responce.key_shape, dtype=np.int64)
        value_shape = np.frombuffer(responce.value_shape, dtype=np.int64)
        worker0_key_data = torch.from_numpy(np.frombuffer(responce.key, dtype=dtype).reshape(key_shape))
        worker0_value_data = torch.from_numpy(np.frombuffer(responce.value, dtype=dtype).reshape(value_shape))
    elif attention_type == 'ragged_flash':
        kv_shape = np.frombuffer(responce.kv_shape, dtype=np.int64)
        worker0_kv_data = torch.from_numpy(np.frombuffer(responce.kv, dtype=dtype).reshape(kv_shape))

    if attention_type == 'paged':
        assert torch.allclose(dummy_key_data, worker0_key_data)
        assert torch.allclose(dummy_value_data, worker0_value_data)
    elif attention_type == 'ragged_flash':
        assert torch.allclose(dummy_kv_data, worker0_kv_data)
    else:
        raise RuntimeError(f"Unknown attention type: {attention_type}")

    dst_blocks = list(range(total_migration_cache_blocks))
    random.shuffle(dst_blocks)

    src_worker_info = migration_worker_pb2.WorkerInfo(
        ip_address=worker_socket_addrs[0],
        instance_id=0,
        worker_id=0
    )
    worker1_stub.migrate_cache(migration_worker_pb2.MigrateRequests(
        src_handlers=[src_worker_info, migration_worker_pb2.WorkerInfo()],
        src_blocks=list(range(total_migration_cache_blocks)),
        dst_blocks=dst_blocks,
    ))

    request = mock_migration_worker_pb2.GetGpuCacheRequest(block_idxs=list(range(total_migration_cache_blocks)))
    responce = mock_worker1_stub.get_gpu_cache(request)
    dtype = getattr(np, responce.np_dtype, None)
    if attention_type == 'paged':
        key_shape = np.frombuffer(responce.key_shape, dtype=np.int64)
        value_shape = np.frombuffer(responce.value_shape, dtype=np.int64)
        worker1_key_data = torch.from_numpy(np.frombuffer(responce.key, dtype=dtype).reshape(key_shape))
        worker1_value_data = torch.from_numpy(np.frombuffer(responce.value, dtype=dtype).reshape(value_shape))
        num_layer = worker1_key_data.shape[0]
    elif attention_type == 'ragged_flash':
        kv_shape = np.frombuffer(responce.kv_shape, dtype=np.int64)
        worker1_kv_data = torch.from_numpy(np.frombuffer(responce.kv, dtype=dtype).reshape(kv_shape))
        num_layer = worker1_kv_data.shape[0]

    for layer_idx in range(num_layer):
        for src_block_idx, dst_block_idx in enumerate(dst_blocks):
            if attention_type == 'paged':
                assert torch.allclose(worker1_key_data[layer_idx][dst_block_idx], worker0_key_data[layer_idx][src_block_idx])
                assert torch.allclose(worker1_value_data[layer_idx][dst_block_idx], worker0_value_data[layer_idx][src_block_idx])
            elif attention_type == 'ragged_flash':
                assert torch.allclose(worker1_kv_data[layer_idx][dst_block_idx], worker0_kv_data[layer_idx][src_block_idx])
            else:
                raise RuntimeError(f"Unknown attention type: {attention_type}")

    worker0_channel.close()
    worker1_channel.close()

    for i in range(worker_count):
        backends[i].terminate()
        backends[i].join()

    shm.close()
    shm.unlink()
