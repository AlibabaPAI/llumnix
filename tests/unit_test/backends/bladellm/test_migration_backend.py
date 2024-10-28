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
import signal
from typing import List
import random
import asyncio
import torch
import pytest
import grpc
from multiprocessing import Process, set_start_method
from concurrent.futures import ThreadPoolExecutor
from google.protobuf import empty_pb2
import numpy as np

from blade_llm.service.args import ServingArgs
from blade_llm.utils.load_model_options import LoadModelOptions
from blade_llm.service.proto import bladellm_pb2_grpc

from llumnix.backends.bladellm.worker import MigrationWorker
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.internal_config import MigrationConfig
from llumnix.utils import random_uuid
from llumnix.arg_utils import EngineManagerArgs

from tests.unit_test.backends.bladellm.proto import mock_migration_worker_pb2_grpc, mock_migration_worker_pb2

class MockMigrationWorker(mock_migration_worker_pb2_grpc.MockMigrationWorkerServicer, MigrationWorker):
    def __init__(self, *args, **kwargs) -> None:
        MigrationWorker.__init__(self, *args, **kwargs)

    def get_kv_cache_meta(self, request, context):
        num_layers = len(self.migration_backend.kv_cache_arena.gpu_k_cache)
        key_shape = [num_layers] + list(self.migration_backend.kv_cache_arena.gpu_k_cache[0].shape)
        value_shape = [num_layers] + list(self.migration_backend.kv_cache_arena.gpu_v_cache[0].shape)

        return mock_migration_worker_pb2.KvCacheMeta(
            key_shape=np.array(key_shape, dtype=np.int64).tobytes(),
            value_shape=np.array(value_shape, dtype=np.int64).tobytes(),
            kv_cache_dtype=str(self.migration_backend.state_manager.dtype),
            migration_cache_dtype=str(self.migration_backend.migration_cache_dtype)
        )

    def set_gpu_cache(self, request, context):
        dtype = getattr(np, request.np_dtype, None)
        key_shape = np.frombuffer(request.key_shape, dtype=np.int64)
        value_shape = np.frombuffer(request.value_shape, dtype=np.int64)
        block_idxs = list(request.block_idxs)
        key = torch.from_numpy(np.frombuffer(request.key, dtype=dtype).reshape(key_shape))
        value = torch.from_numpy(np.frombuffer(request.value, dtype=dtype).reshape(value_shape))

        for layer_idx in range(key.shape[0]):
            for src_block_idx, target_block_idx in enumerate(block_idxs):
                self.migration_backend.kv_cache_arena.gpu_k_cache[layer_idx][target_block_idx].copy_(key[layer_idx][src_block_idx])
                self.migration_backend.kv_cache_arena.gpu_v_cache[layer_idx][target_block_idx].copy_(value[layer_idx][src_block_idx])
        torch.cuda.synchronize()

        return empty_pb2.Empty()

    def get_gpu_cache(self, request, context):
        key = []
        value = []
        for idx in range(len(self.migration_backend.kv_cache_arena.gpu_k_cache)):
            key.append(self.migration_backend.kv_cache_arena.gpu_k_cache[idx][request.block_idxs])
            value.append(self.migration_backend.kv_cache_arena.gpu_v_cache[idx][request.block_idxs])

        key_tensor = torch.stack(key).type(self.migration_backend.migration_cache_dtype).cpu()
        value_tensor = torch.stack(value).type(self.migration_backend.migration_cache_dtype).cpu()
        torch.cuda.synchronize()

        return mock_migration_worker_pb2.KvCacheData(
            key=key_tensor.numpy().tobytes(),
            value=value_tensor.numpy().tobytes(),
            np_dtype=str(key_tensor.numpy().dtype),
            key_shape=np.array(key_tensor.shape, dtype=np.int64).tobytes(),
            value_shape=np.array(value_tensor.shape, dtype=np.int64).tobytes(),
            block_idxs=request.block_idxs
        )

def worker_main(listen_addr: str, rank: int, args: ServingArgs, instance_id: str, migration_config: MigrationConfig):
    asyncio.run(launch_worker(listen_addr, rank, args, instance_id, migration_config))

async def launch_worker(listen_addr: str, rank: int, args: ServingArgs, instance_id: str, migration_config: MigrationConfig):
    worker = MockMigrationWorker(instance_id, listen_addr, migration_config, rank, args)
    server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=2))
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
@pytest.mark.parametrize("backend", ['grpc'])
def test_migrate_cache(backend):
    worker_count = 2
    worker_socket_addrs = []
    migration_cache_blocks=8
    total_migration_cache_blocks = 2 * migration_cache_blocks + 1
    migration_config = EngineManagerArgs(migration_backend=backend,
        migration_cache_blocks=migration_cache_blocks).create_migration_config()

    set_start_method("spawn", force=True)
    backends: List[Process] = []
    for i in range(worker_count):
        instance_id = random_uuid()
        worker_socket_addrs.append(f"localhost:{1234+i}")
        worker_args = ServingArgs(
            enable_remote_worker=True,
            device=i,
            server_ip="127.0.0.1",
            rank=0,
            max_gpu_memory_utilization=0.3,
            block_size=3,
            load_model_options=LoadModelOptions(
                model='facebook/opt-125m', attn_cls="paged", disable_cuda_graph=True
            )
        )
        p = Process(target=worker_main, daemon=True,
                    args=(worker_socket_addrs[-1], 0, worker_args, instance_id, migration_config))
        p.start()
        backends.append(p)

    time.sleep(10)

    worker0_channel = grpc.insecure_channel(worker_socket_addrs[0])
    worker1_channel = grpc.insecure_channel(worker_socket_addrs[1])

    worker0_stub = migration_worker_pb2_grpc.MigrationWorkerStub(worker0_channel)
    worker1_stub = migration_worker_pb2_grpc.MigrationWorkerStub(worker1_channel)

    mock_worker0_stub = mock_migration_worker_pb2_grpc.MockMigrationWorkerStub(worker0_channel)
    mock_worker1_stub = mock_migration_worker_pb2_grpc.MockMigrationWorkerStub(worker1_channel)

    assert worker0_stub.warmup(empty_pb2.Empty()).ok
    assert worker1_stub.warmup(empty_pb2.Empty()).ok

    responce = mock_worker0_stub.get_kv_cache_meta(empty_pb2.Empty())
    key_shape = list(np.frombuffer(responce.key_shape, dtype=np.int64))
    value_shape = list(np.frombuffer(responce.value_shape, dtype=np.int64))
    kv_cache_dtype = eval(responce.kv_cache_dtype)
    migration_cache_dtype = eval(responce.migration_cache_dtype)

    key_shape[1] = total_migration_cache_blocks
    value_shape[1] = total_migration_cache_blocks
    dummy_key_data = torch.randn(size=tuple(key_shape), dtype=kv_cache_dtype)
    dummy_value_data = torch.randn(size=tuple(value_shape), dtype=kv_cache_dtype)

    np_key = dummy_key_data.type(migration_cache_dtype).numpy()
    np_value = dummy_value_data.type(migration_cache_dtype).numpy()
    responce = mock_worker0_stub.set_gpu_cache(mock_migration_worker_pb2.KvCacheData(
        key=np_key.tobytes(),
        value=np_value.tobytes(),
        np_dtype=str(np_key.dtype),
        key_shape=np.array(dummy_key_data.shape, dtype=np.int64).tobytes(),
        value_shape=np.array(dummy_value_data.shape, dtype=np.int64).tobytes(),
        block_idxs=list(range(total_migration_cache_blocks)),
    ))

    request = mock_migration_worker_pb2.GetGpuCacheRequest(block_idxs=list(range(total_migration_cache_blocks)))
    responce = mock_worker0_stub.get_gpu_cache(request)
    dtype = getattr(np, responce.np_dtype, None)
    key_shape = np.frombuffer(responce.key_shape, dtype=np.int64)
    value_shape = np.frombuffer(responce.value_shape, dtype=np.int64)
    worker0_key_data = torch.from_numpy(np.frombuffer(responce.key, dtype=dtype).reshape(key_shape))
    worker0_value_data = torch.from_numpy(np.frombuffer(responce.value, dtype=dtype).reshape(value_shape))

    assert torch.allclose(dummy_key_data, worker0_key_data)
    assert torch.allclose(dummy_value_data, worker0_value_data)

    dst_blocks = list(range(total_migration_cache_blocks))
    random.shuffle(dst_blocks)

    worker1_stub.migrate_cache(migration_worker_pb2.MigrateRequest(
        src_handlers=[worker_socket_addrs[0], 'None'],
        src_blocks=list(range(total_migration_cache_blocks)),
        dst_blocks=dst_blocks,
    ))

    request = mock_migration_worker_pb2.GetGpuCacheRequest(block_idxs=list(range(total_migration_cache_blocks)))
    responce = mock_worker1_stub.get_gpu_cache(request)
    dtype = getattr(np, responce.np_dtype, None)
    key_shape = np.frombuffer(responce.key_shape, dtype=np.int64)
    value_shape = np.frombuffer(responce.value_shape, dtype=np.int64)
    worker1_key_data = torch.from_numpy(np.frombuffer(responce.key, dtype=dtype).reshape(key_shape))
    worker1_value_data = torch.from_numpy(np.frombuffer(responce.value, dtype=dtype).reshape(value_shape))

    for layer_idx in range(worker1_key_data.shape[0]):
        for src_block_idx, dst_block_idx in enumerate(dst_blocks):
            assert torch.allclose(worker1_key_data[layer_idx][dst_block_idx], worker0_key_data[layer_idx][src_block_idx])
            assert torch.allclose(worker1_value_data[layer_idx][dst_block_idx], worker0_value_data[layer_idx][src_block_idx])

    worker0_channel.close()
    worker1_channel.close()
    
    for i in range(worker_count):
        backends[i].terminate()
        backends[i].join()
