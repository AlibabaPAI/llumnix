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
from blade_llm.service.proto import bladellm_pb2 as pb
from blade_llm.generation.request_state import RequestGroup
from blade_llm.service.proto.bladellm_pb2 import PagedState, RequestMeta, WorkerRequest

from llumnix.backends.bladellm.worker import MigrationRemoteWorker, MigrationLocalWorker
from llumnix.backends.bladellm.migration_backend import NUMPY_SUPPORTED_DTYPES
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.internal_config import MigrationConfig
from llumnix.arg_utils import EngineManagerArgs

from tests.unit_test.backends.bladellm.proto import mock_migration_request_group_pb2_grpc, mock_migration_request_group_pb2

def _make_data(n: int):
    def _make_request(i):
        return pb.WorkerRequest(
            id=i,
            prompt='hello',
            sampling_params=pb.SamplingParams(temperature=0.8, top_p=0.1, top_k=3),
            stopping_criterial=pb.StoppingCriteria(max_new_tokens=6),
            logits_processors_params=pb.LogitsProcessorParams(repetition_penalty=1.1),
        )

    request_groups = {}
    for i in range(n):
        request_groups[i] = RequestGroup(_make_request(i))#, device=torch.device("cuda"))
        request_groups[i].request_states[0].input_ids = torch.empty(
            1, 10, dtype=torch.int32, device=torch.device("cuda:0")
        )
        request_groups[i].request_states[0].in_flight_tokens = 1
    return request_groups[0]

class MockMigrationRequestGroup(mock_migration_request_group_pb2_grpc.MockMigrationRequestGroupServicer):
    def __init__(self, migration_backend):
        mock_migration_request_group_pb2_grpc.MockMigrationRequestGroupServicer.__init__(self)
        self.migration_backend = migration_backend

    def set_request_group(self, request, context):
        state_manager = self.migration_backend.state_manager
        state_manager._request_groups[11] = _make_data(1)
        print("why",state_manager._request_groups[11])
        print("why",state_manager._request_groups[11].request_states)
        print("why",state_manager._request_groups[11].request_metas)
        return empty_pb2.Empty()

class MockMigrationRemoteWorker(MockMigrationRequestGroup, MigrationRemoteWorker):
    def __init__(self, *args, **kwargs) -> None:
        MigrationRemoteWorker.__init__(self, *args, **kwargs)
        MockMigrationRequestGroup.__init__(self, self.migration_backend)

class MockMigrationLocalWorker(MockMigrationRequestGroup, MigrationLocalWorker):
    def __init__(self, *args, **kwargs) -> None:
        MigrationLocalWorker.__init__(self, *args, **kwargs)
        MockMigrationRequestGroup.__init__(self, self.migration_backend)

MAX_MESSAGE_LENGHT = 1024 * 1024 * 1024
options=[
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGHT),
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGHT),
]

def worker_main(rank: int, args: ServingArgs, instance_id: int, migration_config, worker_type):
    asyncio.run(launch_worker(rank, args, instance_id, migration_config, worker_type))

async def launch_worker(rank: int, args: ServingArgs, instance_id: int,
                        migration_config: MigrationConfig, worker_type):
    worker = worker_type(instance_id, migration_config, rank, args)
    server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=2), options=options)
    bladellm_pb2_grpc.add_WorkerServicer_to_server(worker, server)
    migration_worker_pb2_grpc.add_MigrationWorkerServicer_to_server(worker, server)
    mock_migration_request_group_pb2_grpc.add_MockMigrationRequestGroupServicer_to_server(worker, server)
    listen_addr = migration_config.migration_backend_server_address
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
        # ('kvtransfer', 'ragged_flash', "cuda_ipc"),
        # ('kvtransfer', 'ragged_flash', "rdma"),
        ('grpc', 'ragged_flash', ""),
        ('grpc', 'paged', ""),
    ],
)
@pytest.mark.parametrize('worker_type', [MockMigrationRemoteWorker])
def test_migrate_request_group(backend, attention_type, transfer_type, worker_type):
    worker_count = 2
    worker_socket_addrs = []
    migration_cache_blocks = 2
    migration_config = EngineManagerArgs(migration_backend=backend,
        migration_cache_blocks=migration_cache_blocks).create_migration_config()
    
    set_start_method("spawn", force=True)
    backends: List[Process] = []
    for i in range(worker_count):
        worker_socket_addrs.append(f"localhost:{11234+i}")
        worker_args = ServingArgs(
            enable_remote_worker=True,
            device=i,
            server_ip="127.0.0.1",
            rank=0,
            max_gpu_memory_utilization=0.5,
            load_model_options=LoadModelOptions(
                model='/mnt/dataset/opt-125m', attn_cls=attention_type, disable_cuda_graph=True
            )
        )
        migration_config.migration_backend_server_address = worker_socket_addrs[-1]
        p = Process(target=worker_main, daemon=True,
                    args=(0, worker_args, i, migration_config, worker_type))
        p.start()
        backends.append(p)

    time.sleep(10)

    print(worker_socket_addrs)
    print(backends)

    worker0_channel = grpc.insecure_channel(worker_socket_addrs[0], options=options)
    worker1_channel = grpc.insecure_channel(worker_socket_addrs[1], options=options)


    worker0_stub = migration_worker_pb2_grpc.MigrationWorkerStub(worker0_channel)
    worker1_stub = migration_worker_pb2_grpc.MigrationWorkerStub(worker1_channel)


    mock_worker0_stub = mock_migration_request_group_pb2_grpc.MockMigrationRequestGroupStub(worker0_channel)
    mock_worker1_stub = mock_migration_request_group_pb2_grpc.MockMigrationRequestGroupStub(worker1_channel)


    warmup_resp = worker0_stub.warmup(empty_pb2.Empty())
    assert warmup_resp.is_ok
    warmup_resp = worker1_stub.warmup(empty_pb2.Empty())
    assert warmup_resp.is_ok

    responce = mock_worker0_stub.set_request_group(empty_pb2.Empty())

    src_worker_info = migration_worker_pb2.WorkerInfo(
        ip_address=worker_socket_addrs[0],
        instance_id=0,
        worker_id=0
    )

    warmup_resp = worker1_stub.migrate_request_group(migration_worker_pb2.MigrateResGroupRequests(
        id = 11,
        src_handlers=[src_worker_info],
    ))
    print(warmup_resp)
    assert warmup_resp.is_ok

    # request = mock_migration_request_group_pb2.(block_idxs=list(range(total_migration_cache_blocks)))


    worker0_channel.close()
    worker1_channel.close()

    for i in range(worker_count):
        backends[i].terminate()
        backends[i].join()

    # shm.close()
    # shm.unlink()
