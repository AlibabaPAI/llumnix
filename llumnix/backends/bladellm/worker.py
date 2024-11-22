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

from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from google.protobuf import empty_pb2
from multiprocessing import Process, set_start_method
from typing import List
import asyncio
import sys

from blade_llm.generation.kvcache.kv_transfer import TransferType
from blade_llm.service.workers.remote_worker import RemoteWorker, RemoteManager
from blade_llm.service.args import ServingArgs
from blade_llm.service.worker import _RemoteWorkerProcesses, _WorkerProcesses, setup_dist
from blade_llm.utils.network import get_free_port
# from blade_llm.service.worker import worker_main
from blade_llm.service.workers.local_worker import LocalWorker
from blade_llm.service.workers.remote_worker import RemoteWorker
# from blade_llm.service.workers.local_worker import start_local_worker_server
from blade_llm.service.workers.remote_worker import (
    RemoteManager,
    start_remote_worker_server,
)
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.internal_config import MigrationConfig
from llumnix.backends.bladellm.migration_worker import MigrationWorker
from llumnix.logger import init_logger

logger = init_logger(__name__)

class _WorkerProcessesLlumnix(_WorkerProcesses):
    def __init__(self, args: ServingArgs, instance_id: int, migration_config: MigrationConfig):
        super().__init__(args)
        self.instance_id = instance_id
        self.migration_config = migration_config

    def _spawn_workers(self):
        """Spawn backend servers in background processes for inference"""
        set_start_method("spawn", force=True)
        backends: List[Process] = []
        for i in range(self._args.tensor_parallel_size * self._args.pipeline_parallel_size):
            p = Process(target=worker_main, args=(i, self._args),
                kwargs=dict(
                    instance_id=self.instance_id,
                    migration_config=self.migration_config,
                )
                )
            p.start()
            backends.append(p)
        return backends

# TODO(xinyi): may be only need for test
def worker_main(rank: int, serving_args: ServingArgs, *args, **kwargs):
    asyncio.run(worker_server(rank, serving_args, *args, **kwargs))

# TODO(xinyi): may be only need for test
async def worker_server(rank: int, serving_args: ServingArgs, *args, **kwargs):
    # logger.remove()
    # logger.add(sys.stderr, level=serving_args.log_level)
    # logger.info("================= Worker {} =================", rank)
    for k, v in serving_args.__dict__.items():
        logger.info(f"{k:>20}: {v}")
    if serving_args.tensor_parallel_size * serving_args.pipeline_parallel_size > 1:
        setup_dist(rank, serving_args)
    # if serving_args.server_ip or serving_args.pipeline_parallel_size > 1:
    #     await start_remote_worker_server(rank, serving_args)
    if True: #'llumnix' in sys.modules
        # TODO(xinyi)
        await start_local_worker_server(*args, rank=rank, serving_args=serving_args, **kwargs)
    else:
        await start_local_worker_server(rank, serving_args)

async def start_local_worker_server(rank: int, serving_args: ServingArgs, *args, **kwargs):
    import os
    import gc
    socket_path = f"{serving_args.worker_socket_path}.{rank}"
    if os.path.exists(socket_path):
        os.remove(socket_path)
    running_loop = asyncio.get_running_loop()
    logger.info(f"Worker serving on {socket_path}")
    server = await running_loop.create_unix_server(protocol_factory=make_protocol_factory(rank=rank, serving_args=serving_args,
                                                                                          *args, **kwargs), path=socket_path)
    # async with server:
    await server.serve_forever()
    del server
    gc.collect()

MAX_MESSAGE_LENGHT = 1024 * 1024 * 1024
options=[
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGHT),
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGHT),
]

async def create_migrate_service(worker: MigrationWorker):
    import grpc
    server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=2), options=options)
    migration_worker_pb2_grpc.add_MigrationWorkerServicer_to_server(worker, server)
    server.add_insecure_port(worker.migration_config.migration_backend_server_address)
    await server.start()
    await server.wait_for_termination()

# TODO(xinyi): how to pass parameters for MigrationLocalWorker
def make_protocol_factory(*args, **kwargs):
    def protocol_factory():
        try:
            protocol = MigrationLocalWorker(*args, **kwargs)
            # TODO(xinyi): how to launch migrate_service
            asyncio.create_task(create_migrate_service(protocol))
            return protocol
        except Exception as e:
            logger.error(f"Exception in LocalWorker: {e}")
            exit(-1)

    return protocol_factory

# not ready
class _RemoteWorkerProcessesLlumnix(_RemoteWorkerProcesses):
    def __init__(self, args: ServingArgs, instance_id: int, migration_config: MigrationConfig):
        super().__init__(args)
        self.instance_id = instance_id
        self.migration_config = migration_config

def launch_worker(args: ServingArgs, instance_id: int, migration_config: MigrationConfig):
    if args.enable_remote_worker:
        # TODO(xinyi): not support RemoteWorkerProcesses
        return _RemoteWorkerProcessesLlumnix(args, instance_id, migration_config)
    else:
        return _WorkerProcessesLlumnix(args, instance_id, migration_config)

class MigrationLocalWorker(LocalWorker, MigrationWorker):
    def __init__(self, instance_id: int, migration_config: MigrationConfig,
                rank: int, serving_args: ServingArgs) -> None:
        LocalWorker.__init__(self, rank, serving_args)

        state_manager = self._engine._state_manager
        MigrationWorker.__init__(self, state_manager, instance_id, migration_config,
                                 rank, serving_args)
        
class MigrationRemoteWorker(RemoteWorker, MigrationWorker):
    def __init__(self, instance_id: int, migration_config: MigrationConfig,
                rank: int, serving_args: ServingArgs) -> None:
        RemoteWorker.__init__(self, rank, serving_args)

        state_manager = self._engine._state_manager
        MigrationWorker.__init__(self, state_manager, instance_id, migration_config,
                                 rank, serving_args)
