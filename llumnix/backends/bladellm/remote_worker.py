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
from blade_llm.service.worker import RemoteManager, _RemoteWorkerProcesses, _WorkerProcesses
from blade_llm.service.workers.remote_worker import RemoteWorker
from blade_llm.utils.network import get_free_port
from blade_llm.service.proto import bladellm_pb2_grpc
from blade_llm.service.worker import Worker

from llumnix.backends.bladellm.proto import (
    llumnix_bladellm_pb2,
    llumnix_bladellm_pb2_grpc,
)
from llumnix.logger import init_logger

logger = init_logger(__name__)

# TODO[xinyi]: now llumnix only support remoteWorker
class LlumnixRemoteWorker(llumnix_bladellm_pb2_grpc.LlumnixWorkerServicer, RemoteWorker):
    def __init__(self, *args, **kwargs) -> None:
        # replace sampler
        # pylint: disable=import-outside-toplevel
        super().__init__(*args, **kwargs)

        # too many logs in BladeLLM, redefine the log level
        # TODO[xinyi]: check
        # loguru.logger.remove()
        # loguru.logger.add(sys.stderr, level="INFO")

    # pylint: disable=unused-argument
    def migrate_cache(
        self, request: llumnix_bladellm_pb2.MigrateCacheRequest, context: grpc.ServicerContext
    ):
        # TODO[kuilong, xinyi]: adapt to kvTransfer 
        return empty_pb2.Empty()

# TODO[xinyi]: revise in bladellm repo
# TODO[xinyi]: any idea to simplify the revision?
# async def start_remote_worker_server(rank: int, args: ServingArgs):
#     if args.server_ip:
#         worker_port = int(get_free_port())
#         await RemoteManager.start_watch_dog(args, worker_port)
#         await RemoteManager.wait_until_all_workers_ready()
#     import sys
#     if 'llumnix' in sys.modules:
#         worker = LlumnixRemoteWorker(rank, args)
#     else:
#         worker = RemoteWorker(rank, args)
#     server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=1))
#     bladellm_pb2_grpc.add_WorkerServicer_to_server(worker, server)
#     import sys
#     if 'llumnix' in sys.modules:
#         llumnix_bladellm_pb2_grpc.add_LlumnixRemoteWorkerServicer_to_server(worker, server)
#     listen_addr = f"0.0.0.0:{worker_port}" if args.server_ip else f"unix://{args.worker_socket_path}.{rank}"
#     server.add_insecure_port(listen_addr)
#     await server.start()
#     if args.server_ip:
#         logger.info(f"Backend {rank} ready on port {worker_port}")
#         await RemoteManager.wait_for_termination(server)
#     else:
#         await server.wait_for_termination()
#     # explicit cleanup
#     del server
#     del worker
#     gc.collect()