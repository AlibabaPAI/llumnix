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
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor
from google.protobuf import empty_pb2
import torch
import grpc

from blade_llm.generation.kvcache.kv_transfer import TransferType
from blade_llm.service.args import ServingArgs
from blade_llm.utils.network import get_free_port

from llumnix.backends.bladellm.migration_backend import get_migration_backend
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.internal_config import MigrationConfig
from llumnix.logger import init_logger

logger = init_logger(__name__)

class MigrationWorker(migration_worker_pb2_grpc.MigrationWorkerServicer):
    def __init__(self, state_manager, instance_id: int, worker_addr: str, migration_config: MigrationConfig,
                 naming_url: str,  tranfer_type: TransferType, rank: int, args: ServingArgs) -> None:
        migration_worker_pb2_grpc.MigrationWorkerServicer.__init__(self)
        torch.cuda.set_device(args.device)
        self.instance_id = instance_id
        self.migration_backend = get_migration_backend(instance_id, rank, rank, worker_addr, migration_config,
                                                       state_manager, naming_url, args, tranfer_type)

    # pylint: disable=unused-argument
    def migrate_cache(self, request: migration_worker_pb2.MigrateRequests, context) -> None:
        src_worker_handle = request.src_handlers[self._rank]
        try:
            self.migration_backend.migrate_cache(src_worker_handle, request.src_blocks, request.dst_blocks)
        # pylint: disable=broad-except
        except Exception as e:
            logger.info("[migrate_cache] rank: {}, {} is dead, err : {}.".format(self._rank, src_worker_handle, e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))

        return empty_pb2.Empty()

    def do_send(self, request: migration_worker_pb2.SendKvCacheRequest, context):
        return self.migration_backend.do_send(request, context)

    def rebuild_migration_backend(self, instance_rank: Dict[str, int], group_name: str) -> bool:
        self.migration_backend.destory_backend()
        return self.migration_backend.init_backend(None, None, None)

    def warmup(self, request, context):
        resp = migration_worker_pb2.WarmupResponse(is_ok=True)
        try:
            self.migration_backend.warmup()
        except Exception as e:
            resp.is_ok = False
            resp.error_msg = f"warmup failed: {e}"
        return resp

    def shutdown(self) -> None:
        torch.cuda.synchronize()
        del self.migration_backend
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

# TODO(xinyi): revise in bladellm repo
# def worker_main(rank: int, serving_args: ServingArgs, *args):
#     asyncio.run(worker_server(rank, serving_args, *args))

# TODO(xinyi): revise `start_local_worker_server` in bladellm repo 
# async def worker_server(rank: int, args: ServingArgs, instance_id: int, migration_config: MigrationConfig,
#                         naming_url: str, tranfer_type: TransferType):
#     if args.server_ip:
#         worker_port = int(get_free_port())
#         await RemoteManager.start_watch_dog(args, worker_port)
#         await RemoteManager.wait_until_all_workers_ready()
#     listen_addr = f"0.0.0.0:{worker_port}" if args.server_ip else f"unix://{args.worker_socket_path}.{rank}"
#     worker = MigrationWorker(instance_id, listen_addr, migration_config, naming_url, tranfer_type, rank, args)
#     server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=1))
#     bladellm_pb2_grpc.add_WorkerServicer_to_server(worker, server)
#     import sys
#     if 'llumnix' in sys.modules:
#         migration_worker_pb2_grpc.add_MigrationWorkerServicer_to_server(worker, server)
#     server.add_insecure_port(listen_addr)
#     await server.start()

#     if args.server_ip:
#         await RemoteManager.wait_for_termination(server)
#     else:
#         await server.wait_for_termination()

#     del server
#     del worker
#     gc.collect()
