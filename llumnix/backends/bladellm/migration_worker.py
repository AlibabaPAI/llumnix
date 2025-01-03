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

# pylint: disable=invalid-overridden-method

import asyncio
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch
from google.protobuf import empty_pb2
from loguru import logger

from blade_llm.service.args import ServingArgs
from blade_llm.service.workers.local_worker import LocalWorker

from llumnix.entrypoints.setup import get_ip_address
from llumnix.backends.bladellm.migration_backend import get_migration_backend
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.backends.bladellm.proto.migration_worker_pb2 import MigrateCacheResponse
from llumnix.internal_config import MigrationConfig

class MigrationWorker(migration_worker_pb2_grpc.MigrationWorkerServicer):
    def __init__(self, state_manager, instance_id: int, migration_config: MigrationConfig,
                 rank: int, args: ServingArgs) -> None:
        migration_worker_pb2_grpc.MigrationWorkerServicer.__init__(self)
        device = args.device if args.device else torch.cuda.device(rank)
        torch.cuda.set_device(device)
        self.instance_id = instance_id
        self.migration_config = migration_config
        self.rank = rank
        self.state_manager = state_manager
        self.migration_backend = get_migration_backend(instance_id, rank, rank, migration_config,
                                                       state_manager, args)

        self.migration_grpc_addr = get_ip_address() + ":" + \
            str(self.migration_config.migration_backend_server_port+self.rank)
        asyncio.create_task(self._launch_grpc_service())

    async def _launch_grpc_service(self):
        # pylint: disable=invalid-name
        MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
        server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=1), options=options)
        migration_worker_pb2_grpc.add_MigrationWorkerServicer_to_server(self, server)
        server.add_insecure_port(self.migration_grpc_addr)
        await server.start()
        # TODO(KuilongCui): clear server state
        await server.wait_for_termination()

    async def is_ready(self, request, context):
        return empty_pb2.Empty()

    # pylint: disable=unused-argument
    async def migrate_cache(self, request, context):
        try:
            self.migration_backend.migrate_cache(request, request.src_blocks, request.dst_blocks)
        # pylint: disable=broad-except
        except Exception:
            logger.exception("[migrate_cache] rank: {}, {} is dead.".format(self._rank, request))

        # pylint: disable=protected-access
        num_out_token = self.state_manager._request_groups[request.request_id].request_metas[0].num_out_token
        num_in_token = self.state_manager._request_groups[request.request_id].request_metas[0].num_in_token
        return MigrateCacheResponse(request_id=request.request_id, num_output_token=num_out_token, num_in_token=num_in_token)

    async def do_send(self, request, context):
        return self.migration_backend.do_send(request, context)

    async def rebuild_migration_backend(self, request, context) -> bool:
        self.migration_backend.destory_backend()
        return self.migration_backend.init_backend(None, None, None)

    async def warmup(self, request, context):
        resp = migration_worker_pb2.WarmupResponse(is_ok=True)
        try:
            self.migration_backend.warmup()
        # pylint: disable=broad-except
        except Exception as e:
            resp.is_ok = False
            resp.error_msg = f"warmup failed: {e}"
        return resp

    async def shutdown(self) -> None:
        torch.cuda.synchronize()
        del self.migration_backend
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

class MigrationLocalWorker(LocalWorker, MigrationWorker):
    def __init__(self, rank: int, serving_args: ServingArgs,
                 instance_id: int, migration_config: MigrationConfig,) -> None:
        LocalWorker.__init__(self, rank, serving_args)
        state_manager = self._engine._state_manager
        MigrationWorker.__init__(self, state_manager, instance_id, migration_config,
                                 rank, serving_args)

    async def info(self, req: empty_pb2.Empty) -> str:
        async with grpc.aio.insecure_channel(self.migration_grpc_addr) as channel:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)
            await stub.is_ready(empty_pb2.Empty())
        return await super().info(req)
