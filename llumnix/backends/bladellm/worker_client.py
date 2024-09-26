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

import asyncio
from typing import List

from blade_llm.service.workers.worker_client import PipelineWorkerClient, LocalWorkerClient
from blade_llm.service.proto import bladellm_pb2_grpc, bladellm_pb2
from google.protobuf.empty_pb2 import Empty

from llumnix.backends.bladellm.proto import migration_worker_pb2, migration_worker_pb2_grpc
from llumnix.logger import init_logger

logger = init_logger(__name__)

class MigrationWorkerStub:
    def __init__(self, channel) -> None:
        migration_worker_pb2_grpc.MigrationWorkerStub.__init__(self, channel)
        bladellm_pb2_grpc.WorkerStub.__init__(self, channel)

# not ready
class LlumnixPipelineWorkerClient(PipelineWorkerClient):
    async def migrate_cache(self, request: migration_worker_pb2.MigrateRequests)-> None:
        req_str = request.SerializeToString()
        tasks = [stub.migrate_cache(req_str) for stub in self.stubs]
        await asyncio.gather(*tasks)

    async def do_send(self, request: migration_worker_pb2.SendKvCacheRequest)-> List[migration_worker_pb2.SendKvCacheResponse]:
        req_str = request.SerializeToString()
        tasks = [stub.migrate_cache(req_str) for stub in self.stubs]
        resps = await asyncio.gather(*tasks)
        return resps
    
    async def warmup(self)-> List[migration_worker_pb2.WarmupResponse]:
        request = Empty()
        tasks = [stub.warmup(request) for stub in self.stubs]
        resps = await asyncio.gather(*tasks)
        return resps

class LlumnixLocalWorkerClient(LocalWorkerClient):
    async def migrate_cache(self, request: migration_worker_pb2.MigrateRequests)-> None:
        request = bladellm_pb2.WorkerMetaRequest(method="migrate_cache", drop=request).SerializeToString()
        [self.rpc_call(request, i) for i in range(len(self.reader))]
        tasks = [self.rpc_response(i) for i in range(len(self.reader))]
        self.futures.put_nowait((tasks, None))

    async def do_send(self, request: migration_worker_pb2.SendKvCacheRequest)-> List[migration_worker_pb2.SendKvCacheResponse]:
        req_str = request.SerializeToString()
        tasks = [stub.migrate_cache(req_str) for stub in self.stubs]
        resps = await asyncio.gather(*tasks)
        return resps

    async def warmup(self)-> List[migration_worker_pb2.WarmupResponse]:
        request = Empty()
        tasks = [stub.warmup(request) for stub in self.stubs]
        resps = await asyncio.gather(*tasks)
        return resps

# def make_grpc_stubs(channels):
#     stubs = []
#     for c in channels:
#         stub = LlumnixWorkerStub(c)
#         stub.step = c.unary_unary(
#             '/Worker/step',
#             request_serializer=None,
#             response_deserializer=bladellm_pb2.WorkerStepResponse.FromString,
#         )
#         stub.drop = c.unary_unary(
#             '/Worker/drop',
#             request_serializer=None,
#             response_deserializer=Empty.FromString,
#         )
#         stubs.append(stub)
#     return stubs

# TODO[xinyi]: revise in bladellm repo
# TODO[xinyi]: llumnix now only support designated server_ip
# def worker_client_main(pp_enabled: bool, args: ServingArgs, client_args):
#     if args.enable_remote_worker or args.server_ip:
#         import sys
#         if 'llumnix' in sys.modules:
#             return LlumnixPipelineWorkerClient(*client_args)
#         else:
#             return PipelineWorkerClient(*client_args)
#     else:
#         return LocalWorkerClient(*client_args) if not pp_enabled else PipelineWorkerClient(*client_args)