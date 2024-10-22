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
import grpc

from blade_llm.service.worker_client import PipelineWorkerClient
from blade_llm.service.proto import bladellm_pb2_grpc
from blade_llm.service.proto import bladellm_pb2
from blade_llm.service.args import ServingArgs
from google.protobuf.empty_pb2 import Empty

# pylint: disable=unused-import
from llumnix.logger import init_logger

from llumnix.backends.bladellm.proto import (
    llumnix_bladellm_pb2,
    llumnix_bladellm_pb2_grpc,
)

logger = init_logger(__name__)


class LlumnixWorkerStub:
    def __init__(self, channel) -> None:
        llumnix_bladellm_pb2_grpc.LlumnixWorkerStub.__init__(self, channel)
        bladellm_pb2_grpc.WorkerStub.__init__(self, channel)

class LlumnixPipelineWorkerClient(PipelineWorkerClient):
    def __init__(self, args: ServingArgs, addrs=None, inst_id=None):
        super().__init__(args, addrs, inst_id)

        self.stubs = make_grpc_stubs(self.channels)
        self.stub_groups = [
            tuple(self.stubs[i] for i in range(p * self.tp_size, (p + 1) * self.tp_size)) for p in range(self.pp_size)
        ]
        self.stub_group_requests = {group: asyncio.Queue() for group in self.stub_groups}

    # TODO[xinyi]: function demo, need to adapt to the proto.
    def migrate_cache(
        self, request: llumnix_bladellm_pb2.MigrateCacheRequest
    )-> None:
        for stub in self.stubs:
            stub.migrate_cache(request)

def make_grpc_stubs(channels):
    stubs = []
    for c in channels:
        stub = LlumnixWorkerStub(c)
        stub.step = c.unary_unary(
            '/Worker/step',
            request_serializer=None,
            response_deserializer=bladellm_pb2.WorkerStepResponse.FromString,
        )
        stub.drop = c.unary_unary(
            '/Worker/drop',
            request_serializer=None,
            response_deserializer=Empty.FromString,
        )
        stubs.append(stub)
    return stubs

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