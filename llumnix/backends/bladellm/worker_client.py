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

from typing import List
import grpc

from blade_llm.service.worker_client import AioWorkerClient
from blade_llm.service.proto import bladellm_pb2_grpc
from blade_llm.service.args import ServingArgs

# pylint: disable=unused-import
from llumnix.logger import init_logger

from llumnix.backends.bladellm.proto import (
    llumnix_bladellm_pb2,
    llumnix_bladellm_pb2_grpc,
)

logger = init_logger(__name__)


class LlumnixWorkerStub:
    def __init__(self, channel) -> None:
        self.llumnix_worker_stub = llumnix_bladellm_pb2_grpc.LlumnixWorkerStub(channel)
        self.blade_worker_stub = bladellm_pb2_grpc.WorkerStub(channel)

    def info(self, request):
        return self.blade_worker_stub.info(request)

    def step(self, request):
        return self.blade_worker_stub.step(request)

    def reset(self, request):
        return self.blade_worker_stub.reset(request)

    def drop(self, request):
        return self.blade_worker_stub.drop(request)

    def estimate(self, request):
        return self.blade_worker_stub.estimate(request)

    def migrate_gpu_cache_grpc(self, request):
        return self.llumnix_worker_stub.migrate_gpu_cache_grpc(request)

    def allocate_migration_cache(self, request):
        return self.llumnix_worker_stub.allocate_migration_cache(request)

    def send_cpu_cache(self, request):
        return self.llumnix_worker_stub.send_cpu_cache(request)


class LlumnixAioWorkerClient(AioWorkerClient):
    def __init__(self, args: ServingArgs, addrs=None, inst_id=None):
        super().__init__(args, addrs, inst_id)

        self.stubs = [LlumnixWorkerStub(c) for c in self.channels]

        if addrs:
            self.sync_channels = [
                grpc.insecure_channel(addrs[i])
                for i in range(self.tp_size * self.pp_size)
            ]
            logger.info("created {} grpc channels to backends: {}".format(len(addrs),addrs))
        else:
            # cluster mode, attention the socket address path
            if self._inst_id is not None:
                self.sync_channels = [
                    grpc.insecure_channel(
                        f"unix://{args.worker_socket_path}.{self._inst_id}.{i}"
                    )
                    for i in range(self.tp_size * self.pp_size)
                ]
            else:
                self.sync_channels = [
                    grpc.insecure_channel(f"unix://{args.worker_socket_path}.{i}")
                    for i in range(self.tp_size * self.pp_size)
                ]
        self.sync_stubs = [LlumnixWorkerStub(c) for c in self.sync_channels]
        logger.info("channels {}".format(self.sync_channels))

    def allocate_migration_cache(
        self,
        request: llumnix_bladellm_pb2.AllocRequest,
    ) -> None:
        for stub in self.sync_stubs:
            stub.allocate_migration_cache(request)

    def migrate_gpu_cache_grpc(
        self,
        request: llumnix_bladellm_pb2.MigrateRequest,
    ) -> None:
        for stub in self.sync_stubs:
            stub.migrate_gpu_cache_grpc(request)

    def send_cpu_cache(
        self,
        request: llumnix_bladellm_pb2.SendRequest,
    ) -> List[llumnix_bladellm_pb2.SendNumpyResponse]:
        tasks = [stub.send_cpu_cache(request) for stub in self.sync_stubs]
        return [task.result() for task in tasks]
