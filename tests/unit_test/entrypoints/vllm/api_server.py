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

import argparse
import uvicorn
import ray

from vllm.outputs import CompletionOutput, RequestOutput

import llumnix.entrypoints.vllm.api_server
import llumnix.manager
from llumnix.server_info import ServerInfo
from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.utils import random_uuid, get_manager_name
from llumnix.queue.utils import init_request_output_queue_server, init_request_output_queue_client, QueueType
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.entrypoints.vllm.client import LlumnixClientVLLM

import tests.unit_test.entrypoints.vllm.api


class MockManager:
    def __init__(self, request_output_queue_type: QueueType):
        self._num_generates = 0
        self._num_aborts = 0
        self.request_output_queue = init_request_output_queue_client(request_output_queue_type)

    async def generate(self, request_id, server_info, *args, **kwargs):
        self._num_generates += 1
        completion_output = CompletionOutput(0, "", [], 0.0, None)
        request_output = RequestOutput(request_id, "", [], None, [completion_output], finished=True)
        request_output.request_timestamps = RequestTimestamps()
        await self.request_output_queue.put_nowait([request_output], server_info)

    async def abort(self, request_id):
        self._num_aborts += 1

    def testing_stats(self):
        return {"num_aborted_requests": self._num_aborts}

    @classmethod
    def from_args(cls, request_output_queue_type: QueueType):
        manager_class = ray.remote(num_cpus=1,
                                   name=get_manager_name(),
                                   namespace='llumnix',
                                   lifetime='detached')(cls)
        manager = manager_class.remote(request_output_queue_type)
        return manager

def setup_entrypoints_context(request_output_queue_type: QueueType):
    manager = ray.get_actor(get_manager_name(), namespace="llumnix")
    tests.unit_test.entrypoints.vllm.api.manager = manager
    ip = '127.0.0.1'
    port = 1234
    request_output_queue = init_request_output_queue_server(ip, port, request_output_queue_type)
    server_info = ServerInfo(random_uuid(), request_output_queue_type, request_output_queue, ip, port)
    entrypoints_context = EntrypointsContext(manager,
                                             {'0': None},
                                             request_output_queue,
                                             server_info,
                                             None,
                                             None)
    return entrypoints_context

def run_uvicorn_server(host: str, port: int, entrypoints_context: EntrypointsContext):
    llumnix.entrypoints.vllm.api_server.llumnix_client = LlumnixClientVLLM(entrypoints_context)
    app = tests.unit_test.entrypoints.vllm.api.app

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug",
        timeout_keep_alive=llumnix.entrypoints.vllm.api_server.SERVER_TIMEOUT_KEEP_ALIVE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-output-queue-type", type=str, choices=["zmq", "rayqueue"])
    entrypoints_args = parser.parse_args()

    request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)
    manager = MockManager.from_args(request_output_queue_type)
    entrypoints_context = setup_entrypoints_context(request_output_queue_type)

    run_uvicorn_server(entrypoints_args.host, entrypoints_args.port, entrypoints_context)
