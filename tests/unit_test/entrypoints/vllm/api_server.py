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
import asyncio

import uvicorn
import ray

import llumnix.entrypoints.vllm.api_server
from llumnix.server_info import ServerInfo
from llumnix.utils import random_uuid
from llumnix.ray_utils import get_manager_name, get_instance_name, get_instance
from llumnix.queue.utils import init_request_output_queue_server, init_request_output_queue_client, QueueType
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.entrypoints.vllm.client import LlumnixClientVLLM

import tests.unit_test.entrypoints.vllm.api
from tests.unit_test.entrypoints.vllm.test_client import get_request_output_engine


@ray.remote
class MockLlumlet:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self._num_aborts = 0

    def abort(self, request_id):
        self._num_aborts += 1

    def testing_stats(self):
        return {"num_aborted_requests": self._num_aborts}


class MockManager:
    def __init__(self, request_output_queue_type: QueueType, instance_id):
        self._num_generates = 0
        self.request_output_queue = init_request_output_queue_client(request_output_queue_type)
        self.instance_id = instance_id

    async def generate(self, request_id, server_info, *args, **kwargs):
        self._num_generates += 1
        await self.request_output_queue.put_nowait(
            [get_request_output_engine(request_id, self.instance_id, finished=True)], server_info)

    @classmethod
    def from_args(cls, request_output_queue_type: QueueType, instance_id):
        manager_class = ray.remote(num_cpus=1,
                                   name=get_manager_name(),
                                   namespace='llumnix',
                                   lifetime='detached')(cls)
        manager = manager_class.remote(request_output_queue_type, instance_id)
        return manager

class MockLlumnixClientVLLM(LlumnixClientVLLM):
    def _clear_client_request_states(self, request_id: str):
        if request_id in self.request_stream:
            self.request_stream[request_id].finish()


def setup_entrypoints_context(request_output_queue_type: QueueType, instance_id):
    manager = ray.get_actor(get_manager_name(), namespace="llumnix")
    tests.unit_test.entrypoints.vllm.api.manager = manager
    instance = get_instance(instance_id)
    tests.unit_test.entrypoints.vllm.api.instance = instance
    ip = '127.0.0.1'
    request_output_queue = init_request_output_queue_server(ip, request_output_queue_type)
    server_info = ServerInfo(random_uuid(), request_output_queue_type, request_output_queue,
                             ip, request_output_queue.port)
    entrypoints_context = EntrypointsContext(manager,
                                             {'0': None},
                                             request_output_queue,
                                             None,
                                             server_info,
                                             None,
                                             None)
    return entrypoints_context

def run_server(host: str, port: int, entrypoints_context: EntrypointsContext):
    app = tests.unit_test.entrypoints.vllm.api.app
    loop = asyncio.new_event_loop()
    llumnix_client = MockLlumnixClientVLLM(entrypoints_context, loop)
    llumnix.entrypoints.vllm.api_server.llumnix_client = llumnix_client
    asyncio.set_event_loop(loop)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="debug",
        timeout_keep_alive=llumnix.entrypoints.vllm.api_server.SERVER_TIMEOUT_KEEP_ALIVE
    )
    server = uvicorn.Server(config)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-output-queue-type", type=str, choices=["zmq", "rayqueue"])
    entrypoints_args = parser.parse_args()

    instance_id = random_uuid()
    instance = MockLlumlet.options(name=get_instance_name(instance_id),
                                   namespace="llumnix").remote(instance_id)
    tests.unit_test.entrypoints.vllm.api.instance = instance

    request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)
    manager = MockManager.from_args(request_output_queue_type, instance_id)

    entrypoints_context = setup_entrypoints_context(request_output_queue_type, instance_id)
    run_server(entrypoints_args.host, entrypoints_args.port, entrypoints_context)
