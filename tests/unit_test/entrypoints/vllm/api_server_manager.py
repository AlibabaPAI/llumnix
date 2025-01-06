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
from fastapi.responses import JSONResponse, Response

from vllm.outputs import CompletionOutput, RequestOutput

import llumnix.entrypoints.vllm.api_server
import llumnix.manager
from llumnix.arg_utils import ManagerArgs
from llumnix.server_info import ServerInfo, RequestTimestamps
from llumnix.utils import random_uuid, get_manager_name
from llumnix.queue.utils import init_request_output_queue_server, init_request_output_queue_client, QueueType
from llumnix.entrypoints.setup import EntrypointsContext
from llumnix.entrypoints.vllm.client import LlumnixClientVLLM

app = llumnix.entrypoints.vllm.api_server.app
manager = None


@ray.remote(num_cpus=0)
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


def init_manager(request_output_queue_type: QueueType):
    manager = MockManager.options(name=get_manager_name(),
                                  namespace='llumnix').remote(request_output_queue_type)
    return manager

@app.get("/stats")
def stats() -> Response:
    """Get the statistics of the engine."""
    return JSONResponse(ray.get(manager.testing_stats.remote()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-output-queue-type", type=str, choices=["zmq", "rayqueue"])
    parser = ManagerArgs.add_cli_args(parser)
    args = parser.parse_args()

    request_output_queue_type = QueueType(args.request_output_queue_type)
    manager = init_manager(request_output_queue_type)
    ip = '127.0.0.1'
    port = 1234
    request_output_queue = init_request_output_queue_server(ip, port, request_output_queue_type)
    server_info = ServerInfo(random_uuid(), request_output_queue_type, request_output_queue, ip, port)
    llumnix_context = EntrypointsContext(manager,
                                         {'0': None},
                                         request_output_queue,
                                         server_info,
                                         None,
                                         None)
    llumnix.entrypoints.vllm.api_server.llumnix_client = LlumnixClientVLLM(llumnix_context)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=llumnix.entrypoints.vllm.api_server.TIMEOUT_KEEP_ALIVE)
