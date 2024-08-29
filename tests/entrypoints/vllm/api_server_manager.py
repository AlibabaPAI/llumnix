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
from ray.util.queue import Queue as RayQueue

from vllm.outputs import CompletionOutput, RequestOutput

import llumnix.entrypoints.vllm.api_server
import llumnix.llm_engine_manager
from llumnix.arg_utils import EngineManagerArgs


app = llumnix.entrypoints.vllm.api_server.app
engine_manager = None
request_output_queue = RayQueue()
llumnix.entrypoints.vllm.api_server.request_output_queue = request_output_queue
MANAGER_ACTOR_NAME = llumnix.llm_engine_manager.MANAGER_ACTOR_NAME


@ray.remote(num_cpus=0)
class MockLLMEngineManager:
    def __init__(self):
        self._num_generates = 0
        self._num_aborts = 0

    async def generate(self, request_id, server_info, *args, **kwargs):
        self._num_generates += 1
        completion_output = CompletionOutput(0, "", [], 0.0, None)
        request_output = RequestOutput(request_id, "", [], None, [completion_output], finished=True)
        request_output_queue.put(request_output)

    async def abort(self, request_id):
        self._num_aborts += 1

    def testing_stats(self):
        return {"num_aborted_requests": self._num_aborts}


def init_manager():
    engine_manager = MockLLMEngineManager.options(name=MANAGER_ACTOR_NAME,
                                                  namespace='llumnix').remote()
    return engine_manager

@app.get("/stats")
def stats() -> Response:
    """Get the statistics of the engine."""
    return JSONResponse(ray.get(engine_manager.testing_stats.remote()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser = EngineManagerArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_manager = init_manager()
    llumnix.entrypoints.vllm.api_server.engine_manager = engine_manager

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=llumnix.entrypoints.vllm.api_server.TIMEOUT_KEEP_ALIVE)
