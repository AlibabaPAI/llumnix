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
import llumnix.entrypoints.bladellm
import asyncio
import ray
from aiohttp import web
from fastapi.responses import JSONResponse, Response

from blade_llm.service.server import Entrypoint

import llumnix.entrypoints.bladellm.api_server
import llumnix.llm_engine_manager
from llumnix.arg_utils import EngineManagerArgs
from llumnix.server_info import ServerInfo, RequestTimestamps
from llumnix.utils import random_uuid
from llumnix.queue.utils import init_output_queue_server, init_output_queue_client, QueueType
from llumnix.entrypoints.utils import LlumnixEntrypointsContext
from llumnix.entrypoints.bladellm.api_server import EntrypointLlumnix, DummyAsyncLLMEngineClient
from blade_llm.protocol import GenerateStreamResponse, ErrorInfo
from llumnix.backends.bladellm.sequence import GenerateStreamResponseLlumnix

from blade_llm.protocol import Token

engine_manager = None
MANAGER_ACTOR_NAME = llumnix.llm_engine_manager.MANAGER_ACTOR_NAME
from llumnix.logger import init_logger

logger = init_logger(__name__)

class MockEntrypoint(EntrypointLlumnix):
    def stats() -> Response:
        """Get the statistics of the engine."""
        return web.json_response(text=ray.get(engine_manager.testing_stats.remote()))
    
    async def _create_request(self, req_text: str, *args, **kwargs):
        return req_text

class MockLLMClient(DummyAsyncLLMEngineClient):
    def get_tokenizer(self):
        return None
    
@ray.remote(num_cpus=0)
class MockLLMEngineManager:
    def __init__(self, output_queue_type: QueueType):
        self._num_generates = 0
        self._num_aborts = 0
        self.request_output_queue = init_output_queue_client(output_queue_type)

    async def generate(self, request_id, server_info, *args, **kwargs):
        self._num_generates += 1
        request_output = {'is_ok': True, 'is_finished': True, 'tokens': [{'id': 0, 'text': ' kernel::$.Q_open', 'logprob': None, 'is_special': False, 'bytes': None, 'top_logprobs': None}], 'texts': [], 'detail': None, 'error_info': None, 'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}, 'logprobs': None, 'request_id': request_id}
        await self.request_output_queue.put_nowait([request_output], server_info)

    async def abort(self, request_id):
        self._num_aborts += 1

    def testing_stats(self):
        return {"num_aborted_requests": self._num_aborts}


def init_manager(output_queue_type: QueueType):
    engine_manager = MockLLMEngineManager.options(name=MANAGER_ACTOR_NAME,
                                                  namespace='llumnix').remote(output_queue_type)
    return engine_manager

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--output-queue-type", type=str, default="zmq", choices=["zmq", "rayqueue"])
        parser = EngineManagerArgs.add_cli_args(parser)
        args = parser.parse_args()

        output_queue_type = QueueType(args.output_queue_type)
        engine_manager = init_manager(output_queue_type)
        llumnix.entrypoints.bladellm.api_server.llumnix_context = LlumnixEntrypointsContext()
        llumnix.entrypoints.bladellm.api_server.llumnix_context.engine_manager = engine_manager
        ip = '127.0.0.1'
        port = 1234
        llumnix.entrypoints.bladellm.api_server.llumnix_context.request_output_queue = \
            init_output_queue_server(ip, port, output_queue_type)
        ray_queue_server = None
        if output_queue_type == QueueType.RAYQUEUE:
            ray_queue_server = llumnix.entrypoints.bladellm.api_server.llumnix_context.request_output_queue
        server_info = ServerInfo(random_uuid(), output_queue_type, ray_queue_server, ip, port)
        llumnix.entrypoints.bladellm.api_server.llumnix_context.server_info = server_info
        
        loop = asyncio.get_event_loop()
        web_app = EntrypointLlumnix(client=MockLLMClient()).create_web_app()
        logger.info(f"Entrypoint API ready at {args.host}:{args.port}")
        web.run_app(web_app, host=args.host, port=args.port, loop=loop)
    except Exception as e:
        print("error",e)
        import traceback
        print(traceback.format_exc())
