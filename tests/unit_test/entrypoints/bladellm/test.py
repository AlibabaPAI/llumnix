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

# from blade_llm.protocol import GenerateStreamResponse, ErrorInfo
from blade_llm.service.server import Entrypoint

import llumnix.entrypoints.bladellm.api_server
import llumnix.llm_engine_manager
from llumnix.arg_utils import EngineManagerArgs
from llumnix.server_info import ServerInfo, RequestTimestamps
from llumnix.utils import random_uuid
from llumnix.queue.utils import init_output_queue_server, init_output_queue_client, QueueType
from llumnix.entrypoints.utils import LlumnixEntrypointsContext
from llumnix.entrypoints.bladellm.api_server import EntrypointLlumnix, DummyAsyncLLMEngineClient
engine_manager = None
MANAGER_ACTOR_NAME = llumnix.llm_engine_manager.MANAGER_ACTOR_NAME
from llumnix.logger import init_logger

logger = init_logger(__name__)

import ray
from pydantic import BaseModel, Field
from typing import List, Optional


class GenerateStreamResponse():
    is_ok: bool = Field(default=True, title='if generation step success.')
    is_finished: bool = Field(default=False, title='if generation for whole sequence is finished.')
    tokens: List[str] = Field(default=[], title='generated tokens.')
    texts: List[str] = Field(default=[], title='generated texts from beam search.')
    detail: Optional[dict] = None
    error_info: Optional[dict] = None
    def to_serializable_dict(self):
        return self.model_dump()
    
@ray.remote(num_cpus=0)
class MockLLMEngineManager:
    def __init__(self,):
        self._num_generates = 0
        self._num_aborts = 0

    async def generate(self, request_id, server_info, *args, **kwargs):
        self._num_generates += 1
        request_output = GenerateStreamResponse(is_finished=True, is_ok=True, texts=["done"]).to_serializable_dict()
        # request_output.request_timestamps = RequestTimestamps()
        # await self.request_output_queue.put_nowait([request_output], server_info)

    async def abort(self, request_id):
        self._num_aborts += 1

    def testing_stats(self):
        return {"num_aborted_requests": self._num_aborts}


def init_manager():
    engine_manager = MockLLMEngineManager.options(name=MANAGER_ACTOR_NAME,
                                                  namespace='llumnix').remote()
    return engine_manager

if __name__ == "__main__":
    ip = '127.0.0.1'
    port = 1234
    engine_manager = init_manager()
    
    loop = asyncio.get_event_loop()
    app = web.Application()
    web.run_app(app, host="localhost", port=8000, loop=loop)