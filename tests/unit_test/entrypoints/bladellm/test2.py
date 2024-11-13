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
import ray
from aiohttp import web

engine_manager = None


    
@ray.remote(num_cpus=0)
class MockLLMEngineManager:
    def __init__(self):
        self._num_generates = 0
        self._num_aborts = 0
        # self.request_output_queue = init_output_queue_client(output_queue_type)


    async def abort(self, request_id):
        self._num_aborts += 1

    def testing_stats(self):
        return {"num_aborted_requests": self._num_aborts}


def init_manager():
    engine_manager = MockLLMEngineManager.options(name="MANAGER_ACTOR_NAME",
                                                  namespace='llumnix').remote()
    return engine_manager

async def handle(request):
    return web.Response(text="Hello, world")

if __name__ == "__main__":

    engine_manager = init_manager()
    
    loop = asyncio.get_event_loop()
    app = web.Application()
    app.router.add_get('/', handle)
    web.run_app(app, host="localhost", port=8000, loop=loop)
