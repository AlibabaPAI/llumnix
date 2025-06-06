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

import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.logging.logger import init_logger
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.utils import get_ip_address
from llumnix.constants import SERVER_GRACEFUL_SHUTDOWN_TIMEOUT

logger = init_logger(__name__)


class APIServerActorVLLM(APIServerActor):
    def _set_host(self, entrypoints_args: EntrypointsArgs, engine_args):
        if entrypoints_args.host not in ("127.0.0.1", "0.0.0.0"):
            entrypoints_args.host = get_ip_address()
        self.host = entrypoints_args.host

    def _set_health_api(self):
        self.health_api = "health"

    def _run_server(self,
                    entrypoints_args: EntrypointsArgs,
                    engine_args: AsyncEngineArgs,
                    entrypoints_context: EntrypointsContext):
        # pylint: disable=import-outside-toplevel
        import llumnix.entrypoints.vllm.api_server
        from llumnix.entrypoints.vllm.client import LlumnixClientVLLM

        app = llumnix.entrypoints.vllm.api_server.app
        config = uvicorn.Config(
            app,
            host=self.host,
            port=entrypoints_args.port,
            log_level=entrypoints_args.server_log_level,
            timeout_keep_alive=llumnix.entrypoints.vllm.api_server.SERVER_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=entrypoints_args.ssl_keyfile,
            ssl_certfile=entrypoints_args.ssl_certfile,
            timeout_graceful_shutdown=SERVER_GRACEFUL_SHUTDOWN_TIMEOUT
        )
        self.server = uvicorn.Server(config)
        self.loop = asyncio.new_event_loop()
        llumnix.entrypoints.vllm.api_server.llumnix_client = LlumnixClientVLLM(entrypoints_context, self.loop)
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.server.serve())
        finally:
            self.loop.close()

    def _stop_server(self):
        def stop_server():
            self.server.should_exit = True

        if self.loop.is_running():
            self.loop.call_soon_threadsafe(stop_server)
