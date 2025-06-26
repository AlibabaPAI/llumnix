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
import threading

from aiohttp import web
from aiohttp.web_runner import _raise_graceful_exit

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.entrypoints.bladellm.arg_utils import BladeLLMEngineArgs
from llumnix.logging.logger import init_logger
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.utils import get_ip_address, wait_port_free

logger = init_logger(__name__)


class APIServerActorBladeLLM(APIServerActor):
    def _set_host(self, entrypoints_args: EntrypointsArgs, engine_args):
        assert isinstance(engine_args, BladeLLMEngineArgs)
        # pylint: disable=import-outside-toplevel
        from blade_llm.service.args import ServingArgs
        engine_args: ServingArgs = engine_args.load_engine_args()
        if engine_args.host not in ("127.0.0.1", "0.0.0.0"):
            engine_args.host = get_ip_address()
        self.host = engine_args.host
        wait_port_free(entrypoints_args.port)

    def _start_server(self):
        self.run_server_thread = threading.Thread(
            target=self._run_server, args=(self.entrypoints_args, self.engine_args, self.entrypoints_context),
            daemon=True, name="run_server"
        )
        self.run_server_thread.start()

    def _set_health_api(self):
        self.health_api = ""

    def _run_server(self,
                    entrypoints_args: EntrypointsArgs,
                    engine_args: BladeLLMEngineArgs,
                    entrypoints_context: EntrypointsContext):
        # pylint: disable=import-outside-toplevel
        from llumnix.entrypoints.bladellm.api_server import LlumnixEntrypoint
        from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
        from blade_llm.service.args import ServingArgs
        # bladellm engine_args is dumped by pickle
        engine_args: ServingArgs = engine_args.load_engine_args()
        engine_args.host = self.host

        self.loop = asyncio.new_event_loop()
        llumnix_client = LlumnixClientBladeLLM(engine_args, entrypoints_context, self.loop)
        import llumnix.entrypoints.bladellm.api_server
        llumnix.entrypoints.bladellm.api_server.llumnix_client = llumnix_client
        web_app = LlumnixEntrypoint(client=llumnix_client, args=engine_args).create_web_app()
        # Loop is setted and closed inside.
        web.run_app(web_app, host=self.host, port=self.entrypoints_args.port, loop=self.loop, handle_signals=False)

    def _stop_server(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(_raise_graceful_exit)
