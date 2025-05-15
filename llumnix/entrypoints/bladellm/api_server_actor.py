import asyncio

from aiohttp import web
from aiohttp.web_runner import _raise_graceful_exit

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.entrypoints.bladellm.arg_utils import BladellmEngineArgs
from llumnix.logging.logger import init_logger
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.utils import get_ip_address, wait_port_free

logger = init_logger(__name__)


class APIServerActorBladeLLM(APIServerActor):
    def _set_host(self, entrypoints_args: EntrypointsArgs, engine_args):
        assert isinstance(engine_args, BladellmEngineArgs)
        # pylint: disable=import-outside-toplevel
        from blade_llm.service.args import ServingArgs
        engine_args: ServingArgs = engine_args.unwrap_engine_args_if_needed()
        if engine_args.host not in ("127.0.0.1", "0.0.0.0"):
            engine_args.host = get_ip_address()
        self.host = engine_args.host
        wait_port_free(entrypoints_args.port)

    def _set_health_api(self):
        self.health_api = ""

    def _run_server(self,
                    entrypoints_args: EntrypointsArgs,
                    engine_args: BladellmEngineArgs,
                    entrypoints_context: EntrypointsContext):
        # pylint: disable=import-outside-toplevel
        from llumnix.entrypoints.bladellm.api_server import LlumnixEntrypoint
        from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
        from blade_llm.service.args import ServingArgs
        # bladellm engine_args is dumped by pickle
        engine_args: ServingArgs = engine_args.unwrap_engine_args_if_needed()
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
