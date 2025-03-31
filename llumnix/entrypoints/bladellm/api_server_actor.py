import asyncio
import pickle
from aiohttp import web

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.logging.logger import init_logger
from llumnix.entrypoints.bladellm.api_server import LlumnixEntrypoint
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM

from llumnix.entrypoints.api_server_actor import APIServerActor

logger = init_logger(__name__)


class APIServerActorBladeLLM(APIServerActor):
    def _run_server(self,
                    entrypoints_args: EntrypointsArgs,
                    entrypoints_context: EntrypointsContext):
        # bladellm engine_args is dumped by pickle
        engine_args = pickle.loads(entrypoints_args.engine_args)
        loop = asyncio.new_event_loop()
        llumnix_client = LlumnixClientBladeLLM(engine_args, entrypoints_context, loop)
        web_app = LlumnixEntrypoint(client=llumnix_client, args=engine_args).create_web_app()
        logger.info("Start api server on '{}:{}'.".format(self.host, entrypoints_args.port))
        web.run_app(web_app, host=self.host, port=entrypoints_args.port, loop=loop, handle_signals=False)
