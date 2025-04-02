import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.logging.logger import init_logger
from llumnix.entrypoints.api_server_actor import APIServerActor

logger = init_logger(__name__)


class APIServerActorVLLM(APIServerActor):
    def _run_server(self,
                    entrypoints_args: EntrypointsArgs,
                    engine_args: AsyncEngineArgs,
                    entrypoints_context: EntrypointsContext):
        # pylint: disable=import-outside-toplevel
        import llumnix.entrypoints.vllm.api_server
        from llumnix.entrypoints.vllm.client import LlumnixClientVLLM
        llumnix.entrypoints.vllm.api_server.llumnix_client = LlumnixClientVLLM(entrypoints_context)
        app = llumnix.entrypoints.vllm.api_server.app

        logger.info("Start api server on '{}:{}'.".format(self.host, entrypoints_args.port))
        uvicorn.run(app,
                    host=self.host,
                    port=entrypoints_args.port,
                    log_level=entrypoints_args.log_level,
                    timeout_keep_alive=llumnix.entrypoints.vllm.api_server.SERVER_TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=entrypoints_args.ssl_keyfile,
                    ssl_certfile=entrypoints_args.ssl_certfile)
