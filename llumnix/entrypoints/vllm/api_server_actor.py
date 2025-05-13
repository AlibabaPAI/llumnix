import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.logging.logger import init_logger
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.utils import get_ip_address

logger = init_logger(__name__)


class APIServerActorVLLM(APIServerActor):
    def _set_host(self, entrypoints_args: EntrypointsArgs, engine_args):
        if entrypoints_args.host not in ("127.0.0.1", "0.0.0.0"):
            entrypoints_args.host = get_ip_address()
        self.host = entrypoints_args.host

    def _run_server(self,
                    entrypoints_args: EntrypointsArgs,
                    engine_args: AsyncEngineArgs,
                    entrypoints_context: EntrypointsContext):
        # pylint: disable=import-outside-toplevel
        import llumnix.entrypoints.vllm.api_server
        from llumnix.entrypoints.vllm.client import LlumnixClientVLLM
        llumnix.entrypoints.vllm.api_server.llumnix_client = LlumnixClientVLLM(entrypoints_context)
        app = llumnix.entrypoints.vllm.api_server.app

        logger.info("Start api server on '{}:{}'.".format(entrypoints_args.host, entrypoints_args.port))
        uvicorn.run(app,
                    host=entrypoints_args.host,
                    port=entrypoints_args.port,
                    log_level=entrypoints_args.server_log_level,
                    timeout_keep_alive=llumnix.entrypoints.vllm.api_server.SERVER_TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=entrypoints_args.ssl_keyfile,
                    ssl_certfile=entrypoints_args.ssl_certfile)
