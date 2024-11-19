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

from typing import AsyncGenerator
from contextlib import asynccontextmanager
import time
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn


from llumnix.arg_utils import LlumnixArgumentParser
from llumnix.entrypoints.utils import (setup_ray_cluster,
                                       setup_llumnix,
                                       is_gpu_available,
                                       LlumnixEntrypointsContext,
                                       _background_process_outputs,
                                       init_per_token_latency_breakdown_dict,
                                       record_per_token_latency_breakdown)
from llumnix.entrypoints.bladellm.utils import (add_cli_args_llumnix,
                                            get_args_llumnix,
                                            manager_generate,
                                            manager_abort,
                                            manager_is_ready)
from llumnix.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.config import get_llumnix_config, LlumnixConfig
from llumnix.backends.backend_interface import BackendType

# Code file with __main__ should set the logger name to inherit the llumnix logger configuration.
logger = init_logger("llumnix.entrypoints.bladellm.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.

llumnix_context: LlumnixEntrypointsContext = None


# pylint: disable=unused-argument
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.create_task(llumnix_context.request_output_queue.run_server_loop())
    asyncio.create_task(_background_process_outputs(llumnix_context))
    yield
    llumnix_context.request_output_queue.cleanup()

app = FastAPI(lifespan=lifespan)



if __name__ == "__main__":
    parser: LlumnixArgumentParser = LlumnixArgumentParser()

    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)

    cli_args = add_cli_args_llumnix(parser)
    cfg: LlumnixConfig = get_llumnix_config(cli_args.config_file, cli_args)
    _, engine_manager_args, engine_args = get_args_llumnix(cfg, parser, cli_args)

    # Launch or connect to the ray cluster for multi-node serving.
    setup_ray_cluster(cfg)

    # if gpu is not available, it means that this node is head pod without any llumnix components
    if is_gpu_available():
        llumnix_context = setup_llumnix(cfg, engine_manager_args, BackendType.BLADELLM, 1, engine_args)
        # Start the api server after all the components of llumnix are ready.
        logger.info("Start Api Server on '{}:{}'".format(cfg.SERVER.HOST, cfg.SERVER.PORT))
        uvicorn.run(app,
                    host=cfg.SERVER.HOST,
                    port=cfg.SERVER.PORT,
                    log_level="debug",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=cfg.SERVER.SSL_KEYFILE,
                    ssl_certfile=cfg.SERVER.SSL_CERTFILE)
