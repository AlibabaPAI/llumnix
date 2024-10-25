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
from aiohttp import web

from client import GeneralLLMClientLlumnix, DummyAsyncLLMEngineClient
from blade_llm.model.config_utils import GenerationConfigProcessor
from blade_llm.service.server import Entrypoint


from llumnix.arg_utils import LlumnixArgumentParser
from llumnix.entrypoints.utils import (setup_ray_cluster,
                                       setup_llumnix,
                                       is_gpu_available,
                                       LlumnixEntrypointsContext,
                                       _background_process_outputs,
                                       init_per_token_latency_breakdown_dict,
                                       record_per_token_latency_breakdown)
from llumnix.entrypoints.bladellm.utils import (add_cli_args,
                                            get_args,
                                            get_model_conf)
from llumnix.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.config import get_llumnix_config, LlumnixConfig
from llumnix.backends.backend_interface import BackendType

# Code file with __main__ should set the logger name to inherit the llumnix logger configuration.
logger = init_logger("llumnix.entrypoints.vllm.api_server")

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

def setup_llumnix_api_server(bladellm_args):
    llumnixParser: LlumnixArgumentParser = LlumnixArgumentParser()

    llumnixParser.add_argument("--host", type=str)
    llumnixParser.add_argument("--port", type=int)
    llumnixParser.add_argument("--ssl-keyfile", type=str)
    llumnixParser.add_argument("--ssl-certfile", type=str)

    add_cli_args(llumnixParser)
    # TODO[xinyi]: now only support use config_file to define llumnix arguments
    llumnixCfg: LlumnixConfig = get_llumnix_config(bladellm_args.config_file, bladellm_args)
    _, engine_manager_args, engine_args = get_args(llumnixCfg, llumnixParser, bladellm_args)
    setup_ray_cluster(llumnixCfg)

    # if gpu is not available, it means that this node is head pod x any llumnix components
    if is_gpu_available():
        world_size = engine_args.tensor_parallel_size * engine_args.tensor_parallel_size 
        llumnix_context = setup_llumnix(llumnixCfg, engine_manager_args, BackendType.BLADELLM, world_size, engine_args)
        engine_model_conf = get_model_conf(bladellm_args)
        llm_client = GeneralLLMClientLlumnix(bladellm_args, DummyAsyncLLMEngineClient(), engine_model_conf)
        return engine_model_conf, llm_client
    return None, None

# TODO[xinyi]: benchmark metrics
# TODO[xinyi]: revise in bladellm repo
# def main():
#     parser = add_args()
#     args = parser.parse_args()
#     args = ServingArgs.from_cli_args(args)

#     # Check whether FP8 paged kvcache quant is appointed to use and could be imported under current arch.
#     # If not, fallback to non-quant kvcache.
#     if (
#         args.load_model_options.kv_cache_quant
#         in ['fp8_e5m2', 'fp8_e4m3', "mix_f852i4", "mix_f843i4", "mix_i8i4", "mix_i4i4"]
#         and not fp8_paged_enabled()
#     ):
#         logger.warning(
#             "Experimental feature FP8 KV-Cache could not be imported, architecture may be incompatible, fallback to non-quant KV-Cache."
#         )
#         args.load_model_options.kv_cache_quant = 'no_quant'

#     logger.remove()
#     logger.add(sys.stderr, level=args.log_level)
#     logger.info('================ Serving Arguments ================')
#     for k, v in args.__dict__.items():
#         logger.info(f"{k:>20}: {v}")

#     # check port first
#     check_ports(args)

#     init_metric(args.serving_metric_options.metric_export_interval_sec, *args.metric_exporters)

#     loop = asyncio.get_event_loop()

#     if args.enable_llumnix:
#         import llumnix
#         engine_model_conf, llm_client = setup_llumnix_api_server(args)
#     else:
#         llm_engine = AsyncLLMEngine(args)
#         engine_model_conf = llm_engine.model_conf
#         llm_engine.start(loop)
#         llm_client = llm_engine.get_client()

#     try:
#         generation_conf_processor = GenerationConfigProcessor(args.generation_configs, engine_model_conf)
#     except Exception:
#         logger.exception('Failed to load generation config processor when create server.')
#         generation_conf_processor = None

#     # start entrypoint server
#     web_app = Entrypoint(
#         client=llm_client,
#         model_conf=engine_model_conf,
#         generation_conf_processor=generation_conf_processor,
#         chat_template_path=args.load_model_options.chat_template,
#         pp_enabled=args.pipeline_parallel_size > 1,
#     ).create_web_app()
#     logger.info(f"Entrypoint API ready at {args.host}:{args.port}")
#     web.run_app(web_app, host=args.host, port=args.port, loop=loop)