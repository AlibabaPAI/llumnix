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
from typing import Optional
from aiohttp import web

from blade_llm.service.args import ServingArgs
from blade_llm.service.server import Entrypoint
from blade_llm.service.disagg_decode_server import DecodeEntrypoint
from blade_llm.service.communications import BaseLLMClient

from llumnix.entrypoints.bladellm.utils import get_args
from llumnix.config import get_llumnix_config, LlumnixConfig
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import LlumnixEntrypointsArgs, EngineManagerArgs, LlumnixArgumentParser
from llumnix.entrypoints.utils import setup_ray_cluster, setup_llumnix, is_gpu_available
from llumnix.entrypoints.bladellm.client import AsyncLLMEngineClientLlumnix, background_process_outputs
from llumnix.entrypoints.utils import LlumnixEntrypointsContext
from llumnix.logger import init_logger

logger = init_logger(__name__)

llumnix_context: LlumnixEntrypointsContext = None

# pylint: disable=unused-argument
async def on_startup(app):
    app['server_task'] = asyncio.create_task(llumnix_context.request_output_queue.run_server_loop())
    app['background_task'] = asyncio.create_task(background_process_outputs(llumnix_context))

async def on_cleanup(app):
    app['server_task'].cancel()
    app['background_task'].cancel()
    await asyncio.gather(app['server_task'], app['background_task'], return_exceptions=True)
    llumnix_context.request_output_queue.cleanup()

app = web.Application()
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

class EntrypointLlumnix(Entrypoint):
    def __init__(self, client: BaseLLMClient, args: Optional[ServingArgs]):
        super().__init__(client, args)
        global app
        self.app = app

class DecodeEntrypointLlumnix:
    def __init__(self, *args, **kwargs):
        pass

def setup_llumnix_api_server(bladellm_args: ServingArgs):
    # generate llumnix_parser for checking parameters with choices
    llumnix_parser: LlumnixArgumentParser = LlumnixArgumentParser()
    llumnix_parser = LlumnixEntrypointsArgs.add_cli_args(llumnix_parser)
    llumnix_parser = EngineManagerArgs.add_cli_args(llumnix_parser)
    llumnix_config: LlumnixConfig = get_llumnix_config(bladellm_args.llumnix_config)
    _, engine_manager_args, engine_args = get_args(llumnix_config, llumnix_parser, bladellm_args)

    setup_ray_cluster(llumnix_config)

    llm_client, entrypoint_cls, decode_entrypoint_cls = None, None, None
    # if gpu is not available, it means that this node is head pod x any llumnix components
    if is_gpu_available():
        world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
        global llumnix_context
        llumnix_context = setup_llumnix(llumnix_config, engine_manager_args, BackendType.BLADELLM, world_size, engine_args)
        llm_client = AsyncLLMEngineClientLlumnix(bladellm_args)
        entrypoint_cls = EntrypointLlumnix 
        decode_entrypoint_cls = DecodeEntrypointLlumnix

    return llm_client, entrypoint_cls, decode_entrypoint_cls
