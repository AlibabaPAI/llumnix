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
from blade_llm.service.args import ServingArgs

from llumnix.config import get_llumnix_config, LlumnixConfig
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import LlumnixEntrypointsArgs, EngineManagerArgs, LlumnixArgumentParser
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix, is_gpu_available
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
from llumnix.entrypoints.setup import LlumnixEntrypointsContext
from llumnix.entrypoints.bladellm.utils import get_args
from llumnix.utils import random_uuid

def setup_llumnix_api_server(bladellm_args: ServingArgs, loop: asyncio.AbstractEventLoop):
    # generate llumnix_parser for checking parameters with choices
    llumnix_parser: LlumnixArgumentParser = LlumnixArgumentParser()
    llumnix_parser = LlumnixEntrypointsArgs.add_cli_args(llumnix_parser)
    llumnix_parser = EngineManagerArgs.add_cli_args(llumnix_parser)
    llumnix_config: LlumnixConfig = get_llumnix_config(bladellm_args.llumnix_config)
    _, engine_manager_args, instance_args, engine_args = get_args(llumnix_config, llumnix_parser, bladellm_args)

    setup_ray_cluster(llumnix_config)

    bladellm_args.worker_socket_path = bladellm_args.worker_socket_path + "." + random_uuid()
    llm_client = None
    # if gpu is not available, it means that this node is head pod x any llumnix components
    if is_gpu_available():
        world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
        llumnix_context: LlumnixEntrypointsContext = \
            setup_llumnix(engine_manager_args, instance_args, engine_args, llumnix_config,
                          BackendType.BLADELLM, world_size)
        llm_client = LlumnixClientBladeLLM(bladellm_args, llumnix_context, loop)

    return llm_client
