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

# TODO(s5u13b): Refine multiple import codes.
from llumnix.config import get_llumnix_config
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import (EntrypointsArgs, ManagerArgs, LlumnixArgumentParser,
                               DeploymentArgs)
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
from llumnix.entrypoints.bladellm.utils import get_args
from llumnix.entrypoints.utils import EntrypointsContext, DeploymentMode, is_gpu_available


def setup_llumnix_api_server(bladellm_args: ServingArgs, loop: asyncio.AbstractEventLoop):
    # generate llumnix_parser for checking parameters with choices
    # TODO(s5u13b): Add add_cli_args function.
    llumnix_parser = LlumnixArgumentParser()
    llumnix_parser = EntrypointsArgs.add_cli_args(llumnix_parser)
    llumnix_parser = ManagerArgs.add_cli_args(llumnix_parser)
    llumnix_config = get_llumnix_config(bladellm_args.llumnix_config)
    entrypoints_args, manager_args, engine_args = get_args(llumnix_config, llumnix_parser, bladellm_args)

    deployment_args = DeploymentArgs(deployment_mode=DeploymentMode.LOCAL, backend_type=BackendType.VLLM)

    setup_ray_cluster(entrypoints_args)

    llumnix_client = None
    # if gpu is not available, it means that this node is head pod x any llumnix components
    if is_gpu_available():
        instance_ids = None
        if engine_args.enable_disagg:
            instance_ids = [engine_args.disagg_options.inst_id]

        llumnix_context: EntrypointsContext = \
            setup_llumnix(manager_args, entrypoints_args, engine_args, deployment_args, instance_ids=instance_ids)
        llumnix_client = LlumnixClientBladeLLM(bladellm_args, llumnix_context, loop)

    return llumnix_client
