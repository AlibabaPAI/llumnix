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

import time
import asyncio

import pickle
import ray
from aiohttp import web

from blade_llm.service.args import ServingArgs
from blade_llm.service.server import Entrypoint

from llumnix.logging.logger import init_logger
from llumnix.config import get_llumnix_config
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import (EntrypointsArgs, ManagerArgs, LlumnixArgumentParser,
                               LaunchArgs, InstanceArgs)
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
from llumnix.entrypoints.utils import EntrypointsContext, LaunchMode, is_gpu_available
from llumnix.entrypoints.bladellm.arg_utils import BladellmEngineArgs

logger = init_logger(__name__)

llumnix_context: EntrypointsContext = None

class LlumnixEntrypoint(Entrypoint):
    async def generate_benchmark(self, request: web.Request):
        oai_req, server_req = await self._web_request_to_oai_chat_request(request)
        start = time.time()
        results_generator = await self._client.add_request(server_req)

        # Non-streaming case
        tokens = []
        per_token_latency = []
        streamer = results_generator.async_stream()
        async for r in streamer:
            now = time.time()
            per_token_latency.append([now, (now - start)*1000])
            start = now
            tokens.extend(r.tokens)
            assert r.error_info is None, f"Some errors occur, benchmark is stopping: {r.error_info}."

        output_text = "".join([tok.text for tok in tokens])
        num_output_tokens = len(tokens)
        expected_resp_len = oai_req.max_tokens

        assert max(expected_resp_len, 1) == max(num_output_tokens, 1), \
            f"Output token length dismatch: expected {expected_resp_len}, got {num_output_tokens}"

        ret = {
            'request_id': server_req.external_id,
            'generated_text': oai_req.messages[1]["content"] + output_text,
            'num_output_tokens_cf': num_output_tokens,
            'per_token_latency': per_token_latency,
        }
        return web.json_response(data=ret)

    # pylint: disable=unused-argument
    async def is_ready(self, request):
        responce = await self._client.is_ready()
        return web.json_response(text=str(responce))

    def create_web_app(self):
        app = super().create_web_app()
        app.add_routes([
            web.post('/generate_benchmark', self.generate_benchmark),
            web.get('/is_ready', self.is_ready)
        ])
        app.on_cleanup.append(clean_up_llumnix_components)
        return app

# pylint: disable=unused-argument
async def clean_up_llumnix_components(app):
    for instance in llumnix_context.instances.values():
        try:
            ray.kill(instance)
        # pylint: disable=bare-except
        except:
            pass


def detect_unsupported_engine_feature(engine_args: ServingArgs) -> None:
    unsupported_feature = None
    if engine_args.enable_lora:
        unsupported_feature = "multi-lora serving"
    elif not engine_args.disable_prompt_cache:
        unsupported_feature = "automatic prompt caching"
    elif engine_args.use_sps:
        unsupported_feature = "speculative decoding"
    elif engine_args.enable_remote_worker:
        unsupported_feature = "enable_remote_worker"
    elif engine_args.enable_hybrid_dp:
        unsupported_feature = "hybrid data parallel"

    if unsupported_feature:
        raise ValueError(f'Llumnix does not support "{unsupported_feature}" for BladeLLM currently.')

def get_args(llumnix_cfg, llumnix_parser, engine_args: ServingArgs):
    instance_args = InstanceArgs.from_llumnix_config(llumnix_cfg)
    instance_args.init_from_engine_args(engine_args, BackendType.BLADELLM)
    manager_args = ManagerArgs.from_llumnix_config(llumnix_cfg)
    manager_args.init_from_instance_args(instance_args)
    entrypoints_args = EntrypointsArgs.from_llumnix_config(llumnix_cfg)

    EntrypointsArgs.check_args(entrypoints_args, llumnix_parser)
    instance_args.check_args(instance_args, manager_args, LaunchMode.LOCAL, llumnix_parser)
    ManagerArgs.check_args(manager_args, llumnix_parser)

    assert not instance_args.simulator_mode, "Only support the simulator mode for vLLM."

    assert not (engine_args.enable_disagg and manager_args.enable_pd_disagg), \
        "Cannot enable both pd-disaggregation inside the LLM engine and pd-disaggregation from Lluminx."

    assert 'W' not in instance_args.request_migration_policy, \
        "Migrating waiting request is not supported for bladellm temporarily."

    assert not engine_args.enable_disagg or not manager_args.enable_migration, \
        "Migration feature is temporarily unavailable for the engine based pd-disaggregation."

    assert engine_args.pipeline_parallel_size == 1 or not manager_args.enable_migration,\
         "Migration feature is temporarily unavailable for pipeline parallel in BladeLLM."

    detect_unsupported_engine_feature(engine_args)

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args

def setup_llumnix_api_server(bladellm_args: ServingArgs, loop: asyncio.AbstractEventLoop):
    # generate llumnix_parser for checking parameters with choices
    llumnix_parser = LlumnixArgumentParser()
    llumnix_parser = EntrypointsArgs.add_cli_args(llumnix_parser)
    llumnix_parser = ManagerArgs.add_cli_args(llumnix_parser)
    llumnix_parser = InstanceArgs.add_cli_args(llumnix_parser)
    llumnix_config = get_llumnix_config(bladellm_args.llumnix_config, cli_args=bladellm_args.llumnix_opts)

    entrypoints_args, manager_args, instance_args, engine_args = \
        get_args(llumnix_config, llumnix_parser, bladellm_args)

    launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=BackendType.BLADELLM)
    setup_ray_cluster(entrypoints_args)

    llumnix_client = None
    # If gpu is not available, it means that this node is head pod x any llumnix components
    if is_gpu_available():
        # Since importing the bladellm engine arguments require a GPU, serialize the engine parameters
        # before passing them to the manager.
        llumnix_engine_args = BladellmEngineArgs()
        llumnix_engine_args.engine_args = pickle.dumps(engine_args)
        llumnix_engine_args.world_size = bladellm_args.tensor_parallel_size*bladellm_args.pipeline_parallel_size

        global llumnix_context
        llumnix_context = \
            setup_llumnix(entrypoints_args, manager_args, instance_args, llumnix_engine_args, launch_args)
        llumnix_client = LlumnixClientBladeLLM(bladellm_args, llumnix_context, loop)

    return llumnix_client
