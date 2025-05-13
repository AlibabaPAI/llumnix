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

from aiohttp import web
import ray

from blade_llm.service.args import ServingArgs
from blade_llm.service.server import Entrypoint


from llumnix.config import get_llumnix_config
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
from llumnix.entrypoints.utils import LaunchMode, is_gpu_available, EntrypointsContext
from llumnix.entrypoints.bladellm.arg_utils import BladellmEngineArgs, add_cli_args, get_args
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)

entrypoints_context: EntrypointsContext = None


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
    for instance in entrypoints_context.instances.values():
        try:
            ray.kill(instance)
        # pylint: disable=bare-except
        except:
            pass

def setup_llumnix_api_server(engine_args: ServingArgs, loop: asyncio.AbstractEventLoop):
    # generate llumnix_parser for checking parameters with choices
    parser = LlumnixArgumentParser()
    parser = add_cli_args(parser, add_engine_args=False)
    # llumnix_opts is used to receive config options
    llumnix_config = get_llumnix_config(engine_args.llumnix_config, opts=engine_args.llumnix_opts)

    entrypoints_args, manager_args, instance_args, engine_args = get_args(llumnix_config, LaunchMode.LOCAL, parser, engine_args=engine_args)
    launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=BackendType.BLADELLM)

    setup_ray_cluster(entrypoints_args)

    llumnix_client = None
    # If gpu is not available, it means that this node is head pod without any llumnix components.
    if is_gpu_available():
        # Since importing the bladellm engine arguments requires available GPU,
        # serialize the engine parameters before passing them to the manager.
        bladellm_engine_args = BladellmEngineArgs(pickle.dumps(engine_args))
        bladellm_engine_args.world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
        if engine_args.disagg_options is not None:
            bladellm_engine_args.instance_id = engine_args.disagg_options.inst_id

        global entrypoints_context
        entrypoints_context = setup_llumnix(entrypoints_args, manager_args, instance_args, bladellm_engine_args, launch_args)
        llumnix_client = LlumnixClientBladeLLM(engine_args, entrypoints_context, loop)

    return llumnix_client
