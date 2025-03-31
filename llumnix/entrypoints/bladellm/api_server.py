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

from blade_llm.service.args import ServingArgs, add_args
from blade_llm.service.server import check_ports, init_engine_and_client, Entrypoint
from blade_llm.service.elastic_attn.elastic_attention_inst_manager import InstManager
from blade_llm.service.elastic_attn.elastic_attention_entrypoint import ElasticAttnEntrypoint

from llumnix.config import get_llumnix_config
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
from llumnix.entrypoints.utils import LaunchMode, is_gpu_available
from llumnix.entrypoints.bladellm.arg_utils import BladellmEngineArgs, add_llumnix_cli_args, get_args
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


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
        # TODO(s5u13b): Add cleanup codes to kill instance.
        return app


# pylint: disable=redefined-outer-name, unused-argument
def setup_llumnix_api_server(engine_args: ServingArgs, loop: asyncio.AbstractEventLoop):
    llumnix_client = engine_args.llumnix_client

    return llumnix_client


if __name__ == "__main__":
    parser = add_args()
    cli_args = parser.parse_args()
    engine_args = ServingArgs.from_cli_args(cli_args)

    # check port first
    check_ports(engine_args)

    # TODO(s5u13b): Fix it, cannot use parser of bladellm because Llumnix need to set namespace.
    # generate llumnix_parser for checking parameters with choices
    parser: LlumnixArgumentParser = LlumnixArgumentParser()
    parser = add_llumnix_cli_args(parser)
    llumnix_config = get_llumnix_config(engine_args.llumnix_config, cli_args=engine_args.llumnix_opts)

    entrypoints_args, manager_args, instance_args, engine_args = get_args(llumnix_config, parser, engine_args)
    launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=BackendType.BLADELLM)

    # Launch or connect to the ray cluster for multi-node serving.
    setup_ray_cluster(entrypoints_args)

    if is_gpu_available():
        # Since importing the bladellm engine arguments requires available GPU,
        # serialize the engine parameters before passing them to the manager.
        engine_args_llumnix = BladellmEngineArgs()
        engine_args_llumnix.engine_args = pickle.dumps(engine_args)
        engine_args_llumnix.world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
        entrypoints_context = setup_llumnix(entrypoints_args, manager_args, instance_args, engine_args_llumnix, launch_args)
        loop = asyncio.get_event_loop()
        llumnix_client = LlumnixClientBladeLLM(engine_args, entrypoints_context, loop)
        # TODO(s5u13b): Fix it, hack to pass args to setup_llumnix_api_server.
        engine_args.llumnix_client = llumnix_client

        if engine_args.elastic_attn_cluster is not None:
            mgr = InstManager(engine_args)
            web_app = ElasticAttnEntrypoint(
                engine_args,
                inst_mgr=mgr,
            ).create_web_app()
            logger.info("Elastic-attentioon instance entrypoint ready at {}:{}".format(engine_args.host, engine_args.port))
        else:
            llm_client = init_engine_and_client(engine_args, loop)
            # start entrypoint server
            # pylint: disable=invalid-name
            entrypoint_cls = Entrypoint
            if engine_args.enable_llumnix:
                # pylint: disable=invalid-name
                entrypoint_cls = LlumnixEntrypoint
            web_app = entrypoint_cls(client=llm_client, args=engine_args).create_web_app()
            logger.info("Entrypoint API ready at {}:{}".format(engine_args.host, engine_args.port))
        web.run_app(web_app, host=engine_args.host, port=engine_args.port, loop=loop, handle_signals=False)
