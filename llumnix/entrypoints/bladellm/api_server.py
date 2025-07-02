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

from aiohttp import web

from blade_llm.service.args import ServingArgs
from blade_llm.service.server import Entrypoint


from llumnix.config import get_llumnix_config
from llumnix.utils import BackendType, LaunchMode
from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
from llumnix.entrypoints.utils import is_gpu_available
from llumnix.entrypoints.bladellm.arg_utils import BladeLLMEngineArgs, add_cli_args, get_args
from llumnix.logging.logger import init_logger
from llumnix.metrics.timestamps import set_timestamp

logger = init_logger(__name__)

llumnix_client: LlumnixClientBladeLLM = None


class LlumnixEntrypoint(Entrypoint):
    async def generate_benchmark(self, request: web.Request):
        assert isinstance(self._client, LlumnixClientBladeLLM)
        oai_req, server_req = await self._web_request_to_oai_chat_request(request)
        start = time.time()
        results_generator = await self._client.add_request(server_req)

        # Non-streaming case
        tokens = []
        per_token_latency = []
        per_token_latency_breakdown_list = []
        output_streamer = results_generator.async_stream()
        timestamps_streamer = self._client.get_request_timestamps_generator(server_req.id)
        if llumnix_client.log_request_timestamps:
            assert timestamps_streamer, "timestamps_streamer is not available."

        async for r in output_streamer:
            token_timestamps = None
            if llumnix_client.log_request_timestamps:
                try:
                    token_timestamps = timestamps_streamer.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            now = time.time()
            per_token_latency.append([now, (now - start)*1000])
            start = now
            tokens.extend(r.tokens)
            assert r.error_info is None, f"Some errors occur, benchmark is stopping: {r.error_info}."

            set_timestamp(token_timestamps, 'api_server_generate_timestamp_end', now)
            if token_timestamps:
                per_token_latency_breakdown_list.append(token_timestamps.to_latency_breakdown_dict())

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
        if per_token_latency_breakdown_list:
            ret['per_token_latency_breakdown_list'] = per_token_latency_breakdown_list

        return web.json_response(data=ret)

    # pylint: disable=unused-argument
    async def is_ready(self, request):
        response = await self._client.is_ready()
        return web.json_response(text=str(response))

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
    llumnix_client.cleanup()

# TODO(KuilongCui): launch api server in ray actor style to keep consistency with global launch mode
def setup_llumnix_api_server(engine_args: ServingArgs, loop: asyncio.AbstractEventLoop):
    # generate llumnix_parser for checking parameters with choices
    parser = LlumnixArgumentParser()
    parser = add_cli_args(parser, add_engine_args=False)
    # llumnix_opts is used to receive config options
    llumnix_config = get_llumnix_config(engine_args.llumnix_config, opts=engine_args.llumnix_opts)

    entrypoints_args, manager_args, instance_args, engine_args = get_args(llumnix_config, LaunchMode.LOCAL, parser, engine_args=engine_args)
    launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=BackendType.BLADELLM)

    setup_ray_cluster(entrypoints_args)

    # If gpu is not available, it means that this node is head pod without any llumnix components.
    if is_gpu_available():
        bladellm_engine_args = BladeLLMEngineArgs(engine_args)
        global llumnix_client
        entrypoints_context = setup_llumnix(entrypoints_args, manager_args, instance_args, bladellm_engine_args, launch_args)
        llumnix_client = LlumnixClientBladeLLM(engine_args, entrypoints_context, loop)

    return llumnix_client
