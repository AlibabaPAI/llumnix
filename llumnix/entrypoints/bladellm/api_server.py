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
import uuid


from aiohttp import web
from aiohttp_sse import sse_response


from blade_llm.service.args import ServingArgs
from blade_llm.service.server import Entrypoint,SSEResponse
from blade_llm.service.error_handler import handle_http_error
from blade_llm.protocol import (
    OAICompletionsChoice,
    OAICompletionsRequest,
    OAILogprobs,
    OAICompletionsResponse,
    OAIChatCompletionsResponse,
    OAIChatCompletionsChoice
)
from blade_llm.service.otel_provider import extract_trace_headers


from llumnix.config import get_llumnix_config
from llumnix.utils import BackendType, LaunchMode
from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
from llumnix.entrypoints.utils import is_gpu_available
from llumnix.entrypoints.bladellm.arg_utils import BladeLLMEngineArgs, add_cli_args, get_args
from llumnix.logging.logger import init_logger
from llumnix.metrics.timestamps import set_timestamp
from llumnix.backends.bladellm.protocol import (
    LlumnixOAICompletionsResponse,
    LlumnixServerRequest,
    LlumnixGenerateStreamResponse,
    LlumnixOAIChatCompletionsResponse,
)
from llumnix.constants import LLUMNIX_DEBUG_MODE_HEADER

logger = init_logger(__name__)

llumnix_client: LlumnixClientBladeLLM = None


class LlumnixEntrypoint(Entrypoint):

    @handle_http_error
    async def oai_chat_completions(self, request: web.Request):
        # TODO(litan.ls): configurable request id header key
        oai_req, server_req = await self._web_request_to_oai_chat_request(request)
        if llumnix_client.enable_debug_mode and request.headers.get(LLUMNIX_DEBUG_MODE_HEADER, False):
            # collect and return request lantencys
            server_req = LlumnixServerRequest.from_server_request(server_req, True)
        model_name = oai_req.model or ''

        result = await self._client.add_request(server_req)
        if oai_req.stream:
            async with sse_response(request, response_cls=SSEResponse) as sse:
                connection_alive = True
                first_response = True
                streamer = result.async_stream()
                async for stream_resp in streamer:
                    try:
                        resp_cls = (
                            LlumnixOAIChatCompletionsResponse
                            if isinstance(
                                stream_resp, LlumnixGenerateStreamResponse
                            )
                            else OAIChatCompletionsResponse
                        )
                        await sse.send(
                            resp_cls.from_gen_response(
                                server_req.external_id,
                                stream_resp,
                                "chat.completion.chunk",
                                first_response,
                                model=model_name,
                            ).model_dump_json(by_alias=True)
                        )
                        first_response = False
                    except (asyncio.CancelledError, ConnectionResetError):
                        connection_alive = False
                        logger.info('Streaming cancelled or connection reset.')
                        logger.error('Request {} with response: {}', server_req, stream_resp)
                        await self._client.drop_request(server_req.id)
                        break
                if connection_alive:
                    await sse.send('[DONE]')
                    await sse.write_eof()

        else:
            if oai_req.n > 1:
                # TODO(litan.ls): support multiple generation
                return web.Response(text='Do no support to generate multiple completions.', status=500)
            # pylint: enable=no-else-return
            else:
                tokens = []
                llumnix_debug_infos = []
                streamer = result.async_stream()
                streamed_responses = [r async for r in streamer]
                for r in streamed_responses:
                    tokens.extend(r.tokens)
                    if isinstance(r, LlumnixGenerateStreamResponse):
                        llumnix_debug_info = r.llumnix_debug_info
                        llumnix_debug_info.calc_latency()
                        llumnix_debug_infos.append(llumnix_debug_info)
                finish_reason = (
                    streamed_responses[-1].detail.finish_reason.to_oai() if streamed_responses[-1].detail else ''
                )
                token_usage = streamed_responses[-1].usage
                response_cls = LlumnixOAIChatCompletionsResponse if llumnix_debug_infos else OAIChatCompletionsResponse
                response = response_cls(
                    id=server_req.external_id,
                    model=model_name,
                    choices=[
                        OAIChatCompletionsChoice(
                            finish_reason=finish_reason,
                            index=0,
                            message={
                                "role": "assistant",
                                "content": ''.join(tok.text for tok in tokens),
                            },
                            logprobs=(
                                OAILogprobs(content=tokens)
                                if tokens[0].logprob is not None or tokens[0].top_logprobs is not None
                                else None
                            ),
                        )
                    ],
                    object="chat.completion",
                    usage=token_usage,
                )
                if llumnix_debug_infos:
                    response.llumnix_debug_info = llumnix_debug_infos
            return web.json_response(text=response.model_dump_json(by_alias=True))

    @handle_http_error
    async def oai_completions(self, request: web.Request):
        assert isinstance(self._client, LlumnixClientBladeLLM)
        # TODO(litan.ls): configurable request id header key
        external_request_id = request.headers.get('X-DashScope-RequestId') or str(uuid.uuid4())
        decode_inst_str = request.headers.get('X-Decode-Instance') or ''
        decode_instances = [part.strip() for part in decode_inst_str.split(",") if part.strip()]
        trace_headers = extract_trace_headers(request.headers)
        internal_request_id = next(self._counter)
        payload_json = await request.json()
        oai_req = self.request_parser.parse(payload_json, OAICompletionsRequest, self._generation_conf_processor)
        model_name = oai_req.model or ''
        server_req = oai_req.to_server_request(
            internal_request_id, external_request_id, decode_instances, trace_headers
        )
        server_req.arrive_time = time.time()

        # process llumnix header
        if llumnix_client.enable_debug_mode and request.headers.get(LLUMNIX_DEBUG_MODE_HEADER, False):
            # collect and return request lantencys
            server_req: LlumnixServerRequest = LlumnixServerRequest.from_server_request(server_req, True)

        result = await self._client.add_request(server_req)

        if oai_req.stream:
            async with sse_response(request, response_cls=SSEResponse) as sse:
                connection_alive = True
                streamer = result.async_stream()
                async for stream_resp in streamer:
                    try:
                        response_cls = (
                            LlumnixOAICompletionsResponse
                            if isinstance(stream_resp, LlumnixGenerateStreamResponse)
                            else OAICompletionsResponse
                        )
                        await sse.send(
                            response_cls.from_gen_response(
                                external_request_id, stream_resp, model=model_name
                            ).model_dump_json(by_alias=True)
                        )
                    except (asyncio.CancelledError, ConnectionResetError):
                        connection_alive = False
                        logger.info('Streaming cancelled or connection reset.')
                        logger.error('Request {} with response: {}', server_req, stream_resp)
                        await self._client.drop_request(internal_request_id)
                        break
                if connection_alive:
                    await sse.send('[DONE]')
                    await sse.write_eof()
        # pylint: enable=no-else-return
        else:
            if oai_req.n > 1:
                # TODO(litan.ls): support multiple generation
                return web.Response(text='Do no support to generate multiple completions.', status=500)
            else:
                tokens = []
                llumnix_debug_infos = []
                streamer = result.async_stream()
                streamed_responses = [r async for r in streamer]
                for r in streamed_responses:
                    tokens.extend(r.tokens)
                    if isinstance(r, LlumnixGenerateStreamResponse):
                        llumnix_debug_info = r.llumnix_debug_info
                        llumnix_debug_info.calc_latency()
                        llumnix_debug_infos.append(llumnix_debug_info)
                finish_reason = (
                    streamed_responses[-1].detail.finish_reason.to_oai() if streamed_responses[-1].detail else ''
                )
                token_usage = streamed_responses[-1].usage
                response_cls = LlumnixOAICompletionsResponse if llumnix_debug_infos else OAICompletionsResponse
                response = response_cls(
                    id=external_request_id,
                    model=model_name,
                    choices=[
                        OAICompletionsChoice(
                            finish_reason=finish_reason,
                            index=0,
                            text=''.join(tok.text for tok in tokens),
                            logprobs=(
                                OAILogprobs(content=tokens)
                                if tokens[0].logprob is not None or tokens[0].top_logprobs is not None
                                else None
                            ),
                        )
                    ],
                    usage=token_usage,
                    llumnix_debug_info=None
                )
                if llumnix_debug_infos:
                    response.llumnix_debug_info = llumnix_debug_infos
            return web.json_response(text=response.model_dump_json(by_alias=True))

    async def generate_benchmark(self, request: web.Request):
        assert isinstance(self._client, LlumnixClientBladeLLM)
        oai_req, server_req = await self._web_request_to_oai_chat_request(request)
        if llumnix_client.enable_debug_mode and request.headers.get(LLUMNIX_DEBUG_MODE_HEADER, False):
            # collect and return request lantencys
            server_req = LlumnixServerRequest.from_server_request(server_req, True)
        start = time.time()
        results_generator = await self._client.add_request(server_req)

        # Non-streaming case
        tokens = []
        per_token_latency = []
        per_token_latency_breakdown_list = []
        output_streamer = results_generator.async_stream()

        async for r in output_streamer:
            token_timestamps = (
                r.llumnix_debug_info.token_timestamps
                if isinstance(r, LlumnixGenerateStreamResponse)
                else None
            )
            now = time.time()
            per_token_latency.append([now, (now - start)*1000])
            start = now
            tokens.extend(r.tokens)
            assert r.error_info is None, f"Some errors occur, benchmark is stopping: {r.error_info}."

            if token_timestamps:
                set_timestamp(token_timestamps, 'api_server_generate_timestamp_end', now)
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
