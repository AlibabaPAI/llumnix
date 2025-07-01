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

import msgspec
from aiohttp import web
from aiohttp_sse import sse_response


from blade_llm.service.args import ServingArgs
from blade_llm.service.server import Entrypoint, SSEResponse
from blade_llm.service.error_handler import handle_http_error
from blade_llm.protocol import (
    Logprob,
    OAIChatCompletionsChoice,
    OAIChatCompletionsResponse,
    OAICompletionsChoice,
    OAICompletionsRequest,
    OAICompletionsResponse,
    OAILogprobs,
    Token,
    TokenUsage,
)
from blade_llm.service.otel_provider import extract_trace_headers
from blade_llm.service.request_parser import extract_kvt_meta


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

from llumnix.constants import LLUMNIX_TRACE_HEADER

logger = init_logger(__name__)

llumnix_client: LlumnixClientBladeLLM = None


class LlumnixEntrypoint(Entrypoint):

    @handle_http_error
    async def oai_chat_completions(self, request: web.Request):
        # TODO(litan.ls): configurable request id header key
        oai_req, server_req = await self._web_request_to_oai_chat_request(request)
        model_name = oai_req.model or ''

        if request.headers.get(LLUMNIX_TRACE_HEADER, "False").lower() in ('true', '1'):
            # collect and return request lantencys
            server_req = LlumnixServerRequest.from_server_request(server_req, True)

        result = await self._client.add_request(server_req)
        if oai_req.stream:
            async with sse_response(request, response_cls=SSEResponse) as sse:
                connection_alive = True
                first_response = True
                streamer = result.async_stream()
                async for stream_resp in streamer:
                    try:
                        if isinstance(stream_resp, LlumnixGenerateStreamResponse):
                            resp_cls = LlumnixOAIChatCompletionsResponse
                            set_timestamp(
                                stream_resp.llumnix_trace_info.token_timestamps,
                                "api_server_generate_timestamp_end",
                                time.perf_counter(),
                            )
                        else:
                            resp_cls = OAIChatCompletionsResponse
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
                        logger.error('Request {} with response: {}'.format(server_req, stream_resp))
                        await self._client.drop_request(server_req.id)
                        break
                if connection_alive:
                    await sse.send('[DONE]')
                    await sse.write_eof()

        else:
            # pylint: disable=no-else-return
            if oai_req.n > 1:
                # TODO(litan.ls): support multiple generation
                return web.Response(text='Do no support to generate multiple completions.', status=500)
            else:
                tokens = []
                llumnix_trace_infos = []
                streamer = result.async_stream()
                last_response = None
                async for r in streamer:
                    last_response = r
                    tokens.extend(
                        [
                            Token(
                                id=token.id,
                                text=token.text,
                                logprob=token.logprob,
                                is_special=token.is_special,
                                bytes=token.bytes,
                                top_logprobs=[
                                    Logprob(
                                        id=top_logprob.id,
                                        text=top_logprob.text,
                                        logprob=top_logprob.logprob,
                                        bytes=top_logprob.bytes,
                                    )
                                    for top_logprob in token.top_logprobs
                                ]
                                if token.top_logprobs is not None
                                else None,
                            )
                            for token in r.tokens
                        ]
                        if isinstance(r.tokens[0], msgspec.Struct)
                        else r.tokens
                    )
                    if isinstance(r, LlumnixGenerateStreamResponse):
                        llumnix_trace_info = r.llumnix_trace_info
                        set_timestamp(
                                llumnix_trace_info.token_timestamps,
                                "api_server_generate_timestamp_end",
                                time.perf_counter(),
                            )
                        llumnix_trace_info.calc_latency()
                        llumnix_trace_infos.append(llumnix_trace_info)
                finish_reason = (
                    last_response.detail.finish_reason.to_oai() if last_response.detail else ''
                )
                token_usage = last_response.usage
                response_cls = LlumnixOAIChatCompletionsResponse if llumnix_trace_infos else OAIChatCompletionsResponse
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
                    usage=TokenUsage(
                        prompt_tokens=token_usage.prompt_tokens,
                        completion_tokens=token_usage.completion_tokens,
                        total_tokens=token_usage.total_tokens,
                    ),
                )
                if llumnix_trace_infos:
                    response.llumnix_trace_info = llumnix_trace_infos
            return web.json_response(text=response.model_dump_json(by_alias=True))

    @handle_http_error
    async def oai_completions(self, request: web.Request):
        assert isinstance(self._client, LlumnixClientBladeLLM)
        # TODO(litan.ls): configurable request id header key
        external_request_id = request.headers.get('X-DashScope-RequestId') or str(uuid.uuid4())
        decode_inst_str = request.headers.get('X-Decode-Instance') or ''
        kvt_meta_info = extract_kvt_meta(request.headers)
        decode_instances = [part.strip() for part in decode_inst_str.split(",") if part.strip()]
        trace_headers = extract_trace_headers(request.headers)
        internal_request_id = next(self._counter)
        payload_json = await request.json()
        oai_req = self.request_parser.parse(payload_json, OAICompletionsRequest, self._generation_conf_processor)
        model_name = oai_req.model or ''
        server_req = oai_req.to_server_request(
            internal_request_id,
            external_request_id,
            decode_instances,
            trace_headers,
            kvt_meta_info=kvt_meta_info,
        )
        server_req.arrive_time = time.time()

        # process llumnix header
        if request.headers.get(LLUMNIX_TRACE_HEADER, "False").lower() in ('true', '1'):
            # collect and return request lantencys
            server_req: LlumnixServerRequest = LlumnixServerRequest.from_server_request(server_req, True)

        result = await self._client.add_request(server_req)
        if oai_req.stream:
            async with sse_response(request, response_cls=SSEResponse) as sse:
                connection_alive = True
                streamer = result.async_stream()
                async for stream_resp in streamer:
                    try:
                        if isinstance(stream_resp, LlumnixGenerateStreamResponse):
                            response_cls = LlumnixOAICompletionsResponse
                            set_timestamp(
                                stream_resp.llumnix_trace_info.token_timestamps,
                                "api_server_generate_timestamp_end",
                                time.perf_counter(),
                            )
                        else:
                            response_cls = OAICompletionsResponse
                        await sse.send(
                            response_cls.from_gen_response(
                                external_request_id, stream_resp, model=model_name
                            ).model_dump_json(by_alias=True)
                        )
                    except (asyncio.CancelledError, ConnectionResetError):
                        connection_alive = False
                        logger.info('Streaming cancelled or connection reset.')
                        logger.error('Request {} with response: {}'.format(server_req, stream_resp))
                        await self._client.drop_request(internal_request_id)
                        break
                if connection_alive:
                    await sse.send('[DONE]')
                    await sse.write_eof()
        # pylint: disable=no-else-return
        else:
            if oai_req.n > 1:
                # TODO(litan.ls): support multiple generation
                return web.Response(text='Do no support to generate multiple completions.', status=500)
            else:
                tokens = []
                llumnix_trace_infos = []
                streamer = result.async_stream()
                last_response = None
                async for r in streamer:
                    last_response = r
                    tokens.extend(
                        [
                            Token(
                                id=token.id,
                                text=token.text,
                                logprob=token.logprob,
                                is_special=token.is_special,
                                bytes=token.bytes,
                                top_logprobs=[
                                    Logprob(
                                        id=top_logprob.id,
                                        text=top_logprob.text,
                                        logprob=top_logprob.logprob,
                                        bytes=top_logprob.bytes,
                                    )
                                    for top_logprob in token.top_logprobs
                                ]
                                if token.top_logprobs is not None
                                else None,
                            )
                            for token in r.tokens
                        ]
                        if isinstance(r.tokens[0], msgspec.Struct)
                        else r.tokens
                    )
                    if isinstance(r, LlumnixGenerateStreamResponse):
                        llumnix_trace_info = r.llumnix_trace_info
                        set_timestamp(
                            llumnix_trace_info.token_timestamps,
                            "api_server_generate_timestamp_end",
                            time.perf_counter(),
                        )
                        llumnix_trace_info.calc_latency()
                        llumnix_trace_infos.append(llumnix_trace_info)
                finish_reason = (
                    last_response.detail.finish_reason.to_oai() if last_response.detail else ''
                )
                token_usage = last_response.usage
                response_cls = LlumnixOAICompletionsResponse if llumnix_trace_infos else OAICompletionsResponse
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
                    usage=TokenUsage(
                        prompt_tokens=token_usage.prompt_tokens,
                        completion_tokens=token_usage.completion_tokens,
                        total_tokens=token_usage.total_tokens,
                    ),
                )
                if llumnix_trace_infos:
                    response.llumnix_trace_info = llumnix_trace_infos
            return web.json_response(text=response.model_dump_json(by_alias=True))

    async def generate_benchmark(self, request: web.Request):
        assert isinstance(self._client, LlumnixClientBladeLLM)
        oai_req, server_req = await self._web_request_to_oai_chat_request(request)
        if request.headers.get(LLUMNIX_TRACE_HEADER, "False").lower() in ('true', '1'):
            # collect and return request lantencys
            server_req = LlumnixServerRequest.from_server_request(server_req, True)
        start = time.perf_counter()
        results_generator = await self._client.add_request(server_req)

        # Non-streaming case
        tokens = []
        per_token_latency = []
        per_token_latency_breakdown_list = []
        output_streamer = results_generator.async_stream()

        async for r in output_streamer:
            token_timestamps = (
                r.llumnix_trace_info.token_timestamps
                if isinstance(r, LlumnixGenerateStreamResponse)
                else None
            )
            now = time.perf_counter()
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
