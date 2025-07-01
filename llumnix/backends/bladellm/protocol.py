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

from typing import List, Union, Type, Optional
import msgspec
from pydantic import Field, BaseModel

from blade_llm.protocol_msgspec import ServerRequest as _ServerRequest
from blade_llm.protocol_msgspec import GenerateStreamResponse as _GenerateStreamResponse
from blade_llm.protocol import (
    OAICompletionsResponse,
    OAIChatCompletionsResponse,
)

from llumnix.entrypoints.utils import LlumnixTraceInfo


class LlumnixBaseResponse(BaseModel):
    llumnix_trace_info: Optional[Union[LlumnixTraceInfo, List[LlumnixTraceInfo]]] = (
        Field(default=None, title="Llumnix debug info.")
    )


class LlumnixGenerateStreamResponse(_GenerateStreamResponse):

    llumnix_trace_info: Optional[Union[LlumnixTraceInfo, List[LlumnixTraceInfo]]] = None

    @classmethod
    def from_generate_stream_response(
        cls: Type["LlumnixGenerateStreamResponse"],
        gen_resp: _GenerateStreamResponse,
        llumnix_trace_info: LlumnixTraceInfo = None,
    ):
        oai_data = msgspec.structs.asdict(gen_resp)
        llumnix_generate_stream_response = LlumnixGenerateStreamResponse(**oai_data)
        llumnix_generate_stream_response.llumnix_trace_info = llumnix_trace_info
        return llumnix_generate_stream_response

    def set_request_timestamp(self, request_timestamp):
        if self.llumnix_trace_info is None:
            self.llumnix_trace_info = LlumnixTraceInfo()
        self.llumnix_trace_info.token_timestamps = request_timestamp


class LlumnixOAICompletionsResponse(OAICompletionsResponse, LlumnixBaseResponse):

    # pylint: disable=redefined-builtin
    @classmethod
    def from_gen_response(
        cls: Type["LlumnixOAICompletionsResponse"],
        id: str,
        gen_resp: LlumnixGenerateStreamResponse,
        model: str = "",
        object: str = "text_completion",
    ):
        oai_completions_response: OAICompletionsResponse = (
            OAICompletionsResponse.from_gen_response(
                id=id, gen_resp=gen_resp, model=model, object=object
            )
        )
        oai_data = oai_completions_response.model_dump(by_alias=True)
        llumnix_oai_completions_response = LlumnixOAICompletionsResponse(**oai_data)
        if isinstance(gen_resp, LlumnixGenerateStreamResponse):
            gen_resp.llumnix_trace_info.calc_latency()
            llumnix_oai_completions_response.llumnix_trace_info = (
                gen_resp.llumnix_trace_info
            )
        return llumnix_oai_completions_response


class LlumnixOAIChatCompletionsResponse(
    OAIChatCompletionsResponse, LlumnixBaseResponse
):

    # pylint: disable=redefined-builtin
    @classmethod
    def from_gen_response(
        cls: Type["LlumnixOAIChatCompletionsResponse"],
        id: str,
        gen_resp: LlumnixGenerateStreamResponse,
        object: str,
        is_first: bool,
        model: str = "",
    ) -> "OAIChatCompletionsResponse":
        oai_chat_completions_response: OAIChatCompletionsResponse = (
            OAIChatCompletionsResponse.from_gen_response(
                id=id, gen_resp=gen_resp, object=object, is_first=is_first, model=model
            )
        )
        oai_data = oai_chat_completions_response.model_dump(by_alias=True)
        llumnix_oai_chat_completions_response = LlumnixOAIChatCompletionsResponse(
            **oai_data
        )
        if isinstance(gen_resp, LlumnixGenerateStreamResponse):
            gen_resp.llumnix_trace_info.calc_latency()
            llumnix_oai_chat_completions_response.llumnix_trace_info = (
                gen_resp.llumnix_trace_info
            )
        return llumnix_oai_chat_completions_response


class LlumnixServerRequest(_ServerRequest):
    llumnix_trace_request: bool = Field(default=False, title="Enable llumnix debug mode.")

    @classmethod
    def from_server_request(
        cls: Type["LlumnixServerRequest"], request: _ServerRequest, trace_request: bool
    ):
        server_request_data = msgspec.structs.asdict(request)
        llumxix_server_request = LlumnixServerRequest(**server_request_data)
        llumxix_server_request.llumnix_trace_request = trace_request
        return llumxix_server_request
