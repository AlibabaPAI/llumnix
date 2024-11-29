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

import json

from typing import Any, List, Tuple, Union
from pydantic import BaseModel, Field
from blade_llm.service.scheduler_types import GenerationGroupState
from blade_llm.protocol import ServerRequest, GenerateStreamResponse
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType
from llumnix.server_info import ServerInfo

from loguru import logger

class GenerateStreamResponseLlumnix(GenerateStreamResponse):
    request_id: str = Field(default="", description="Request ID associated with the request")
    server_info: Any = Field(default=None, description="Server info associated with the response")

    def __init__(self, resp: GenerateStreamResponse, request_id: str = None, server_info: ServerInfo = None) -> None:
        super().__init__(**resp.model_dump())
        self.request_id = request_id
        self.server_info = server_info

class GenerationGroupStateLlumnix(GenerationGroupState, LlumnixRequest):
    def __init__(self, gen_group: GenerationGroupState, *args) -> None:
        GenerationGroupState.__init__(self, **gen_group.__dict__)
        LlumnixRequest.__init__(self, *args)
        #TODO[xinyi]: pagedreqstate prefill
        self.is_prefill = True
        self.is_finished = False
        # The total chunk size (number of tokens) to process for next iteration.
        # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
        # chunked, it can be smaller than that.
        self.token_chunk_size = 0
        self._num_computed_tokens = 0
    
    def is_finished(self) -> bool:
        return self.is_finished
    
    @property
    def prompt_len(self) -> int:
        return len(self.paged_reqs[0].req_proto.prompt_tokens)
    
    @property
    def request_len(self) -> int:
        return len(self.paged_reqs[0].token_ids)

    @property
    def output_len(self) -> int:
        return self.request_len - self.prompt_len

    @property
    def inference_type(self) -> bool:
        if self.is_prefill:
            return RequestInferenceType.PREFILL
        return RequestInferenceType.DECODE
    
    def get_num_computed_tokens(self) -> int:
        """Return the number of prefill tokens that are already computed."""
        return self._num_computed_tokens
    
    def get_num_new_tokens(self) -> int:
        """Get the number of new tokens to be computed.

        Returns:
            The new number of tokens to be computed. I.e., 1 for decode, or
            the remaining prompt size for prefill.
        """
        if not self.is_prefill:
            return 1
        return self.get_num_uncomputed_tokens()
    
    def update_num_computed_tokens(self, num_new_computed_tokens):
        """Update number of tokens computed so far."""
        self._num_computed_tokens += num_new_computed_tokens
        # If all tokens are computed, it means it is in decoding phase.
        if self.get_num_uncomputed_tokens() == 0:
            self.is_prefill = False
    
    def get_num_uncomputed_tokens(self) -> int:
        """Return the number of prefill tokens that are not computed."""
        return self.request_len - self.get_num_computed_tokens()

class ServerRequestLlumnix(ServerRequest):
    request_id: str = Field(default="", description="Request ID associated with the request")
    server_info: Any = Field(default=None, description="Server info associated with the response")
    expected_steps: int = Field(default=-1, description="Expected number of steps for the request")

    def __init__(self, server_request: str, request_id, server_info, expected_steps) -> None:
        super().__init__(**json.loads(server_request))
        self.request_id = request_id
        self.server_info = server_info
        self.expected_steps = expected_steps
