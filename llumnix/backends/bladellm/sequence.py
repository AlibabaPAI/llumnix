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

from blade_llm.service.scheduler_types import GenerationGroupState
from blade_llm.protocol import ServerRequest
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType


class GenerationGroupStateLlumnix(GenerationGroupState, LlumnixRequest):
    def __init__(self, gen_group: GenerationGroupState, llumnix_request_args) -> None:
        GenerationGroupState.__init__(self, **gen_group.__dict__)
        LlumnixRequest.__init__(self, *llumnix_request_args)
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
        # we use `get_len()` which includes prompt_len + output_len instead
        # of prompt_len here. This is because during recompute we need to
        # prefill for both prompt and output.
        return self.request_len - self.get_num_computed_tokens()

class ServerRequestLlumnix:
    def __init__(self, *args, server_request: ServerRequest) -> None:
        self.server_request = server_request
        self.llumnix_request_args = args