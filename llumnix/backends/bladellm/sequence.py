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
    def __init__(self, gen_group: GenerationGroupState, llumnix_request: LlumnixRequest, *args, **kwargs) -> None:
        GenerationGroupState.__init__(self, **gen_group.__dict__)
        LlumnixRequest.__init__(self, **llumnix_request.__dict__)

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
        if self.is_prefill():
            return RequestInferenceType.PREFILL
        return RequestInferenceType.DECODE


class ServerRequestLlumnix(ServerRequest):
    def __init__(self, request_id, server_info, expected_steps: int, request: ServerRequest) -> None:
        self.server_request = ServerRequest.__init__(self, **request.__dict__)
        self.llumnix_request = LlumnixRequest.__init__(self, request_id, server_info, expected_steps)