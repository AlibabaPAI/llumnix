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

import math

from blade_llm.service.scheduler_types import GenerationGroupState

from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus


class GenerationGroupStateLlumnix(GenerationGroupState, LlumnixRequest):
    def __init__(self, gen_group: GenerationGroupState, *args) -> None:
        GenerationGroupState.__init__(self, **gen_group.__dict__)
        LlumnixRequest.__init__(self, *args)
        self._inference_type = RequestInferenceType.PREFILL

    @property
    def status(self) -> RequestStatus:
        return self._status

    @property
    def inference_type(self) -> RequestInferenceType:
        return self._inference_type

    @property
    def request_len(self) -> int:
        return len(self.paged_reqs[0].token_ids)

    @property
    def prompt_len(self) -> int:
        return len(self.paged_reqs[0].req_proto.prompt_tokens)

    @property
    def output_len(self) -> int:
        return len(self.paged_reqs[0].token_ids) - len(self.paged_reqs[0].req_proto.prompt_tokens)

    @property
    def finished(self) -> bool:
        return self._status == RequestStatus.FINISHED

    @property
    def request_arrival_time(self) -> float:
        return self.receive_time

    @property
    def prefill_num_blocks(self) -> int:
        return math.ceil(len(self.paged_reqs[0].req_proto.prompt_tokens) / self.paged_reqs[0].block_size)
