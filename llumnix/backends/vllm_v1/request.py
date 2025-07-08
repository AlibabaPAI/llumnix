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

from vllm.v1.request import Request
from vllm.v1.request import RequestStatus as RequestStatusVLLMV1

from llumnix.llumlet.request import LlumnixRequest, RequestStatus, RequestInferenceType


class LlumnixRequestVLLMV1(Request, LlumnixRequest):
    def __init__(self, request_id, server_info, expected_steps: int, *args, **kwargs) -> None:
        Request.__init__(self, request_id, *args, **kwargs)
        LlumnixRequest.__init__(self, request_id, server_info, expected_steps)

    @property
    def request_len(self) -> int:
        raise NotImplementedError

    @property
    def inference_type(self) -> RequestInferenceType:
        return RequestInferenceType.UNKNOWN

    @property
    def prompt_len(self) -> int:
        return self.num_prompt_tokens

    @property
    def output_len(self) -> int:
        return self.num_computed_tokens

    @property
    def finished(self) -> bool:
        return self.is_finished()

    @property
    def request_arrival_time(self) -> float:
        raise NotImplementedError

    @property
    def llumnix_status(self) -> RequestStatus:
        if self._llumnix_status:
            return self._llumnix_status
        status = self.llumnix_status
        if status == RequestStatusVLLMV1.RUNNING:
            request_status = RequestStatus.RUNNING
        elif status == RequestStatusVLLMV1.WAITING:
            request_status = RequestStatus.WAITING
        else:
            request_status = RequestStatus.FINISHED
        return request_status

    @property
    def prefill_num_blocks(self) -> int:
        raise NotImplementedError
