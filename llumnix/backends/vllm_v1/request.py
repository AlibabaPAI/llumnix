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

from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus


class LlumnixRequestVLLMV1(Request, LlumnixRequest):
    def __init__(self, request_id, server_info, expected_steps: int, *args, **kwargs) -> None:
        Request.__init__(self, request_id, *args, **kwargs)
        LlumnixRequest.__init__(self, request_id, server_info, expected_steps)

    # @property
    # def block_size(self) -> int:
    #     return self.

    @property
    def prompt_len(self) -> int:
        return self.num_prompt_tokens

    # @property
    # def request_len(self) -> int:
    #     return self.get_seqs()[0].get_len()

    @property
    def output_len(self) -> int:
        return self.num_computed_tokens

    # @property
    # def n_blocks(self) -> int:
    #     return self.get_seqs()[0].n_blocks
    # @property
    # def token_ids(self) -> int:
    #     return self.all_token_ids

    # @property
    # def inference_type(self) -> RequestInferenceType:
    #     if self.is_prefill():
    #         return RequestInferenceType.PREFILL
    #     return RequestInferenceType.DECODE

    @property
    def finished(self) -> bool:
        return self.is_finished()

    # @property
    # def request_arrival_time(self) -> float:
    #     return self.arrival_time

    # @property
    # def status(self) -> RequestStatus:
    #     if self._status:
    #         return self._status
    #     status = self.status
    #     if status == SequenceStatus.RUNNING:
    #         request_status = RequestStatus.RUNNING
    #     elif status == SequenceStatus.WAITING:
    #         request_status = RequestStatus.WAITING
    #     else:
    #         request_status = RequestStatus.FINISHED
    #     return request_status

    # @property
    # def prefill_num_blocks(self) -> int:
    #     # Get the prefill len of the waiting request.
    #     return self.get_seqs()[0].n_blocks
