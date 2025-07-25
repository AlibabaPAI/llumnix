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

from vllm.sequence import SequenceGroup, SequenceStatus

from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus


class SequenceGroupLlumnix(SequenceGroup, LlumnixRequest):
    def __init__(
        self,
        request_id,
        request_processing_context,
        expected_steps: int,
        num_hit_tokens: int,
        transfer_penalty: int,
        *args,
        **kwargs
    ) -> None:
        SequenceGroup.__init__(self, request_id, *args, **kwargs)
        LlumnixRequest.__init__(self, request_id, request_processing_context, expected_steps, num_hit_tokens, transfer_penalty)

    @property
    def block_size(self) -> int:
        return self.get_seqs()[0].block_size

    @property
    def prompt_len(self) -> int:
        return self.get_seqs()[0].get_prompt_len()

    def is_finished(self) -> bool:
        return self.get_seqs()[0].is_finished()

    @property
    def request_len(self) -> int:
        return self.get_seqs()[0].get_len()

    @property
    def output_len(self) -> int:
        return self.get_seqs()[0].get_output_len()

    @property
    def n_blocks(self) -> int:
        return self.get_seqs()[0].n_blocks
    @property
    def token_ids(self) -> int:
        return self.get_seqs()[0].get_token_ids()

    @property
    def inference_type(self) -> RequestInferenceType:
        if self.is_prefill():
            return RequestInferenceType.PREFILL
        return RequestInferenceType.DECODE

    @property
    def finished(self) -> bool:
        return self.get_seqs()[0].is_finished()

    @property
    def request_arrival_time(self) -> float:
        return self.arrival_time

    @property
    def llumnix_status(self) -> RequestStatus:
        if self._llumnix_status:
            return self._llumnix_status
        status = self.get_seqs()[0].status
        if status == SequenceStatus.RUNNING:
            request_status = RequestStatus.RUNNING
        elif status == SequenceStatus.WAITING:
            request_status = RequestStatus.WAITING
        else:
            request_status = RequestStatus.FINISHED
        return request_status

    @property
    def prefill_num_blocks(self) -> int:
        # Get the prefill len of the waiting request.
        return self.get_seqs()[0].n_blocks
