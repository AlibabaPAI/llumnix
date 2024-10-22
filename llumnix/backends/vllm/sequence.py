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

from vllm.sequence import SequenceGroup, SequenceStatus

from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus


class SequenceGroupLlumnix(SequenceGroup, LlumnixRequest):
    def __init__(self, request_id, server_info, expected_steps: int, *args, **kwargs) -> None:
        SequenceGroup.__init__(self, request_id, *args, **kwargs)
        LlumnixRequest.__init__(self, request_id, server_info, expected_steps)

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
    def inference_type(self) -> RequestInferenceType:
        if self.is_prefill():
            return RequestInferenceType.PREFILL
        return RequestInferenceType.DECODE

    @property
    def finished(self) -> bool:
        return self.get_seqs()[0].is_finished()

    @property
    def arrival_time(self) -> float:
        return self.metrics.arrival_time

    @property
    def status(self) -> RequestStatus:
        status = self.get_seqs()[0].status
        assert status in [SequenceStatus.RUNNING, SequenceStatus.WAITING], \
            "Only RUNNING, WAITING are expected status for LlumnixRequest"
        if status == SequenceStatus.RUNNING:
            request_status = RequestStatus.RUNNING
        else:
            request_status = RequestStatus.WAITING
        return request_status

    @property
    def prefill_num_blocks(self) -> int:
        # Get the prefill len of the waiting request.
        return math.ceil(self.request_len / self.get_seqs()[0].block_size)
