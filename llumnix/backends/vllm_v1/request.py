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

from typing import TYPE_CHECKING, Any, Optional, Union

from vllm.v1.request import Request
from vllm.v1.request import RequestStatus as RequestStatusVLLMV1
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils import is_list_of
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus


class LlumnixRequestVLLMV1(Request, LlumnixRequest):
    def __init__(self, request_id, server_info, expected_steps: int, *args, **kwargs) -> None:
        # Request.__init__(self, request_id, *args, **kwargs)
        self._init_vllm_request(request_id, *args, **kwargs)
        LlumnixRequest.__init__(self, request_id, server_info, expected_steps)

    #FIXME(zhaozhiyu): This is a temperary implementation copied from Request.__init__ to avoid multiple inheritance of self.status
    def _init_vllm_request(self, 
        request_id: str,
        prompt_token_ids: list[int],
        multi_modal_inputs: list[MultiModalKwargs] | None,
        multi_modal_hashes: list[str] | None,
        multi_modal_placeholders: list[PlaceholderRange] | None,
        sampling_params: SamplingParams,
        eos_token_id: int | None,
        client_index: int = 0,
        lora_request = None,
        structured_output_request: StructuredOutputRequest | None = None,
        cache_salt: str | None = None
    ):
        self.request_id = request_id
        self.client_index = client_index
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = structured_output_request

        # self.status = (RequestStatus.WAITING_FOR_FSM
        #                if sampling_params.guided_decoding is not None else
        #                RequestStatus.WAITING)
        self.vllm_status = (RequestStatus.WAITING_FOR_FSM
                       if sampling_params.guided_decoding is not None else
                       RequestStatus.WAITING)
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: Optional[str] = cache_salt

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: list[str] = multi_modal_hashes or []
        self.num_encoder_inputs = len(self.mm_inputs)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # P/D: Connector-specific KV transfer parameters.
        kv_params = (None if sampling_params.extra_args is None else
                     sampling_params.extra_args.get("kv_transfer_params"))
        self.kv_transfer_params: Optional[dict[str, Any]] = kv_params

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1
    
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

    @property
    def status(self) -> RequestStatus:
        if self._status:
            return self._status
        status = self.vllm_status
        if status == RequestStatusVLLMV1.RUNNING:
            request_status = RequestStatus.RUNNING
        elif status == RequestStatusVLLMV1.WAITING:
            request_status = RequestStatus.WAITING
        else:
            request_status = RequestStatus.FINISHED
        return request_status

    # @property
    # def prefill_num_blocks(self) -> int:
    #     # Get the prefill len of the waiting request.
    #     return self.get_seqs()[0].n_blocks
