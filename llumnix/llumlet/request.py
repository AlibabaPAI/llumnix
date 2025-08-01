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

from enum import Enum
import math

from llumnix.request_processing_context import RequestProcessingContext
from llumnix.utils import RequestIDType


class RequestInferenceType(str, Enum):
    UNKNOWN = "unknown"
    PREFILL = "prefill"
    DECODE = "decode"
    PREFILL_AND_DECODE = "prefill_and_decode"

    @classmethod
    def generate_inference_type(cls, exist_prefill: bool, exist_decode: bool):
        if exist_prefill and exist_decode:
            inference_type = RequestInferenceType.PREFILL_AND_DECODE
        elif exist_prefill:
            inference_type = RequestInferenceType.PREFILL
        elif exist_decode:
            inference_type = RequestInferenceType.DECODE
        else:
            inference_type = RequestInferenceType.UNKNOWN

        return RequestInferenceType(inference_type)


class RequestStatus(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"
    FINISHED = "finished"
    RUNNING_MIGRATING = "running_migrating"
    WAITING_MIGRATING = "waiting_migrating"

    @staticmethod
    def is_migrating(status) -> bool:
        return status in [RequestStatus.RUNNING_MIGRATING, RequestStatus.WAITING_MIGRATING]


class LlumnixRequest:
    def __init__(
        self,
        request_id: RequestIDType,
        request_processing_context: RequestProcessingContext,
        expected_steps: int = math.inf,
        num_hit_tokens: int = 0,
        transfer_penalty: int = 1
    ) -> None:
        self.request_id = request_id
        self.request_processing_context: RequestProcessingContext = request_processing_context

        # strict pre-migration args
        self.expected_steps = expected_steps

        self.num_hit_tokens = num_hit_tokens
        self.transfer_penalty = transfer_penalty

        # migration states
        self.last_preemption_time = None
        self.stage_timestamps = []
        self.stage_num_blocks_list = []
        self.try_schedule_times = 0
        self._llumnix_status = None
        self.migration_start_time = None
        self.is_migrating = False

        # end-of-migration, for multiple requests migration
        self.eom = False

    def reset_migration_states_dst(self):
        # By default, there is no limit on the number of steps expected for the request.
        self.expected_steps = math.inf

        self.last_preemption_time = None
        self.stage_timestamps = []
        self.stage_num_blocks_list = []
        self.try_schedule_times = 0
        self.migration_start_time = None
        self.is_migrating = False

    def reset_migration_states_src(self):
        self.last_preemption_time = None
        self.stage_timestamps = []
        self.stage_num_blocks_list = []
        self.migration_start_time = None
        self.is_migrating = False

    def reset_llumnix_status(self):
        self._llumnix_status = None

    def set_llumnix_status(self, status: RequestStatus):
        self._llumnix_status = status

    @property
    def inference_type(self) -> RequestInferenceType:
        raise NotImplementedError

    @property
    def request_len(self) -> int:
        raise NotImplementedError

    @property
    def prompt_len(self) -> int:
        raise NotImplementedError

    @property
    def output_len(self) -> int:
        raise NotImplementedError

    @property
    def finished(self) -> bool:
        raise NotImplementedError

    @property
    def request_arrival_time(self) -> float:
        raise NotImplementedError

    @property
    def llumnix_status(self) -> RequestStatus:
        raise NotImplementedError

    @property
    def prefill_num_blocks(self) -> int:
        raise NotImplementedError

    # Whether the migration of request is completed within one stage. For requests that have already reached
    # the expected steps, blocking_migration is True.
    @property
    def blocking_migration(self) -> bool:
        return self.output_len >= self.expected_steps

    def should_abort_migration(self) -> bool:
        begin_time = self.stage_timestamps[-1] if len(self.stage_timestamps) > 0 else self.migration_start_time
        preempted = self.last_preemption_time is not None and self.last_preemption_time > begin_time

        return self.finished or preempted
