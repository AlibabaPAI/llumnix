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

from llumnix.server_info import ServerInfo

class RequestInferenceType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"

class LlumnixRequest:
    def __init__(self, request_id: int, server_info: ServerInfo) -> None:
        self.request_id = request_id
        self.server_info = server_info
        self.last_preemption_time = None

        # migration args
        self.stage_timestamps = []
        self.stage_num_blocks_list = []

    def reset_migration_args(self):
        self.stage_timestamps = []
        self.stage_num_blocks_list = []

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

    def should_abort_migration(self) -> bool:
        return self.output_len == 0 \
            or (self.last_preemption_time and self.last_preemption_time > self.stage_timestamps[-1]) \
            or self.inference_type == RequestInferenceType.PREFILL
