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

class RequestStatus(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"

class LlumnixRequest:
    def __init__(self, request_id: str, server_info: ServerInfo) -> None:
        self.request_id = request_id
        self.server_info = server_info

        # migration args
        self.last_preemption_time = None
        self.stage_timestamps = []
        self.stage_num_blocks_list = []
        self.waiting_migrating = False
        # end-of-migration
        self.eom = False

    def reset_migration_args(self):
        self.last_preemption_time = None
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

    @property
    def finished(self) -> bool:
        raise NotImplementedError

    @property
    def arrival_time(self) -> float:
        raise NotImplementedError

    @property
    def status(self) -> RequestStatus:
        raise NotImplementedError

    @property
    def prefill_num_blocks(self) -> int:
        raise NotImplementedError

    def should_abort_migration(self) -> bool:
        return self.finished \
            or (self.last_preemption_time is not None and self.last_preemption_time > self.stage_timestamps[-1])
