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

from typing import List, Optional
import numpy as np

from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType
from llumnix.backends.backend_interface import BackendInterface

class LocalMigrationScheduler:
    def __init__(self, request_migration_policy: str, backend_engine: BackendInterface) -> None:
        self.request_migration_policy = request_migration_policy
        self.backend_engine = backend_engine

    def get_migrate_out_request(self, min_request_len=0, max_request_len=np.inf) -> Optional[LlumnixRequest]:
         # Requests meet the strict pre-migration always have higher prioirity than other migration policy.
        migrate_out_request = self.get_ready_migration_request(min_request_len, max_request_len)
        if migrate_out_request is None:
            if self.request_migration_policy == 'LCFS':
                migrate_out_request = self.get_last_running_request(min_request_len, max_request_len)
            elif self.request_migration_policy == 'LJF':
                migrate_out_request = self.get_longest_running_request(min_request_len, max_request_len)
            elif self.request_migration_policy == 'SJF':
                migrate_out_request = self.get_shortest_running_request(min_request_len, max_request_len)
        return migrate_out_request

    # The function is used to retrieve requests on the backend that have already met the expected_steps.
    # TODO(xinyi): Currently, the function is only used for Prefill-decoding disaggregation,
    # and only selects request that migrates from the prefill instance to the decoding instance.
    def get_ready_migration_request(self, min_request_len, max_request_len):
        running: List[LlumnixRequest] = self.backend_engine.get_running_queue()
        target_request: LlumnixRequest = None
        for request in reversed(running):
            if request.migrating:
                continue

            if request.output_len >= request.expected_steps \
                and request.inference_type == RequestInferenceType.DECODE \
                and min_request_len <= request.request_len <= max_request_len:
                target_request = request
                break

        return target_request

    def get_last_running_request(self, min_request_len, max_request_len):
        running: List[LlumnixRequest] = self.backend_engine.get_running_queue()
        target_request: LlumnixRequest = None

        for request in reversed(running):
            if request.migrating:
                continue

            if request.inference_type == RequestInferenceType.DECODE \
                and min_request_len <= request.request_len <= max_request_len:
                target_request=request
                break

        return target_request

    def get_longest_running_request(self, min_request_len, max_request_len):
        running: List[LlumnixRequest] = self.backend_engine.get_running_queue()
        condition = lambda request : request.inference_type == RequestInferenceType.DECODE \
                                        and min_request_len <= request.request_len <= max_request_len \
                                        and (not request.migrating)

        longest_seq_group = max((request for request in running if condition(request)), \
                                key=lambda request: request.request_len, default=None)

        return longest_seq_group

    def get_shortest_running_request(self, min_request_len, max_request_len):
        running: List[LlumnixRequest] = self.backend_engine.get_running_queue()
        condition = lambda request : request.inference_type == RequestInferenceType.DECODE \
                                         and min_request_len <= request.request_len <= max_request_len \
                                         and (not request.migrating)

        shortest_seq_group = min((request for request in running if condition(request)), \
                                key=lambda request: request.request_len, default=None)

        return shortest_seq_group
