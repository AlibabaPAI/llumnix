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

from typing import Deque, List
import numpy as np

from llumnix.llumlet.request import LlumnixRequest, RequestStatus, RequestInferenceType
from llumnix.backends.backend_interface import BackendBaseInterface


class LocalMigrationScheduler:
    def __init__(self, request_migration_policy: str, backend_engine: BackendBaseInterface) -> None:
        self.request_migration_policy = request_migration_policy
        self.backend_engine = backend_engine

    def get_migrate_out_requests(self, min_request_len=0, max_request_len=np.inf) -> List[LlumnixRequest]:
        # Requests meet the strict pre-migration always have higher prioirity than other migration policy.
        migrate_out_requests: List[LlumnixRequest] = self.get_required_migration_request()
        if len(migrate_out_requests) == 0:
            if self.request_migration_policy == 'LCR':
                migrate_out_requests = self._get_last_running_request(min_request_len, max_request_len)
            elif self.request_migration_policy == 'LR':
                migrate_out_requests = self._get_longest_running_request(min_request_len, max_request_len)
            elif self.request_migration_policy == 'SR':
                migrate_out_requests = self._get_shortest_running_request(min_request_len, max_request_len)
            elif self.request_migration_policy == 'FCW':
                migrate_out_requests = self._get_first_waiting_request(min_request_len, max_request_len)
            elif self.request_migration_policy == 'FCWSR':
                migrate_out_requests = self._get_first_waiting_and_shortest_running_requests(min_request_len, max_request_len)
        return migrate_out_requests

    # The function is used to retrieve requests on the backend that have already met the expected_steps.
    # (xinyi): Currently, the function is only used for Prefill-decode disaggregation,
    # and only selects request that migrates from the prefill instance to the decode instance.
    def get_required_migration_request(self):
        running: List[LlumnixRequest] = self.backend_engine.get_running_queue()
        required_migration_requests = []
        for request in reversed(running):
            if request.llumnix_status == RequestStatus.RUNNING \
                and request.inference_type == RequestInferenceType.DECODE \
                and request.output_len >= request.expected_steps:
                required_migration_requests.append(request)
        return required_migration_requests

    def _filter_running_queue(self, running, min_request_len, max_request_len):
        filtered_running = [
            request for request in running \
                if request.llumnix_status == RequestStatus.RUNNING \
                    and request.inference_type == RequestInferenceType.DECODE \
                    and min_request_len < request.request_len < max_request_len \
                    and (not request.is_migrating) \
        ]
        return filtered_running

    def _filter_waiting_queue(self, waiting, min_request_len, max_request_len):
        filtered_waiting = [
            request for request in waiting \
                if request.llumnix_status == RequestStatus.WAITING \
                    and request.try_schedule_times >= 1 \
                    and min_request_len < request.request_len < max_request_len \
                    and (not request.is_migrating) \
        ]
        return filtered_waiting

    def _get_last_running_request(self, min_request_len, max_request_len):
        running: Deque[LlumnixRequest] = self.backend_engine.get_running_queue()
        filtered_running = self._filter_running_queue(running, min_request_len, max_request_len)
        return [filtered_running[-1]] if filtered_running else []

    def _get_longest_running_request(self, min_request_len, max_request_len) -> List[LlumnixRequest]:
        running: Deque[LlumnixRequest] = self.backend_engine.get_running_queue()
        filtered_running = self._filter_running_queue(running, min_request_len, max_request_len)
        longest_seq_group = max((request for request in filtered_running), \
                                 key=lambda request: request.request_len, default=None)
        return [longest_seq_group] if longest_seq_group is not None else []

    def _get_shortest_running_request(self, min_request_len, max_request_len) -> List[LlumnixRequest]:
        running: Deque[LlumnixRequest] = self.backend_engine.get_running_queue()
        filtered_running = self._filter_running_queue(running, min_request_len, max_request_len)
        shortest_seq_group = min((request for request in filtered_running), \
                                  key=lambda request: request.request_len, default=None)
        return [shortest_seq_group] if shortest_seq_group is not None else []

    def _get_first_waiting_request(self, min_request_len, max_request_len) -> List[LlumnixRequest]:
        waiting: Deque[LlumnixRequest] = self.backend_engine.get_waiting_queue()
        filtered_waiting = self._filter_waiting_queue(waiting, min_request_len, max_request_len)
        return [waiting[0]] if filtered_waiting else []

    def _get_first_waiting_and_shortest_running_requests(self, min_request_len, max_request_len) -> List[LlumnixRequest]:
        waiting_requests = self._get_first_waiting_request(min_request_len, max_request_len)
        running_requests = self._get_shortest_running_request(min_request_len, max_request_len)
        if waiting_requests:
            waiting_requests[0].eom = True
        return waiting_requests + running_requests
