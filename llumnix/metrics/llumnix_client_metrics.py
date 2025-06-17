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


import time
from typing import Dict
from llumnix.metrics.base_metrics import BaseMetrics
from llumnix.metrics.metrics_types import Registery, Summary, TimeAveragedCounter
from llumnix.logging.logger import init_logger
from llumnix.metrics.utils import is_metrics_enabled
from llumnix import envs as llumnix_envs

logger = init_logger(__name__)


class LlumnixClientMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.register = Registery()
        self.metrics_sampling_interval = int(
            llumnix_envs.LLUMNIX_CLIENT_METRICS_SAMPLE_EVERY_N_RECORDS
        )
        self.curr_reqeust_index = -1
        self.request_received_timestamps: Dict[str, float] = {}
        self.request_last_token_received_timestamp: Dict[str, float] = {}

        self.llumnix_client_request_qps = TimeAveragedCounter(
            name="llumnix_client_request_qps",
            registry=self.register,
            metrics_sampling_interval=self.metrics_sampling_interval,
        )

        # reqeust in llumnix client is all is_stream=True
        self.llumnix_client_ttft = Summary(
            name="llumnix_client_ttft",
            registry=self.register,
            metrics_sampling_interval=1,  # special check before observe, so use 1 here
        )

        self.llumnix_client_tpot = Summary(
            name="llumnix_client_tpot",
            registry=self.register,
            metrics_sampling_interval=1,  # special check before observe, so use 1 here
        )

        self.enable_metrics = is_metrics_enabled(
            llumnix_envs.LLUMNIX_CLIENT_METRICS_SAMPLE_EVERY_N_RECORDS
        )
        if self.enable_metrics:
            self.start_metrics_export_loop()

    def increase_request_index_and_check_need_sample(self):
        if self.metrics_sampling_interval <= 0:
            # disable meitrics
            return False
        self.curr_reqeust_index = (
            self.curr_reqeust_index + 1
        ) % self.metrics_sampling_interval
        return self.curr_reqeust_index == 0

    # record qps, ttft and tpot
    def add_request(self, reqeust_id: str):
        self.llumnix_client_request_qps.increase()
        if self.increase_request_index_and_check_need_sample():
            self.request_received_timestamps[reqeust_id] = time.time()

    def remove_request(self, request_id: str):
        if not self.request_received_timestamps.pop(request_id, None):
            logger.warning('Request id {} not in dict request_received_timestamps, skip del.'.format(request_id))
        if not self.request_last_token_received_timestamp.pop(request_id, None):
            logger.warning('Request id {} not in dict request_pre_chunk_received_timestamp, skip del.'.format(request_id))

    def observe_tpot_and_ttft(self, request_id: str):
        if request_id not in self.request_received_timestamps:
            # not sample
            return
        if request_id not in self.request_last_token_received_timestamp:
            # first chunk, record ttft
            curr_timestamp = time.time()
            self.request_last_token_received_timestamp[request_id] = curr_timestamp
            ttft_ms = (
                curr_timestamp - self.request_received_timestamps[request_id]
            ) * 1000
            self.llumnix_client_ttft.observe(value=ttft_ms)
        else:
            # not first chunk, record tpot
            curr_timestamp = time.time()
            tpot_ms = (
                curr_timestamp - self.request_last_token_received_timestamp[request_id]
            ) * 1000
            self.request_last_token_received_timestamp[request_id] = curr_timestamp
            self.llumnix_client_tpot.observe(value=tpot_ms)
