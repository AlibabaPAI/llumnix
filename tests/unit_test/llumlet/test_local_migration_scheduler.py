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
from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus


class MockRequest(LlumnixRequest):
    def __init__(self, request_id, length, expected_steps, status=RequestStatus.RUNNING) -> None:
        super().__init__(request_id=request_id, request_processing_context=None, expected_steps=expected_steps)
        self.length = length
        self._llumnix_status = status
        self._inference_type = RequestInferenceType.DECODE
        self._finished = False
        self.try_schedule_times = 0
        self.eom = False

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def inference_type(self) -> RequestInferenceType:
        return self._inference_type

    @property
    def request_len(self) -> int:
        return self.length

    @property
    def prompt_len(self) -> int:
        return self.length

    @property
    def output_len(self) -> int:
        return self.length

    @property
    def request_arrival_time(self) -> float:
        pass

    @property
    def llumnix_status(self) -> RequestStatus:
        return self._llumnix_status

    @property
    def prefill_num_blocks(self) -> int:
        pass

    @property
    def n_blocks(self) -> int:
        pass

    @property
    def token_ids(self) -> int:
        pass

    @property
    def block_size(self) -> int:
        pass


class MockeEngine():
    def __init__(self) -> None:
        self.running = []
        self.waiting = []

    def add_request(self, request_id, length, expected_steps) -> None:
        self.running.append(MockRequest(request_id, length, expected_steps))

    def add_request_waiting(self, request_id, length, expected_steps) -> None:
        request = MockRequest(request_id, length, expected_steps, status=RequestStatus.WAITING)
        request.try_schedule_times += 1
        self.waiting.append(request)

    def get_running_queue(self):
        return self.running

    def get_waiting_queue(self):
        return self.waiting


def test_scheduler_policy():
    engine = MockeEngine()
    scheduler = LocalMigrationScheduler("", engine)

    engine.add_request(request_id="0", length=1, expected_steps=math.inf)
    engine.add_request(request_id="1", length=3, expected_steps=math.inf)
    engine.add_request(request_id="2", length=2, expected_steps=math.inf)
    engine.add_request_waiting(request_id="3", length=2, expected_steps=math.inf)
    engine.add_request_waiting(request_id="4", length=2, expected_steps=math.inf)

    scheduler.request_migration_policy = "LCR"
    assert scheduler.get_migrate_out_requests()[0].request_id == "2"
    scheduler.request_migration_policy = "LR"
    assert scheduler.get_migrate_out_requests()[0].request_id == "1"
    scheduler.request_migration_policy = "SR"
    assert scheduler.get_migrate_out_requests()[0].request_id == "0"
    scheduler.request_migration_policy = "FCW"
    assert scheduler.get_migrate_out_requests()[0].request_id == "3"
    scheduler.request_migration_policy = "FCWSR"
    assert scheduler.get_migrate_out_requests()[0].request_id == "3"
    assert scheduler.get_migrate_out_requests()[1].request_id == "0"

    engine.add_request(request_id="5", length=2, expected_steps=1)
    request = scheduler.get_migrate_out_requests()[0]
    assert request.request_id == "5"
    assert request.output_len >= request.expected_steps and request.inference_type == RequestInferenceType.DECODE
    engine.add_request(request_id="6", length=3, expected_steps=math.inf)
    scheduler.request_migration_policy = "LCR"
    request = scheduler.get_migrate_out_requests()[0]
    assert request.request_id == "5"
    assert request.output_len >= request.expected_steps and request.inference_type == RequestInferenceType.DECODE

def test_scheduler_should_abort_migration():
    req_0 = MockRequest(request_id="0", length=1, expected_steps=math.inf)
    req_0.stage_timestamps = [1]
    assert req_0.should_abort_migration() is False
    req_0.last_preemption_time = 2
    assert req_0.should_abort_migration() is True
    req_0.last_preemption_time = None
    req_0._finished = True
    assert req_0.should_abort_migration() is True

def test_blocking_migration():
    req_0 = MockRequest(request_id="0", length=1, expected_steps=math.inf)
    assert req_0.blocking_migration is False
    req_1 = MockRequest(request_id="1", length=2, expected_steps=1)
    assert req_1.blocking_migration is True
    req_2 = MockRequest(request_id="2", length=1, expected_steps=1)
    assert req_2.blocking_migration is True
