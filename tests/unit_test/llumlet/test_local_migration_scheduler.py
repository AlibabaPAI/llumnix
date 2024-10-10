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

from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType

class MockRequest(LlumnixRequest):
    def __init__(self, request_id, length) -> None:
        super().__init__(request_id=request_id, server_info=None)
        self.length = length
        self.status = RequestInferenceType.DECODE

    @property
    def inference_type(self) -> RequestInferenceType:
        return self.status

    @property
    def request_len(self) -> int:
        return self.length

    @property
    def prompt_len(self) -> int:
        return self.length

    @property
    def output_len(self) -> int:
        return self.length

class MockeEngine():
    def __init__(self) -> None:
        self.running = []

    def add_request(self, request_id, length) -> None:
        self.running.append(MockRequest(request_id, length))

    def get_running_queue(self):
        return self.running

def test_scheduler_policy():
    engine = MockeEngine()
    scheduler = LocalMigrationScheduler("", engine)

    engine.add_request(request_id="0", length=1)
    engine.add_request(request_id="1", length=3)
    engine.add_request(request_id="2", length=2)

    scheduler.request_migration_policy = "LCFS"
    assert scheduler.get_migrate_out_request().request_id == "2"
    scheduler.request_migration_policy = "LJF"
    assert scheduler.get_migrate_out_request().request_id == "1"
    scheduler.request_migration_policy = "SJF"
    assert scheduler.get_migrate_out_request().request_id == "0"

def test_scheduler_should_abort_migration():
    req_0 = MockRequest(request_id="0", length=1)
    req_0.stage_timestamps = [1]
    assert req_0.should_abort_migration() is False
    req_0.status = RequestInferenceType.PREFILL
    assert req_0.should_abort_migration() is True
    req_0.status = RequestInferenceType.DECODE
    req_0.last_preemption_time = 2
    assert req_0.should_abort_migration() is True
