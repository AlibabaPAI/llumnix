import time
from unittest.mock import MagicMock

from vllm.core.policy import PolicyFactory

from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler
from llumnix.llumlet.request import LlumnixRequest, RequestInferenceType, RequestStatus

class MockRequest(LlumnixRequest):
    def __init__(self, request_id, length, arrival_time=None) -> None:
        super().__init__(request_id=request_id, server_info=None)
        self.length = length
        self._inference_type = RequestInferenceType.DECODE
        self._finished = False
        self.last_preemption_time = -1
        self.try_schedule_times = 1
        self.metrics = MagicMock()
        self._arrival_time = arrival_time
        self.metrics.arrival_time = arrival_time

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
    def finished(self) -> bool:
        return self._finished

    @property
    def arrival_time(self) -> float:
        pass

    @property
    def status(self) -> RequestStatus:
        pass

    @property
    def prefill_num_blocks(self) -> int:
        pass

class MockeEngine():
    def __init__(self) -> None:
        self.running = []
        self.waiting = []

    def add_request_running(self, request_id, length) -> None:
        self.running.append(MockRequest(request_id, length))

    def add_request_waiting(self, request_id, length, arrival_time) -> None:
        self.waiting.append(MockRequest(request_id, length, arrival_time))
        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        self.waiting = fcfs_policy.sort_by_priority(time.time(), self.waiting)

    def get_running_queue(self):
        return self.running

    def get_waiting_queue(self):
        return self.waiting

def test_scheduler_policy():
    engine = MockeEngine()
    scheduler = LocalMigrationScheduler("", engine)

    engine.add_request_running(request_id="0", length=1)
    engine.add_request_running(request_id="1", length=3)
    engine.add_request_running(request_id="2", length=2)

    scheduler.request_migration_policy = "LCFS"
    assert scheduler.get_migrate_out_requests()[0].request_id == "2"
    scheduler.request_migration_policy = "LRF"
    assert scheduler.get_migrate_out_requests()[0].request_id == "1"
    scheduler.request_migration_policy = "SRF"
    assert scheduler.get_migrate_out_requests()[0].request_id == "0"

    t1 = time.time()
    t2 = time.time()
    t3 = time.time()
    engine.add_request_waiting(request_id="3", length=1, arrival_time=t2)
    engine.add_request_waiting(request_id="4", length=3, arrival_time=t3)
    engine.add_request_waiting(request_id="5", length=2, arrival_time=t1)
    scheduler.request_migration_policy = "EWF"
    assert scheduler.get_migrate_out_requests()[0].request_id == "5"

def test_scheduler_should_abort_migration():
    req_0 = MockRequest(request_id="0", length=1)
    req_0.stage_timestamps = [1]
    assert req_0.should_abort_migration() is False
    req_0._finished = True
    assert req_0.should_abort_migration() is True
    req_0._finished = False
    assert req_0.should_abort_migration() is False
    req_0.last_preemption_time = 2
    assert req_0.should_abort_migration() is True
