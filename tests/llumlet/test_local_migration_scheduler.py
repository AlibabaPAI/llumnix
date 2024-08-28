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
    assert req_0.should_abort_migration() == False
    req_0.status = RequestInferenceType.PREFILL
    assert req_0.should_abort_migration() == True
    req_0.status = RequestInferenceType.DECODE
    req_0.last_preemption_time = 2
    assert req_0.should_abort_migration() == True