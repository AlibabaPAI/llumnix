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

import argparse
import time
import threading
import ray
from ray.util.queue import Queue as RayQueue

from llumnix.queue.utils import init_request_output_queue_client, QueueType
from llumnix.ray_utils import get_manager_name

from tests.unit_test.entrypoints.vllm.api_server import (MockManager, setup_entrypoints_context,
                                                         run_uvicorn_server)

ENTRYPOINTS_ACTOR_NAME = "entrypoints"


class MockManagerServer(MockManager):
    def __init__(self, entrypoints_args):
        self._num_generates = 0
        self._num_aborts = 0
        self.request_output_queue = init_request_output_queue_client(
                                        QueueType(entrypoints_args.request_output_queue_type))
        self.server = self.init_server(entrypoints_args)
        ray.get(self.server.run.remote())

    def init_server(self, entrypoints_args):
        server = APIServerActor.options(name=ENTRYPOINTS_ACTOR_NAME,
                                        namespace='llumnix').remote(entrypoints_args)
        return server

    # pylint: disable=arguments-renamed
    @classmethod
    def from_args(cls, entrypoints_args):
        manager_class = ray.remote(num_cpus=1,
                                   name=get_manager_name(),
                                   namespace='llumnix',
                                   lifetime='detached')(cls)
        manager = manager_class.remote(entrypoints_args)
        return manager


@ray.remote(num_cpus=1, lifetime="detached")
class APIServerActor:
    def __init__(self, entrypoints_args):
        self.host = entrypoints_args.host
        self.port = entrypoints_args.port
        self.request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)

    def _setup_entrypoints_context(self):
        self.entrypoints_context = setup_entrypoints_context(self.request_output_queue_type)

    def _run_uvicorn_server(self):
        run_uvicorn_server(self.host, self.port, self.entrypoints_context)

    def run(self):
        self._setup_entrypoints_context()
        self.run_uvicorn_server_thread = threading.Thread(
            target=self._run_uvicorn_server, args=(),
            daemon=True, name="run_uvicorn_server"
        )
        self.run_uvicorn_server_thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-output-queue-type", type=str, choices=["zmq", "rayqueue"])
    entrypoints_args = parser.parse_args()

    # magic actor, without this actor, APIServer cannot initialize correctly.
    # If this actor is placed globally,
    # pylint will hangs if testing api_server_manager and api_server_service concurrently (--jobs > 1).
    request_output_queue = RayQueue(actor_options={"namespace": "llumnix",
                                                   "name": "magic_ray_queue"})

    manager = MockManagerServer.from_args(entrypoints_args)

    while True:
        time.sleep(100.0)
