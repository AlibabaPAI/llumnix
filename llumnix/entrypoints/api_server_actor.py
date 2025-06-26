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

from abc import ABC, abstractmethod
import threading
import time

import requests
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray.actor

from llumnix.arg_utils import EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.queue.utils import init_request_output_queue_server, QueueType
from llumnix.logging.logger import init_logger
from llumnix.ray_utils import log_actor_ray_info, get_server_name
from llumnix.constants import SERVER_STOP_TIMEOUT, SERVER_START_TIMEOUT


logger = init_logger(__name__)


class APIServerActor(ABC):
    def __init__(self,
                 instance_id: str,
                 entrypoints_args: EntrypointsArgs,
                 engine_args,
                 scaler: ray.actor.ActorHandle,
                 manager: ray.actor.ActorHandle,
                 instance: ray.actor.ActorHandle):
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.instance_id = instance_id
        self.entrypoints_args = entrypoints_args
        self.engine_args = engine_args
        self.scaler = scaler
        self.manager = manager
        self.instance = instance
        self._set_host(entrypoints_args, engine_args)
        self._set_health_api()
        self.port = entrypoints_args.port
        self.request_output_queue_type = QueueType(self.entrypoints_args.request_output_queue_type)
        self.request_output_queue = init_request_output_queue_server(
            self.host, self.request_output_queue_type)

        self._setup_entrypoints_context(self.scaler, self.manager, self.instance_id, self.instance)
        self._start_server()
        self._wait_server()

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]})"

    def _setup_entrypoints_context(self,
                                   scaler: ray.actor.ActorHandle,
                                   manager: ray.actor.ActorHandle,
                                   instance_id: str,
                                   instance: ray.actor.ActorHandle):
        # avoid circular import
        # pylint: disable=import-outside-toplevel
        from llumnix.entrypoints.setup import setup_entrypoints_context
        self.entrypoints_context = setup_entrypoints_context(
            self.entrypoints_args, scaler, manager, [instance_id], [instance], self.request_output_queue)

    @abstractmethod
    def _start_server(self):
        raise NotImplementedError

    @abstractmethod
    def _set_host(self, entrypoints_args: EntrypointsArgs, engine_args):
        raise NotImplementedError

    @abstractmethod
    def _set_health_api(self):
        raise NotImplementedError

    @abstractmethod
    def _run_server(self,
                    entrypoints_args: EntrypointsArgs,
                    engine_args,
                    entrypoints_context: EntrypointsContext):
        raise NotImplementedError

    def _wait_server(self):
        start_time = time.time()
        while time.time() - start_time < SERVER_START_TIMEOUT:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/{self.health_api}", timeout=0.1)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"Server {self.instance_id} failed to start in {SERVER_START_TIMEOUT} seconds.")

    def stop(self):
        self._stop_server()
        self._stop_thread()

    @abstractmethod
    def _stop_server(self):
        raise NotImplementedError

    def _stop_thread(self):
        if self.run_server_thread.is_alive():
            self.run_server_thread.join(timeout=SERVER_STOP_TIMEOUT)
            if self.run_server_thread.is_alive():
                logger.error("Failed to stop server {} gracefully.".format(self.instance_id))

    @classmethod
    def from_args(cls,
                  num_gpus: int,
                  instance_id: str,
                  placement_group: PlacementGroup,
                  entrypoints_args: EntrypointsArgs,
                  engine_args,
                  scaler: ray.actor.ActorHandle,
                  manager: ray.actor.ActorHandle,
                  instance: ray.actor.ActorHandle):
        api_server_class = ray.remote(
            num_cpus=1,
            num_gpus=num_gpus,
            name=get_server_name(instance_id),
            namespace="llumnix",
            lifetime="detached"
        )(cls).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True
            )
        )
        api_server = api_server_class.remote(
            instance_id, entrypoints_args, engine_args, scaler, manager, instance)

        return api_server

    def is_ready(self) -> bool:
        return True
