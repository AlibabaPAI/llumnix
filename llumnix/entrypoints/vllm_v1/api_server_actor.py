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

import socket
from dataclasses import fields

import uvloop

import ray

from llumnix.arg_utils import VLLMV1EntrypointsArgs
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.entrypoints.vllm_v1.arg_utils import VLLMV1EngineArgs
from llumnix.logging.logger import init_logger
from llumnix.utils import get_ip_address

# from llumnix.entrypoints.vllm_v1.utils import (setup_server, run_server_proc)

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)


class APIServerActorVLLMV1(APIServerActor):
    def __init__(self,
                 instance_id: str,
                 entrypoints_args: VLLMV1EntrypointsArgs,
                 engine_args: VLLMV1EngineArgs,
                 scaler: "ray.actor.ActorHandle",
                 manager: "ray.actor.ActorHandle",
                 instance: "ray.actor.ActorHandle"):
        self.client_index = entrypoints_args.client_index
        super().__init__(instance_id, entrypoints_args, engine_args,
                         scaler, manager, instance)

    def _set_host(self, entrypoints_args: VLLMV1EntrypointsArgs, engine_args):
        if entrypoints_args.host not in ("127.0.0.1", "0.0.0.0"):
            entrypoints_args.host = get_ip_address()
        self.host = entrypoints_args.host
        
        # Set up listen address and socket
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.sock.bind((self.host, entrypoints_args.port))

        # pylint: disable=import-outside-toplevel
        from vllm.utils import set_ulimit
        set_ulimit()

        is_ssl = entrypoints_args.ssl_keyfile and entrypoints_args.ssl_certfile
        self.listen_address = f"http{'s' if is_ssl else ''}://" + \
                              f"{self.host}:{entrypoints_args.port}"

    def _set_health_api(self):
        self.health_api = "health"

    def _run_server(self,
                    entrypoints_args: VLLMV1EntrypointsArgs,
                    engine_args: VLLMV1EngineArgs,
                    entrypoints_context: EntrypointsContext):
        client_config = {
            "client_index": self.client_index
        }

        serve_args = engine_args.load_engine_args()
        for field in fields(VLLMV1EntrypointsArgs):
            setattr(serve_args, field.name, getattr(entrypoints_args, field.name))
        # NOTE(shejiarui): `speculative_config` has been set somewhere,
        # set it to `None` here to avoid exceptions.
        if hasattr(serve_args, "speculative_config"):
            serve_args.speculative_config = None

        # Set global variable in vLLM
        import vllm.v1.engine.core_client
        from vllm.entrypoints.openai.api_server import run_server_worker
        vllm.v1.engine.core_client.entrypoints_context = self.entrypoints_context
        uvloop.run(
            run_server_worker(self.listen_address, self.sock,
                              serve_args, client_config))

    def _stop_server(self):
        def stop_server():
            self.server.should_exit = True

        if self.loop.is_running():
            self.loop.call_soon_threadsafe(stop_server)
