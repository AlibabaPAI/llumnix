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

import signal
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
        # Set up listen address and socket
        self.listen_address, self.sock = setup_server(entrypoints_args)
        super().__init__(instance_id, entrypoints_args, engine_args,
                         scaler, manager, instance)

    def _set_host(self, entrypoints_args: VLLMV1EntrypointsArgs, engine_args):
        if entrypoints_args.host not in ("127.0.0.1", "0.0.0.0"):
            entrypoints_args.host = get_ip_address()
        self.host = entrypoints_args.host

    def _set_health_api(self):
        self.health_api = "health"

    def _run_server(self,
                    entrypoints_args: VLLMV1EntrypointsArgs,
                    engine_args: VLLMV1EngineArgs,
                    entrypoints_context: EntrypointsContext):
        # If dp_size == 1, client_index will always be 0
        self.client_index = 0
        self.server_id = entrypoints_context.server_info.server_id

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


def setup_server(entrypoints_args: VLLMV1EntrypointsArgs):
    """
    Validate API server args, set up signal handler, create socket ready to serve.
    
    Main logic copied from vLLM's `setup_server`, removed some unnecessary logic.
    """
    # pylint: disable=import-outside-toplevel
    from vllm.version import __version__ as VLLM_VERSION
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    from vllm.entrypoints.openai.api_server import create_server_socket
    from vllm.utils import (is_valid_ipv6_address, set_ulimit)

    logger.info("vLLM API server version %s", VLLM_VERSION)

    if entrypoints_args.tool_parser_plugin and len(entrypoints_args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(entrypoints_args.tool_parser_plugin)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (entrypoints_args.host or "", entrypoints_args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    addr, port = sock_addr
    is_ssl = entrypoints_args.ssl_keyfile and entrypoints_args.ssl_certfile
    host_part = f"[{addr}]" if is_valid_ipv6_address(
        addr) else addr or "0.0.0.0"
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"

    return listen_address, sock
