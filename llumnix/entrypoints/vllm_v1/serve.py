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

from ray.util.queue import Queue as RayQueue

from llumnix.entrypoints.vllm_v1.arg_utils import VLLMV1EngineArgs, add_cli_args, get_args
from llumnix.entrypoints.setup import connect_to_ray_cluster
from llumnix.config import get_llumnix_config
from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.entrypoints.utils import LaunchMode
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.setup import setup_llumnix


if __name__ == "__main__":
    # Assume that there is an existing ray cluster when using centralized deployment.
    connect_to_ray_cluster()

    parser: LlumnixArgumentParser = LlumnixArgumentParser()
    parser.add_argument('--api-server-count', '-asc', type=int, help='How many API server processes to run.')
    parser.add_argument("--server-log-level", type=str, choices=['debug', 'info', 'warning', 'error'])
    parser.add_argument('--disable-keep-serve-process-alive', action='store_true')

    parser = add_cli_args(parser)
    cli_args = parser.parse_args()
    llumnix_config = get_llumnix_config(cli_args.config_file, args=cli_args)
    entrypoints_args, manager_args, instance_args, engine_args = get_args(llumnix_config, LaunchMode.GLOBAL, parser, cli_args)
    backend_type = BackendType.VLLM_V1 if not instance_args.simulator_mode else BackendType.SIM_VLLM
    launch_args = LaunchArgs(launch_mode=LaunchMode.GLOBAL, backend_type=backend_type)
    vllm_engine_args = VLLMV1EngineArgs(engine_args, backend_type)

    # magic actor to avoid fast api server actor initialization error
    request_output_queue = RayQueue(actor_options={"namespace": "llumnix",
                                                   "name": "magic_ray_queue"})

    setup_llumnix(entrypoints_args, manager_args, instance_args, vllm_engine_args, launch_args)

    # keep the process alive to get the terminal output.
    if not entrypoints_args.disable_keep_serve_process_alive:
        while True:
            time.sleep(100.0)
