
import time
from ray.util.queue import Queue as RayQueue

from llumnix.entrypoints.vllm.arg_utils import add_cli_args, get_args
from llumnix.entrypoints.setup import connect_to_ray_cluster
from llumnix.config import get_llumnix_config
from llumnix.arg_utils import LlumnixArgumentParser, DeploymentArgs
from llumnix.entrypoints.utils import DeploymentMode
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.setup import setup_llumnix


if __name__ == "__main__":
    parser: LlumnixArgumentParser = LlumnixArgumentParser()

    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)
    parser.add_argument("--log-level", type=str)

    cli_args = add_cli_args(parser)
    cfg = get_llumnix_config(cli_args.config_file, cli_args)
    entrypoints_args, manager_args, engine_args = get_args(cfg, parser, cli_args)

    deployment_args = DeploymentArgs(deployment_mode=DeploymentMode.GLOBAL, backend_type=BackendType.VLLM)

    # Assume that there is an existing ray cluster when using centralized deployment.
    connect_to_ray_cluster()

    # magic actor to avoid fast api server actor initialization error
    request_output_queue = RayQueue(actor_options={"namespace": "llumnix",
                                                   "name": "magic_ray_queue"})

    setup_llumnix(manager_args, entrypoints_args, engine_args, deployment_args)

    # keep the process alive to get the terminal output.
    while True:
        time.sleep(100.0)