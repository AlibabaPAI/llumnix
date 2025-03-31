import time
import pickle
from ray.util.queue import Queue as RayQueue

from blade_llm.service.args import ServingArgs, add_args
from blade_llm.service.server import check_ports

from llumnix.entrypoints.bladellm.arg_utils import add_llumnix_cli_args, get_args, BladellmEngineArgs
from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.config import get_llumnix_config
from llumnix.entrypoints.utils import LaunchMode
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.setup import connect_to_ray_cluster, setup_llumnix


if __name__ == "__main__":
    parser = add_args()
    cli_args = parser.parse_args()
    engine_args = ServingArgs.from_cli_args(cli_args)

    # check port first
    check_ports(engine_args)

    # TODO(s5u13b): Fix it, cannot use parser of bladellm because Llumnix need to set namespace.
    # generate llumnix_parser for checking parameters with choices
    parser = LlumnixArgumentParser()
    parser = add_llumnix_cli_args(parser)
    llumnix_config = get_llumnix_config(engine_args.llumnix_config, cli_args=engine_args.llumnix_opts)

    # Assume that there is an existing ray cluster when using centralized deployment.
    connect_to_ray_cluster()

    entrypoints_args, manager_args, instance_args, engine_args = get_args(llumnix_config, parser, engine_args)
    launch_args = LaunchArgs(launch_mode=LaunchMode.GLOBAL, backend_type=BackendType.BLADELLM)

    # magic actor to avoid fast api server actor initialization error
    request_output_queue = RayQueue(actor_options={"namespace": "llumnix",
                                                   "name": "magic_ray_queue"})

    # Since importing the bladellm engine arguments requires available GPU,
    # serialize the engine parameters before passing them to the manager.
    engine_args_llumnix = BladellmEngineArgs()
    engine_args_llumnix.engine_args = pickle.dumps(engine_args)
    engine_args_llumnix.world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
    # Hack to pass engine args to APIServerActor.
    entrypoints_args.engine_args = engine_args_llumnix.engine_args
    setup_llumnix(entrypoints_args, manager_args, instance_args, engine_args_llumnix, launch_args)

    # keep the process alive to get the terminal output.
    if not entrypoints_args.disable_keep_serve_process_alive:
        while True:
            time.sleep(100.0)
