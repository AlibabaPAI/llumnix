import time
import sys
import os
import pickle
import ray
from ray.util.queue import Queue as RayQueue

from llumnix.entrypoints.bladellm.arg_utils import add_llumnix_cli_args, get_args, BladellmEngineArgs
from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.logging.logger import init_logger
from llumnix.config import get_llumnix_config
from llumnix.entrypoints.utils import LaunchMode
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.setup import connect_to_ray_cluster, setup_llumnix

logger = init_logger('llumnix.entrypoints.bladellm.server')

# Assume that there is an existing ray cluster when using centralized deployment.
connect_to_ray_cluster()

def main():
    # pylint: disable=import-outside-toplevel
    from blade_llm.service.args import ServingArgs, add_args
    from blade_llm.service.server import check_ports
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

    entrypoints_args, manager_args, instance_args, engine_args = get_args(llumnix_config, LaunchMode.GLOBAL, parser, engine_args)
    launch_args = LaunchArgs(launch_mode=LaunchMode.GLOBAL, backend_type=BackendType.BLADELLM)

    # magic actor to avoid fast api server actor initialization error
    request_output_queue = RayQueue(actor_options={"namespace": "llumnix",
                                                   "name": "magic_ray_queue"})

    # Since importing the bladellm engine arguments requires available GPU,
    # serialize the engine parameters before passing them to the manager.
    engine_args_llumnix = BladellmEngineArgs()
    engine_args_llumnix.engine_args = pickle.dumps(engine_args)
    engine_args_llumnix.world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
    setup_llumnix(entrypoints_args, manager_args, instance_args, engine_args_llumnix, launch_args)

    # keep the process alive to get the terminal output.
    if not entrypoints_args.disable_keep_serve_process_alive:
        while True:
            time.sleep(100.0)


@ray.remote(num_cpus=1)
def remote_launch_task(serve_args):
    # set sys.argv
    sys.argv = ['llumnix.entrypoints.bladellm.serve'] + serve_args

    # avoid "RuntimeError: No CUDA GPUs are available"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main()

def get_current_node_resources():
    current_node_id = ray.get_runtime_context().get_node_id()
    resource = {}
    for node in ray.nodes():
        if node["NodeID"] == current_node_id:
            resource = node["Resources"]
            break
    return resource.get("GPU", 0), resource.get("CPU", 0)


if __name__ == "__main__":

    gpu_num, cpu_num = get_current_node_resources()

    if gpu_num > 0:
        # current node has GPU resources
        logger.info("launch on current node.")
        main()
    elif cpu_num == 0 and gpu_num == 0:
        # currnet node has no CPU resources and no GPU resources, it is highly likely that the current node is the head node
        # bladellm can only run on GPU nodes, so use a ray task to launch bladellm on other nodes
        logger.info("no gpu on current node, launch on another node.")
        # get args
        original_argv = sys.argv[1:]
        ray.get(remote_launch_task.remote(original_argv))
    else:
        logger.info(
            "currnet node has CPU resources but no GPU resources, not support now."
        )
