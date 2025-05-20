import time
import sys
import os

import ray
from ray.util.queue import Queue as RayQueue

from llumnix.entrypoints.bladellm.arg_utils import add_cli_args, get_args, BladellmEngineArgs
from llumnix.arg_utils import LaunchArgs, LlumnixArgumentParser
from llumnix.logging.logger import init_logger
from llumnix.config import get_llumnix_config
from llumnix.entrypoints.utils import LaunchMode
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.setup import connect_to_ray_cluster, setup_llumnix

logger = init_logger('llumnix.entrypoints.bladellm.server')

def main():
    # pylint: disable=import-outside-toplevel
    from blade_llm.utils.argparse_helper import PatchedArgumentParser
    # Make ArgumentParser of Llumnix compatible to the ArgumentParser of BladeLLM.
    # Because import bladellm will raise no available gpu error, so define class inside the main function.
    class LlumnixBladeLLMArgumentParser(LlumnixArgumentParser, PatchedArgumentParser):
        # pylint: disable=super-init-not-called
        def __init__(self, *args, **kwargs):
            # Not explicity call the constructor of LlumnixArgumentParser.
            self.cur_namespace = None
            PatchedArgumentParser.__init__(self, *args, **kwargs)

    parser: LlumnixBladeLLMArgumentParser = LlumnixBladeLLMArgumentParser()
    parser = add_cli_args(parser)
    cli_args = parser.parse_args()
    llumnix_config = get_llumnix_config(cli_args.config_file, args=cli_args)
    entrypoints_args, manager_args, instance_args, engine_args = get_args(llumnix_config, LaunchMode.GLOBAL, parser, cli_args)
    launch_args = LaunchArgs(launch_mode=LaunchMode.GLOBAL, backend_type=BackendType.BLADELLM)
    bladellm_engine_args = BladellmEngineArgs(engine_args)
    # magic actor to avoid fast api server actor initialization error
    # pylint: disable=unused-variable
    request_output_queue = RayQueue(actor_options={"namespace": "llumnix",
                                                   "name": "magic_ray_queue"})

    setup_llumnix(entrypoints_args, manager_args, instance_args, bladellm_engine_args, launch_args)

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
    # Assume that there is an existing ray cluster when using centralized deployment.
    connect_to_ray_cluster()

    num_gpus, num_cpus = get_current_node_resources()

    if num_gpus > 0:
        # current node has GPU resources
        logger.info("Launch on current node.")
        main()
    elif num_cpus == 0 and num_gpus == 0:
        # In some Ray clusters, there may exist a master node that has no CPU resources and no GPU resources (num_cpus=0, num_gpus=0),
        # which is used solely for submitting tasks.
        # Since importing bladellm requires GPU resources, server.py cannot be run on the master node.
        # In this case, use a Ray task with num_cpus=1 to launch bladellm on other worker nodes.
        logger.info("No GPU available on the current node. Launching on another node with GPU resources.")

        # get args
        original_argv = sys.argv[1:]
        ray.get(remote_launch_task.remote(original_argv))
    else:
        logger.info(
            "Current node has CPU resources but no GPU resources, not support now."
        )
