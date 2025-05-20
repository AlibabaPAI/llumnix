import time

from ray.util.queue import Queue as RayQueue

from llumnix.entrypoints.bladellm.arg_utils import add_cli_args, get_args, BladellmEngineArgs
from llumnix.arg_utils import LaunchArgs, LlumnixArgumentParser
from llumnix.logging.logger import init_logger
from llumnix.config import get_llumnix_config
from llumnix.entrypoints.utils import LaunchMode
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.setup import setup_llumnix
from llumnix.entrypoints.bladellm.utils import launch_job_on_gpu_node

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


if __name__ == "__main__":
    launch_job_on_gpu_node(module="llumnix.entrypoints.bladellm.serve", main_func=main)
