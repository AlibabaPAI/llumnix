from llumnix.arg_utils import ServiceArgs, save_engine_args
from llumnix.entrypoints.bladellm.arg_utils import add_engine_cli_args, get_engine_args, BladellmEngineArgs
from llumnix.entrypoints.bladellm.serve import launch_job_on_gpu_node

def main():
    # pylint: disable=import-outside-toplevel
    from blade_llm.utils.argparse_helper import PatchedArgumentParser

    parser = PatchedArgumentParser()
    RegisterServiceArgs.add_cli_args(parser)

    parser = add_engine_cli_args(parser)
    cli_args = parser.parse_args()
    engine_args = get_engine_args(cli_args)
    bladellm_engine_args = BladellmEngineArgs(engine_args)

    save_engine_args(cli_args.engine_type, cli_args.save_path, bladellm_engine_args, cli_args.save_key)


if __name__ == "__main__":
    launch_job_on_gpu_node(module="llumnix.entrypoints.bladellm.register_service", main_func=main)
