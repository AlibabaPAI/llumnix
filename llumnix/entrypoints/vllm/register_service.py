import argparse

from llumnix.arg_utils import ServiceArgs, save_engine_args
from llumnix.entrypoints.vllm.arg_utils import add_engine_cli_args, get_engine_args, VllmEngineArgs

# TODO(s5u13b): Add examples for pdd launch.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServiceArgs.add_cli_args(parser)

    parser = add_engine_cli_args(parser)
    cli_args = parser.parse_args()
    engine_args = get_engine_args(cli_args)
    vllm_engine_args = VllmEngineArgs(engine_args)

    save_engine_args(cli_args.engine_type, cli_args.save_path, vllm_engine_args, cli_args.save_key)
