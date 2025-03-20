import argparse

from llumnix.entrypoints.vllm.arg_utils import add_engine_cli_args, get_engine_args
from llumnix.utils import put_engine_args_to_ray_internal_kv
from llumnix.entrypoints.setup import connect_to_ray_cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--instance-type',
                        type=str,
                        choices=['prefill', 'decode', 'no_constraints'],
                        default='no_constraints',
                        help="Instance type of the engine arguments.")

    cli_args = add_engine_cli_args(parser)
    engine_args = get_engine_args(cli_args)

    connect_to_ray_cluster()

    put_engine_args_to_ray_internal_kv(cli_args.instance_type, engine_args)
