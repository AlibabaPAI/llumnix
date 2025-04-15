import argparse

from llumnix.entrypoints.vllm.arg_utils import add_engine_cli_args, get_engine_args, VllmEngineArgs
from llumnix.arg_utils import save_engine_args
from llumnix.entrypoints.setup import connect_to_ray_cluster

# TODO(s5u13b): Add examples for pdd launch.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--engine-type',
                        type=str,
                        choices=['prefill', 'decode', 'no_constraints'],
                        default='no_constraints',
                        help="Engine type of the engine arguments. The actual save filename is generated according to "
                             "the engine type, following the format f\"engine_args_{engine_type}.pkl\".")
    parser.add_argument('--save-key',
                        type=str,
                        help="Save key of the engine arguments. The actual save filepath is generated according to "
                             "the save path and save key, following the organization f\"{save_path}/{save_key}/\".")
    parser.add_argument('--save-path',
                        type=str,
                        default='.',
                        help="Save path of the engine arguments.")

    parser = add_engine_cli_args(parser)
    cli_args = parser.parse_args()
    engine_args = get_engine_args(cli_args)
    llumnix_engine_args = VllmEngineArgs(override_engine_args=engine_args)

    connect_to_ray_cluster()

    save_engine_args(cli_args.engine_type, cli_args.save_path, llumnix_engine_args, cli_args.save_key)
