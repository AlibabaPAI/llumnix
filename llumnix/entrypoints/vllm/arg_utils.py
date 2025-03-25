from typing import Tuple

from vllm.engine.arg_utils import AsyncEngineArgs

from llumnix.logging.logger import init_logger
from llumnix.backends.backend_interface import BackendType
from llumnix.backends.vllm.utils import check_engine_args, check_instance_args
from llumnix.arg_utils import EntrypointsArgs, ManagerArgs, InstanceArgs, LlumnixArgumentParser
from llumnix.entrypoints.utils import LaunchMode
from llumnix.utils import load_engine_args

logger = init_logger(__name__)


def add_cli_args(parser: LlumnixArgumentParser) -> LlumnixArgumentParser:
    parser.set_namespace("llumnix")
    parser = EntrypointsArgs.add_cli_args(parser)
    parser = ManagerArgs.add_cli_args(parser)
    parser = InstanceArgs.add_cli_args(parser)
    parser.set_namespace("vllm")
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser

def add_engine_cli_args(parser: LlumnixArgumentParser) -> "Namespace":
    parser = AsyncEngineArgs.add_cli_args(parser)
    cli_args = parser.parse_args()

    return cli_args

def get_args(cfg, launch_mode: LaunchMode, parser: LlumnixArgumentParser, cli_args: "Namespace") \
        -> Tuple[EntrypointsArgs, ManagerArgs, InstanceArgs, AsyncEngineArgs]:
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    instance_args: InstanceArgs = InstanceArgs.from_llumnix_config(cfg)
    instance_args.init_from_engine_args(engine_args, BackendType.VLLM)
    manager_args = ManagerArgs.from_llumnix_config(cfg)
    manager_args.init_from_instance_args(instance_args)
    entrypoints_args = EntrypointsArgs.from_llumnix_config(cfg)

    EntrypointsArgs.check_args(entrypoints_args, parser)
    ManagerArgs.check_args(manager_args, parser, launch_mode)
    InstanceArgs.check_args(instance_args, manager_args, launch_mode, parser)

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))

    if manager_args.load_registered_service:
        if not manager_args.enable_pd_disagg and not manager_args.enable_engine_pd_disagg:
            instance_type_list = ['no_constraints']
        else:
            instance_type_list = ['prefill', 'decode']
        for instance_type in instance_type_list:
            engine_args_registered = load_engine_args(instance_type, manager_args.load_registered_service_path)
            check_instance_args(instance_args, engine_args_registered)
        return entrypoints_args, manager_args, instance_args, engine_args

    check_engine_args(engine_args)
    check_instance_args(instance_args, engine_args)
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args

def get_engine_args(cli_args: "Namespace") -> AsyncEngineArgs:
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    check_engine_args(engine_args)
    logger.info("engine_args: {}".format(engine_args))

    return engine_args
