from typing import Tuple

from vllm.engine.arg_utils import AsyncEngineArgs

from llumnix.logging.logger import init_logger
from llumnix.backends.backend_interface import BackendType
from llumnix.backends.vllm.utils import check_engine_args
from llumnix.arg_utils import EntrypointsArgs, ManagerArgs, InstanceArgs, LlumnixArgumentParser
from llumnix.entrypoints.utils import LaunchMode

logger = init_logger(__name__)


def add_cli_args(parser: LlumnixArgumentParser) -> LlumnixArgumentParser:
    parser.set_namespace("llumnix")
    parser = EntrypointsArgs.add_cli_args(parser)
    parser = ManagerArgs.add_cli_args(parser)
    parser = InstanceArgs.add_cli_args(parser)
    parser.set_namespace("vllm")
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser

def get_args(cfg, launch_mode: LaunchMode, parser: LlumnixArgumentParser, cli_args: "Namespace") \
        -> Tuple[EntrypointsArgs, ManagerArgs, InstanceArgs, AsyncEngineArgs]:
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    instance_args: InstanceArgs = InstanceArgs.from_llumnix_config(cfg)
    instance_args.init_from_engine_args(engine_args, BackendType.VLLM)
    manager_args = ManagerArgs.from_llumnix_config(cfg)
    manager_args.init_from_instance_args(instance_args)
    entrypoints_args = EntrypointsArgs.from_llumnix_config(cfg)

    EntrypointsArgs.check_args(entrypoints_args, parser)
    ManagerArgs.check_args(manager_args, parser)
    InstanceArgs.check_args(instance_args, manager_args, launch_mode, parser)
    check_engine_args(engine_args, instance_args)

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args
