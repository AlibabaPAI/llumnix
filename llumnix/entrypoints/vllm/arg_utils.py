from vllm.engine.arg_utils import AsyncEngineArgs
from llumnix.backends.vllm.utils import check_engine_args

from llumnix.arg_utils import EntrypointsArgs, ManagerArgs
from llumnix.logger import init_logger

logger = init_logger(__name__)


def add_cli_args(parser):
    parser.set_namespace("llumnix")
    parser = EntrypointsArgs.add_cli_args(parser)
    parser = ManagerArgs.add_cli_args(parser)
    parser.set_namespace("vllm")
    parser = AsyncEngineArgs.add_cli_args(parser)
    cli_args = parser.parse_args()
    return cli_args

def get_args(cfg, parser, cli_args):
    entrypoints_args = EntrypointsArgs.from_llumnix_config(cfg)
    EntrypointsArgs.check_args(entrypoints_args, parser)
    manager_args = ManagerArgs.from_llumnix_config(cfg)
    ManagerArgs.check_args(manager_args, parser)
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    check_engine_args(engine_args, manager_args)

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, engine_args
