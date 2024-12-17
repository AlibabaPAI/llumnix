from vllm.engine.arg_utils import AsyncEngineArgs
from llumnix.backends.vllm.utils import check_engine_args

from llumnix.arg_utils import LlumnixEntrypointsArgs, EngineManagerArgs
from llumnix.logger import init_logger

logger = init_logger(__name__)


def add_cli_args(parser):
    parser.set_namespace("llumnix")
    parser = LlumnixEntrypointsArgs.add_cli_args(parser)
    parser = EngineManagerArgs.add_cli_args(parser)
    parser.set_namespace("vllm")
    parser = AsyncEngineArgs.add_cli_args(parser)
    cli_args = parser.parse_args()
    return cli_args

def get_args(cfg, parser, cli_args):
    llumnix_entrypoints_args = LlumnixEntrypointsArgs.from_llumnix_config(cfg)
    LlumnixEntrypointsArgs.check_args(llumnix_entrypoints_args, parser)
    engine_manager_args = EngineManagerArgs.from_llumnix_config(cfg)
    EngineManagerArgs.check_args(engine_manager_args, parser)
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    check_engine_args(engine_args, engine_manager_args)

    logger.info("llumnix_entrypoints_args: {}".format(llumnix_entrypoints_args))
    logger.info("engine_manager_args: {}".format(engine_manager_args))
    logger.info("engine_args: {}".format(engine_args))

    return llumnix_entrypoints_args, engine_manager_args, engine_args
