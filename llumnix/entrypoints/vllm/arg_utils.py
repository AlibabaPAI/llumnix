from typing import Tuple

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.config import EngineConfig, ParallelConfig

from llumnix.logging.logger import init_logger
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import EntrypointsArgs, ManagerArgs, InstanceArgs, LlumnixArgumentParser, LlumnixEngineArgs
from llumnix.entrypoints.utils import LaunchMode
from llumnix.arg_utils import load_engine_args
from llumnix.internal_config import MigrationConfig
from llumnix.config import LlumnixConfig

logger = init_logger(__name__)

class VllmEngineArgs(LlumnixEngineArgs):

    def __init__(self, engine_args=None) -> None:
        super().__init__(engine_args=engine_args, backend_type=BackendType.VLLM)

    def unwrap_engine_args_if_needed(self):
        return self.engine_args

    def get_engine_world_size(self):
        engine_config = self.engine_args.create_engine_config()
        return engine_config.parallel_config.world_size

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

    return parser

def detect_unsupported_engine_feature(engine_args: EngineArgs) -> None:
    unsupported_feature = None
    if engine_args.enable_lora:
        unsupported_feature = "multi-lora serving"
    elif engine_args.enable_prefix_caching:
        unsupported_feature = "automatic prefix caching"
    elif engine_args.enable_chunked_prefill:
        unsupported_feature = "chunked prefill"
    elif engine_args.speculative_model:
        unsupported_feature = "speculative decoding"
    elif engine_args.pipeline_parallel_size > 1:
        unsupported_feature = "pipeline parallel"
    elif engine_args.num_scheduler_steps > 1:
        unsupported_feature = "multi-step scheduling"
    if unsupported_feature:
        raise ValueError(f'Unsupported feature: Llumnix does not support "{unsupported_feature}" currently.')

def check_engine_args(engine_args: AsyncEngineArgs) -> None:
    detect_unsupported_engine_feature(engine_args)
    assert engine_args.worker_use_ray, "In Llumnix, engine and worker must be ray actor."

def check_instance_args(intance_args: InstanceArgs, engine_args: AsyncEngineArgs) -> None:
    migration_config: MigrationConfig = intance_args.create_migration_config()
    engine_config: EngineConfig = engine_args.create_engine_config()
    parallel_config: ParallelConfig = engine_config.parallel_config
    assert not (parallel_config.world_size > 1 and migration_config.migration_backend == 'nccl'), \
        "Llumnix does not support TP or PP when the migration backend is nccl, please change migration backend."
    assert not (not engine_args.disable_async_output_proc and intance_args.simulator_mode), \
        "Llumnix does not support async output processing when enabling simualtor mode, please disable async output processing."

def get_args(llumnix_config: LlumnixConfig, launch_mode: LaunchMode, parser: LlumnixArgumentParser, cli_args: "Namespace") \
        -> Tuple[EntrypointsArgs, ManagerArgs, InstanceArgs, AsyncEngineArgs]:
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    instance_args: InstanceArgs = InstanceArgs.from_llumnix_config(llumnix_config)
    instance_args.init_from_engine_args(engine_args)
    manager_args = ManagerArgs.from_llumnix_config(llumnix_config)
    manager_args.init_from_instance_args(instance_args)
    entrypoints_args = EntrypointsArgs.from_llumnix_config(llumnix_config)

    EntrypointsArgs.check_args(entrypoints_args, parser)
    ManagerArgs.check_args(manager_args, launch_mode, parser)
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
            check_instance_args(instance_args, engine_args_registered.engine_args)
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
