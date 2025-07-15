# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
import copy

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.config import EngineConfig, ParallelConfig

from llumnix.logging.logger import init_logger
from llumnix.utils import BackendType, LaunchMode
from llumnix.arg_utils import (EntrypointsArgs, ManagerArgs, InstanceArgs, LlumnixEngineArgsFactory,
                               LlumnixArgumentParser, LlumnixEngineArgs, load_registered_engine_args,
                               init_llumnix_args, post_init_llumnix_args)
from llumnix.internal_config import MigrationConfig
from llumnix.config import LlumnixConfig

logger = init_logger(__name__)


class VLLMEngineArgsFactory(LlumnixEngineArgsFactory):
    # pylint: disable=unused-argument
    def gen_next_engine_args(
        self,
        current_engine_args: LlumnixEngineArgs,
        next_instance_args: InstanceArgs,
        port_offset: int = 0,
        instance_id: str = None,
    ) -> LlumnixEngineArgs:
        if self.load_registered_service:
            instance_type = next_instance_args.instance_type
            current_engine_args = self.engine_args_dict[instance_type]
        return copy.deepcopy(current_engine_args)


class VLLMEngineArgs(LlumnixEngineArgs):
    def __init__(self,
                 engine_args: AsyncEngineArgs,
                 backend_type: BackendType = BackendType.VLLM) -> None:
        engine_args = self._get_engine_args(engine_args)
        super().__init__(engine_args=engine_args, backend_type=backend_type)

    def _get_engine_args(self, engine_args: AsyncEngineArgs):
        return engine_args

    def load_engine_args(self):
        return self.engine_args

    def get_world_size(self):
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

def add_engine_cli_args(parser: "ArgumentParser") -> "Namespace":
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

def check_instance_args(instance_args: InstanceArgs, engine_args: AsyncEngineArgs) -> None:
    migration_config: MigrationConfig = instance_args.create_migration_config()
    engine_config: EngineConfig = engine_args.create_engine_config()
    parallel_config: ParallelConfig = engine_config.parallel_config

    assert instance_args.migration_backend in ['rayrpc', 'gloo', 'nccl'], \
        "Only support rayrpc, gloo and nccl migration backend for vLLM."

    assert not (parallel_config.world_size > 1 and migration_config.migration_backend == 'nccl'), \
        "Llumnix does not support TP or PP when the migration backend is nccl, please change migration backend."

    assert not (not engine_args.disable_async_output_proc and instance_args.simulator_mode), \
        "Llumnix does not support async output processing when enabling simualtor mode, please disable async output processing."

def get_args(llumnix_config: LlumnixConfig, launch_mode: LaunchMode, parser: LlumnixArgumentParser, cli_args: "Namespace") \
        -> Tuple[EntrypointsArgs, ManagerArgs, InstanceArgs, AsyncEngineArgs]:
    entrypoints_args, manager_args, instance_args = init_llumnix_args(llumnix_config)
    if manager_args.load_registered_service:
        engine_args = load_registered_engine_args(manager_args)
    else:
        engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    post_init_llumnix_args(engine_args, instance_args, manager_args, entrypoints_args, BackendType.VLLM, launch_mode, parser)

    # backend related check args
    check_engine_args(engine_args)
    check_instance_args(instance_args, engine_args)

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args

def get_engine_args(cli_args: "Namespace") -> AsyncEngineArgs:
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    check_engine_args(engine_args)
    logger.info("engine_args: {}".format(engine_args))

    return engine_args
