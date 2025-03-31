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

from dataclasses import dataclass

# When importing ServingArgs, it will raise no available gpu error.

from llumnix.arg_utils import (EntrypointsArgs, ManagerArgs, LlumnixArgumentParser,
                               InstanceArgs)
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode
from llumnix.logging.logger import init_logger
from llumnix.config import LlumnixConfig

logger = init_logger(__name__)


@dataclass
class BladellmEngineArgs:
    engine_args: bytes = None
    world_size: int = None


def add_llumnix_cli_args(parser: LlumnixArgumentParser) -> LlumnixArgumentParser:
    parser.set_namespace("llumnix")
    parser = EntrypointsArgs.add_cli_args(parser)
    parser = ManagerArgs.add_cli_args(parser)
    parser = InstanceArgs.add_cli_args(parser)
    return parser

def detect_unsupported_engine_feature(engine_args) -> None:
    unsupported_feature = None
    if engine_args.enable_lora:
        unsupported_feature = "multi-lora serving"
    elif not engine_args.disable_prompt_cache:
        unsupported_feature = "automatic prompt caching"
    elif engine_args.use_sps:
        unsupported_feature = "speculative decoding"
    elif engine_args.enable_remote_worker:
        unsupported_feature = "enable_remote_worker"
    elif engine_args.enable_hybrid_dp:
        unsupported_feature = "hybrid data parallel"
    elif engine_args.elastic_attn_cluster:
        unsupported_feature = "elastic attention"

    if unsupported_feature:
        raise ValueError(f'Llumnix does not support "{unsupported_feature}" for BladeLLM currently.')

def get_args(llumnix_cfg: LlumnixConfig, launch_mode: LaunchMode, llumnix_parser: LlumnixArgumentParser, engine_args):
    instance_args = InstanceArgs.from_llumnix_config(llumnix_cfg)
    instance_args.init_from_engine_args(engine_args, BackendType.BLADELLM)
    manager_args = ManagerArgs.from_llumnix_config(llumnix_cfg)
    manager_args.init_from_instance_args(instance_args)
    entrypoints_args = EntrypointsArgs.from_llumnix_config(llumnix_cfg)
    entrypoints_args.init_from_engine_args(engine_args, BackendType.BLADELLM)

    EntrypointsArgs.check_args(entrypoints_args, llumnix_parser)
    InstanceArgs.check_args(instance_args, manager_args, launch_mode, llumnix_parser)
    ManagerArgs.check_args(manager_args, launch_mode, llumnix_parser)

    assert not instance_args.simulator_mode, "Only support the simulator mode for vLLM."
    assert not (engine_args.enable_disagg and manager_args.enable_pd_disagg), \
        "Cannot enable both pd-disaggregation inside the LLM engine and pd-disaggregation from Llumnix."
    assert 'W' not in instance_args.request_migration_policy, \
        "Migrating waiting request is not supported for BladeLLM temporarily."
    assert not engine_args.enable_disagg or not manager_args.enable_migration, \
        "Migration feature is temporarily unavailable for the engine based pd-disaggregation."
    assert engine_args.pipeline_parallel_size == 1 or not manager_args.enable_migration,\
        "Migration feature is temporarily unavailable for pipeline parallelism in BladeLLM."

    detect_unsupported_engine_feature(engine_args)

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args
