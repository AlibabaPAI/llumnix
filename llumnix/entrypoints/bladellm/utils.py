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


from loguru import logger

from blade_llm.service.args import ServingArgs
from llumnix.arg_utils import LlumnixEntrypointsArgs, EngineManagerArgs

def detect_unsupported_feature(engine_args: ServingArgs) -> None:
    unsupported_feature = None
    if engine_args.enable_lora:
        unsupported_feature = "multi-lora serving"
    elif not engine_args.disable_prompt_cache:
        unsupported_feature = "automatic prompt caching"
    elif engine_args.use_sps:
        unsupported_feature = "speculative decoding"
    elif engine_args.enable_remote_worker:
        unsupported_feature = "enable_remote_worker"

    if unsupported_feature:
        raise ValueError(f'Llumnix does not support "{unsupported_feature}" for bladeLLM currently.')

def check_engine_args(engine_args: ServingArgs, engine_manager_args: EngineManagerArgs) -> None:
    migration_config = engine_manager_args.create_migration_config()
    if (engine_args.tensor_parallel_size > 1 or engine_args.tensor_parallel_size > 1) and \
        migration_config.migration_backend == 'nccl':
        logger.info("Llumnix does not support TP or PP enabled model when the migration backend is nccl, \
                    change migration backend to gloo.")
        engine_manager_args.migration_backend = 'gloo'
    detect_unsupported_feature(engine_args)

def get_args(llumnix_cfg, llumnix_parser, engine_args):
    llumnix_entrypoints_args = LlumnixEntrypointsArgs.from_llumnix_config(llumnix_cfg)
    LlumnixEntrypointsArgs.check_args(llumnix_entrypoints_args, llumnix_parser)
    engine_manager_args = EngineManagerArgs.from_llumnix_config(llumnix_cfg)
    EngineManagerArgs.check_args(engine_manager_args, llumnix_parser)
    check_engine_args(engine_args, engine_manager_args)

    logger.info("llumnix_entrypoints_args: {}", llumnix_entrypoints_args)
    logger.info("engine_manager_args: {}", engine_manager_args)
    logger.info("engine_args: {}", engine_args)

    return llumnix_entrypoints_args, engine_manager_args, engine_args
