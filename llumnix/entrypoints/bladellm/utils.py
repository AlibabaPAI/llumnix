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
from llumnix.arg_utils import LlumnixEntrypointsArgs, EngineManagerArgs, InstanceArgs

def detect_unsupported_engine_feature(engine_args: ServingArgs) -> None:
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

def detect_unsupported_llumnix_feature(engine_manager_args: EngineManagerArgs) -> None:
    assert engine_manager_args.migration_backend in ['grpc', 'kvtransfer'], \
        "Only grpc and kvtransfer are supported for BladeLLM now."
    assert not engine_manager_args.enable_pd_disagg, \
        "PD disaggregation based no llumnix is not supported for BladeLLM now."

def get_args(llumnix_cfg, llumnix_parser, engine_args: ServingArgs):
    llumnix_entrypoints_args = LlumnixEntrypointsArgs.from_llumnix_config(llumnix_cfg)
    LlumnixEntrypointsArgs.check_args(llumnix_entrypoints_args, llumnix_parser)
    engine_manager_args = EngineManagerArgs.from_llumnix_config(llumnix_cfg)
    EngineManagerArgs.check_args(engine_manager_args, llumnix_parser)
    instance_args = InstanceArgs()
    instance_args.instance_type = engine_args.disagg_options.inst_role if engine_args.enable_disagg else "no_constraints"
    detect_unsupported_engine_feature(engine_args)
    detect_unsupported_llumnix_feature(engine_manager_args)

    logger.info("llumnix_entrypoints_args: {}", llumnix_entrypoints_args)
    logger.info("engine_manager_args: {}", engine_manager_args)
    logger.info("instance_args: {}", instance_args)
    logger.info("engine_args: {}", engine_args)

    return llumnix_entrypoints_args, engine_manager_args, instance_args, engine_args
