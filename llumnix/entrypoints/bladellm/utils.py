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

from blade_llm.service.args import ServingArgs

from llumnix.logging.logger import init_logger
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import EntrypointsArgs, ManagerArgs, InstanceArgs, LaunchMode

logger = init_logger(__name__)


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
    elif engine_args.enable_hybrid_dp:
        unsupported_feature = "hybrid data parallel"

    if unsupported_feature:
        raise ValueError(f'Llumnix does not support "{unsupported_feature}" for bladeLLM currently.')

def get_args(llumnix_cfg, llumnix_parser, engine_args: ServingArgs):
    instance_args = InstanceArgs.from_llumnix_config(llumnix_cfg)
    instance_args.init_from_engine_args(engine_args, BackendType.BLADELLM)
    manager_args = ManagerArgs.from_llumnix_config(llumnix_cfg)
    manager_args.init_from_instance_args(instance_args)
    entrypoints_args = EntrypointsArgs.from_llumnix_config(llumnix_cfg)

    EntrypointsArgs.check_args(entrypoints_args, llumnix_parser)
    instance_args.check_args(instance_args, manager_args, LaunchMode.LOCAL, llumnix_parser)
    ManagerArgs.check_args(manager_args, llumnix_parser)

    assert not manager_args.simulator_mode, "Only support the simulator mode for vLLM."

    assert not (engine_args.enable_disagg and manager_args.enable_pd_disagg), \
        "Cannot enable both pd-disaggregation inside the LLM engine and pd-disaggregation from Lluminx."

    detect_unsupported_engine_feature(engine_args)

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args
