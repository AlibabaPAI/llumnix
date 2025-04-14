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

from dataclasses import dataclass, field
import pickle
# When importing ServingArgs, it will raise no available gpu error.


from llumnix.arg_utils import (EntrypointsArgs, ManagerArgs, LlumnixArgumentParser,
                               InstanceArgs, LlumnixEngineArgs)
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode
from llumnix.logging.logger import init_logger
from llumnix.config import LlumnixConfig

logger = init_logger(__name__)


class BladellmEngineArgs(LlumnixEngineArgs):

    def __init__(
        self, origin_engine_args=None, override_engine_args=None
    ):
        super().__init__(
            origin_engine_args=origin_engine_args,
            override_engine_args=override_engine_args,
            backend_type=BackendType.BLADELLM,
        )
        self.override_engine_args = self.EngineOverrideArgs()
        self.world_size: int = None
        self.instance_id: str = None

    @classmethod
    def from_cli_args(cls, cli_args="Namespace"):
        pass

    def get_current_engine_args(self):
        engine_args = pickle.loads(self.origin_engine_args)
        engine_override_args = self.override_engine_args
        if engine_args.disagg_options:
            if engine_override_args.disagg_options_token_port_offset:
                engine_args.disagg_options.token_port += engine_override_args.disagg_options_token_port_offset
            if engine_override_args.disagg_options_inst_role:
                engine_args.disagg_options.inst_role = engine_override_args.disagg_options_inst_role
            if engine_override_args.engine_disagg_inst_id:
                engine_args.disagg_options.inst_id = engine_override_args.engine_disagg_inst_id
        return engine_args

    def get_engine_world_size(self):
        return self.world_size

    @dataclass
    class EngineOverrideArgs:
        # bladellm engine args need to override
        disagg_options_token_port_offset: int = field(default=None)
        disagg_options_inst_role: str = field(default=None)
        engine_disagg_inst_id: str = field(default=None)


def add_llumnix_cli_args(parser: LlumnixArgumentParser) -> LlumnixArgumentParser:
    parser.set_namespace("llumnix")
    parser = EntrypointsArgs.add_cli_args(parser)
    parser = ManagerArgs.add_cli_args(parser)
    parser = InstanceArgs.add_cli_args(parser)
    return parser

def detect_unsupported_engine_feature(engine_args) -> None:
    # pylint: disable=import-outside-toplevel
    from blade_llm.service.args import ServingArgs
    assert isinstance(engine_args, ServingArgs)
    unsupported_feature = None
    if engine_args.enable_lora:
        unsupported_feature = "multi-lora serving"
    elif engine_args.use_sps:
        unsupported_feature = "speculative decoding"
    elif engine_args.pipeline_parallel_size > 1:
        unsupported_feature = "pipeline parallel"
    elif engine_args.dp_attention:
        unsupported_feature = "attention data parallel"
    elif engine_args.elastic_attn_cluster:
        unsupported_feature = "elastic attention"

    if unsupported_feature:
        raise ValueError(f'Llumnix does not support "{unsupported_feature}" for BladeLLM currently.')

def get_args(llumnix_cfg: LlumnixConfig, launch_mode: LaunchMode, llumnix_parser: LlumnixArgumentParser, engine_args):
    # pylint: disable=import-outside-toplevel
    from blade_llm.service.args import ServingArgs
    assert isinstance(engine_args, ServingArgs)

    if not engine_args.serving_multi_processing_options.disable_frontend_multiprocessing:
        logger.warning("In llumnix, the api server and engine are in different ray actors, \
                       just set disable_frontend_multiprocessing to True.")
    engine_args.serving_multi_processing_options.disable_frontend_multiprocessing = True

    if not engine_args.disable_signal_handler:
        logger.warning("Disable the signal handler in BladeLLM, as the llumlet actor is not \
                       a process with a main function.")
    engine_args.disable_signal_handler = True

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
    assert engine_args.pipeline_parallel_size == 1 or not manager_args.enable_migration,\
        "Migration feature is temporarily unavailable for pipeline parallelism in BladeLLM."
    assert not(engine_args.enable_disagg and manager_args.enable_migration), "Migration feature is \
        temporarily unavailable for the engine based pd-disaggregation in BladeLLM."

    detect_unsupported_engine_feature(engine_args)

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args
