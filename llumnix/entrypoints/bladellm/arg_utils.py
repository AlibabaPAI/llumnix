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

from llumnix.arg_utils import (EntrypointsArgs, ManagerArgs, LlumnixArgumentParser,
                               InstanceArgs, LlumnixEngineArgs, get_llumnix_args,
                               load_registered_service_if_needed, print_args)
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode
from llumnix.logging.logger import init_logger
from llumnix.config import LlumnixConfig

logger = init_logger(__name__)


class BladellmEngineArgs(LlumnixEngineArgs):
    # The engine_args here has already been dumpped.
    def __init__(self, engine_args, backend_type: BackendType = BackendType.BLADELLM):
        super().__init__(
            engine_args=engine_args,
            backend_type=backend_type,
        )
        self.engine_args_wrapped = EngineArgsWrapped()
        self.world_size: int = None
        self.instance_id: str = None

    def unwrap_engine_args_if_needed(self):
        # pylint: disable=import-outside-toplevel
        from blade_llm.service.args import ServingArgs
        from blade_llm.utils.disagg_utils import DecodeRoutingPolicy
        engine_args: ServingArgs = pickle.loads(self.engine_args)
        assert isinstance(engine_args, ServingArgs)
        engine_args_wrapped: EngineArgsWrapped = self.engine_args_wrapped
        if engine_args.disagg_options:
            if engine_args_wrapped.disagg_options_token_port_offset:
                engine_args.disagg_options.token_port += engine_args_wrapped.disagg_options_token_port_offset
            if engine_args_wrapped.disagg_options_inst_role:
                engine_args.disagg_options.inst_role = engine_args_wrapped.disagg_options_inst_role
            if engine_args_wrapped.engine_disagg_inst_id:
                engine_args.disagg_options.inst_id = engine_args_wrapped.engine_disagg_inst_id
            engine_args.disagg_options.select_decode_policy = DecodeRoutingPolicy.EXTERNAL_ROUTE
        return engine_args

    def get_engine_world_size(self):
        return self.world_size


@dataclass
class EngineArgsWrapped:
    # bladellm engine args need to override
    disagg_options_token_port_offset: int = field(default=None)
    disagg_options_inst_role: str = field(default=None)
    engine_disagg_inst_id: str = field(default=None)


def add_cli_args(parser: LlumnixArgumentParser, add_engine_args: bool = True) -> LlumnixArgumentParser:
    # pylint: disable=import-outside-toplevel
    from blade_llm.service.args import add_args

    parser.set_namespace("llumnix")
    parser = EntrypointsArgs.add_cli_args(parser)
    parser = ManagerArgs.add_cli_args(parser)
    parser = InstanceArgs.add_cli_args(parser)
    if add_engine_args:
        parser.set_namespace("bladellm")
        parser = add_args(parser)

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

def check_engine_args(engine_args, manager_args) -> None:
    detect_unsupported_engine_feature(engine_args)

    assert not (engine_args.enable_disagg and manager_args.enable_pd_disagg), \
        "Cannot enable both pd-disaggregation inside the LLM engine and pd-disaggregation from Llumnix."

    assert engine_args.pipeline_parallel_size == 1 or not manager_args.enable_migration,\
        "Migration feature is temporarily unavailable for pipeline parallelism in BladeLLM."

    assert not (engine_args.enable_disagg and manager_args.enable_migration), \
        "Migration feature is temporarily unavailable for the engine based pd-disaggregation in BladeLLM."

    if not engine_args.serving_multi_processing_options.disable_frontend_multiprocessing:
        logger.warning("In llumnix, the api server and engine are in different ray actors, "
                       "just set disable_frontend_multiprocessing to True.")
        engine_args.serving_multi_processing_options.disable_frontend_multiprocessing = True

    if not engine_args.disable_signal_handler:
        logger.warning("Disable the signal handler in BladeLLM, as the llumlet actor is not "
                       "a process with a main function.")
        engine_args.disable_signal_handler = True

def check_instance_args(instance_args):
    assert not instance_args.simulator_mode, "Simulator mode is not supported for BladeLLM temporarily."
    assert 'W' not in instance_args.request_migration_policy, \
        "Migrating waiting request is not supported for BladeLLM temporarily."
    assert instance_args.migration_backend in ['grpc', 'kvtransfer'], \
        "Only support grpc and kvtransfer migration backend for BladeLLM."

def get_args(llumnix_config: LlumnixConfig, launch_mode: LaunchMode, llumnix_parser: LlumnixArgumentParser,
             cli_args: "Namespace" = None, engine_args = None):
    if launch_mode == LaunchMode.GLOBAL:
        # pylint: disable=import-outside-toplevel
        from blade_llm.service.args import ServingArgs
        engine_args: ServingArgs = ServingArgs.from_cli_args(cli_args)
    entrypoints_args, manager_args, instance_args = \
        get_llumnix_args(engine_args, BackendType.BLADELLM, launch_mode, llumnix_parser, llumnix_config)

    engine_args = load_registered_service_if_needed(manager_args, engine_args)

    if launch_mode == LaunchMode.LOCAL:
        # pylint: disable=import-outside-toplevel
        from blade_llm.service.server import check_ports
        check_ports(engine_args)

    # backend related check args
    check_instance_args(instance_args)
    if not manager_args.load_registered_service:
        check_engine_args(engine_args, manager_args)

    print_args(entrypoints_args, manager_args, instance_args, engine_args)

    return entrypoints_args, manager_args, instance_args, engine_args
