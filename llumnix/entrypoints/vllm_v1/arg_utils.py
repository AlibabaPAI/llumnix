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
import pickle
from dataclasses import dataclass, field

from llumnix.logging.logger import init_logger
from llumnix.utils import BackendType, LaunchMode
from llumnix.arg_utils import (
    VLLMV1EntrypointsArgs,
    ManagerArgs,
    InstanceArgs,
    LlumnixEngineArgsFactory,
    LlumnixArgumentParser,
    LlumnixEngineArgs,
    load_registered_engine_args,
    init_llumnix_args_v1,
    post_init_llumnix_args
)
from llumnix.config import LlumnixConfig
from llumnix.backends.output_forwarder import RequestOutputForwardingMode

logger = init_logger(__name__)


class VLLMV1EngineArgsFactory(LlumnixEngineArgsFactory):
    # pylint: disable=unused-argument
    def gen_next_engine_args(
        self,
        current_engine_args: LlumnixEngineArgs,
        next_instance_args: InstanceArgs,
        port_offset: int = 0,
        unit_id: str = None,
    ) -> LlumnixEngineArgs:
        if self.load_registered_service:
            instance_type = next_instance_args.instance_type
            current_engine_args = self.engine_args_dict[instance_type]

        next_engine_args = copy.deepcopy(current_engine_args)
        next_engine_args.revised_args.kvt_inst_id = unit_id

        return next_engine_args


class VLLMV1EngineArgs(LlumnixEngineArgs):
    def __init__(self,
                 engine_args: "AsyncEngineArgs",
                 backend_type: BackendType = BackendType.VLLM_V1) -> None:
        self.world_size = self._get_world_size(engine_args)
        self.dp_size = self._get_dp_size(engine_args)
        self.dp_size_local = self._get_dp_size_local(engine_args)
        super().__init__(
            engine_args=self._get_engine_args(engine_args),
            backend_type=backend_type
        )

        self.revised_args = RevisedArgs()

    def _get_world_size(self, engine_args: "AsyncEngineArgs"):
        world_size = engine_args.pipeline_parallel_size * engine_args.tensor_parallel_size
        return world_size

    def _get_dp_size(self, engine_args: "AsyncEngineArgs"):
        dp_size = engine_args.data_parallel_size
        return dp_size

    def _get_dp_size_local(self, engine_args: "AsyncEngineArgs"):
        dp_size_local = engine_args.data_parallel_size_local
        if dp_size_local is None:
            dp_size_local = engine_args.data_parallel_size
        return dp_size_local

    def _get_engine_args(self, engine_args: "AsyncEngineArgs"):
        return pickle.dumps(engine_args)

    def load_engine_args(self):
        # pylint: disable=import-outside-toplevel
        from vllm.engine.arg_utils import AsyncEngineArgs
        engine_args: AsyncEngineArgs = pickle.loads(self.engine_args)
        if self.revised_args.kvt_inst_id:
            engine_args.kv_transfer_config.kv_connector_extra_config["kvt_inst_id"] = self.revised_args.kvt_inst_id
        return engine_args

    def get_world_size(self):
        return self.world_size

    def get_dp_size(self):
        return self.dp_size

    def get_dp_size_local(self):
        return self.dp_size_local


@dataclass
class RevisedArgs:
    # bladellm engine args need to revised
    kvt_inst_id: str = field(default=None)
    kv_port: int = field(default=None)
    rpc_port: int = field(default=None)



def add_cli_args(parser: LlumnixArgumentParser) -> LlumnixArgumentParser:
    parser.set_namespace("llumnix")
    parser = VLLMV1EntrypointsArgs.add_cli_args(parser)
    parser = ManagerArgs.add_cli_args(parser)
    parser = InstanceArgs.add_cli_args(parser)
    parser.set_namespace("vllm")
    # pylint: disable=import-outside-toplevel
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    parser = make_arg_parser(parser)

    return parser

def add_engine_cli_args(parser: "ArgumentParser") -> "Namespace":
    # pylint: disable=import-outside-toplevel
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    parser = make_arg_parser(parser)

    return parser

def detect_unsupported_engine_feature(engine_args: "EngineArgs") -> None:
    unsupported_feature = None
    if engine_args.pipeline_parallel_size > 1:
        unsupported_feature = "pipeline parallel"

    if unsupported_feature:
        raise ValueError(f'Unsupported feature: Llumnix does not support "{unsupported_feature}" currently.')

def check_engine_args(engine_args: "AsyncEngineArgs") -> None:
    detect_unsupported_engine_feature(engine_args)

def check_instance_args(instance_args: InstanceArgs) -> None:
    assert not instance_args.request_output_forwarding_mode == RequestOutputForwardingMode.ACTOR, \
        "Llumnix does not support actor request output forwarding mode for vLLM v1 temporalily."

    assert not instance_args.simulator_mode, "Llumnix does not support simulator mode for vLLM v1."

def check_envs():
    # pylint: disable=import-outside-toplevel
    import vllm.envs as vllm_env

    assert len(vllm_env.VLLM_HOST_IP) == 0, "For Llumnix, please set VLLM_HOST_IP to empty string."

def get_args(llumnix_config: LlumnixConfig, launch_mode: LaunchMode, parser: LlumnixArgumentParser, cli_args: "Namespace") \
        -> Tuple[VLLMV1EntrypointsArgs, ManagerArgs, InstanceArgs, "AsyncEngineArgs"]:
    # pylint: disable=import-outside-toplevel
    from vllm.engine.arg_utils import AsyncEngineArgs

    entrypoints_args, manager_args, instance_args = init_llumnix_args_v1(llumnix_config)
    if manager_args.load_registered_service:
        engine_args_list = load_registered_engine_args(manager_args)
        # NOTE(s5u13b): hack to post init llumnix args and check args.
        engine_args = engine_args_list[0]
    else:
        engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    post_init_llumnix_args(engine_args, instance_args, manager_args, entrypoints_args, BackendType.VLLM_V1, launch_mode, parser)

    # backend related check args
    check_engine_args(engine_args)
    check_instance_args(instance_args)
    check_envs()

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args

def get_engine_args(cli_args: "Namespace") -> "AsyncEngineArgs":
    # pylint: disable=import-outside-toplevel
    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    check_engine_args(engine_args)
    logger.info("engine_args: {}".format(engine_args))

    return engine_args
