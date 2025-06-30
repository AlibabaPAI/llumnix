# pylint: disable=no-value-for-parameter

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

import argparse
import dataclasses
from dataclasses import dataclass
import os
import pickle
from typing import List, Tuple, Union, Dict
from abc import ABC, abstractmethod

from llumnix.internal_config import GlobalSchedulerConfig, MigrationConfig, PDDConfig
from llumnix.config import LlumnixConfig, get_llumnix_config
from llumnix.config.default import _C
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


# All the default values of llumnix arguments are set in default.py. So all the arguments here are set to None for default.

class LlumnixArgumentParser(argparse.ArgumentParser):
    _deprecated: set[argparse.Action] = set()

    def __init__(self, *args, **kwargs):
        self.cur_namespace = "llumnix"
        super().__init__(*args, **kwargs)

    def set_namespace(self, namespace: str):
        self.cur_namespace = namespace

    def add_argument(self, *args, **kwargs):
        # Compatible with vllm v0.9.0 FlexibleArgumentParser
        deprecated = kwargs.pop("deprecated", False)
        if self.cur_namespace == 'llumnix' and "--help" not in args:
            assert 'default' not in kwargs or kwargs['default'] is None, \
                f"Do not set the default value for '{args[0]}' in CLI, or set default value to None. " \
                f"The default value will be retrieved from config/default.py in get_llumnix_config."
            if kwargs.get('action') == 'store_true':
                kwargs['default'] = None
        action = super().add_argument(*args, **kwargs)
        if deprecated:
            self._deprecated.add(action)
        return action

    class _LlumnixArgumentGroup(argparse._ArgumentGroup):  # pylint: disable=protected-access
        def add_argument(self, *args, **kwargs):
            # Compatible with vllm v0.9.0 FlexibleArgumentParser
            deprecated = kwargs.pop("deprecated", False)
            action = super().add_argument(*args, **kwargs)
            if deprecated:
                LlumnixArgumentParser._deprecated.add(action)  # pylint: disable=protected-access
            return action

    def add_argument_group(self, *args, **kwargs):
        group = self._LlumnixArgumentGroup(self, *args, **kwargs)
        return group


class LlumnixEngineArgs(ABC):

    def __init__(
        self, engine_args, backend_type: BackendType
    ) -> None:
        self.engine_args = engine_args
        self.revised_args = None
        self.backend_type: BackendType = backend_type

    @abstractmethod
    def _get_engine_args(self, engine_args):
        raise NotImplementedError

    @abstractmethod
    def load_engine_args(self):
        # returun the engine args after overriding
        raise NotImplementedError

    @abstractmethod
    def get_world_size(self):
        raise NotImplementedError

    def update_arg(self, args_key: str, args_value):
        if self.revised_args and hasattr(self.revised_args, args_key):
            setattr(self.revised_args, args_key, args_value)

    def update_args(self, **kwargs):
        for args_key, args_value in kwargs.items():
            self.update_arg(args_key, args_value)


class LlumnixEngineArgsFactory:
    def __init__(
        self,
        load_registered_service: bool,
        enable_port_increment: bool,
        load_registered_service_path: str,
        pdd_config: PDDConfig,
    ) -> None:
        self.load_registered_service: bool = load_registered_service
        self.load_registered_service_path: str = load_registered_service_path
        self.pdd_config: PDDConfig = pdd_config
        self.engine_args_dict: Dict[str, LlumnixEngineArgs] = {}
        self.enable_port_increment: bool = enable_port_increment

        if self.load_registered_service:
            if (
                not self.pdd_config.enable_pd_disagg
                and not self.pdd_config.enable_engine_pd_disagg
            ):
                instance_type_list = ["no_constraints"]
            else:
                instance_type_list = ["prefill", "decode"]
            for instance_type in instance_type_list:
                self.engine_args_dict[instance_type] = load_engine_args(
                    instance_type, self.load_registered_service_path
                )

    def gen_next_engine_args(
        self, backend_type: BackendType, current_engine_args: LlumnixEngineArgs, instance_type: Union[str, 'InstanceType']
    ) -> LlumnixEngineArgs:
        if self.load_registered_service:
            current_engine_args = self.engine_args_dict[instance_type]

        if backend_type == BackendType.BLADELLM:
            # pylint: disable=import-outside-toplevel
            from llumnix.entrypoints.bladellm.arg_utils import BladeLLMEngineArgs
            from llumnix.instance_info import InstanceType

            next_engine_args = BladeLLMEngineArgs(current_engine_args)
            if self.pdd_config.enable_engine_pd_disagg and not self.load_registered_service:
                next_engine_args.revised_args.disagg_options_inst_role = (
                    instance_type.value if isinstance(instance_type, InstanceType)
                    else instance_type
                )
            return next_engine_args

        if backend_type in [BackendType.VLLM, BackendType.SIM_VLLM]:
            # pylint: disable=import-outside-toplevel
            from llumnix.entrypoints.vllm.arg_utils import VLLMEngineArgs
            next_engine_args = VLLMEngineArgs(current_engine_args, current_engine_args.backend_type)
            return next_engine_args

        if backend_type == BackendType.VLLM_V1:
            # pylint: disable=import-outside-toplevel
            from llumnix.entrypoints.vllm_v1.arg_utils import VLLMV1EngineArgs
            next_engine_args = VLLMV1EngineArgs(current_engine_args, current_engine_args.backend_type)
            return next_engine_args

        raise TypeError("Unsupported engine args type when generating next engine args")


def ensure_args_default_none(args):
    # Check if all fields default to None
    for field_info in dataclasses.fields(args):
        if field_info.default is not None:
            raise ValueError(f"The default value of '{field_info.name}' should be None")

def init_from_default_args_config(args, args_config):
    for attr in dataclasses.fields(args):
        if getattr(args, attr.name) is None:
            if hasattr(args_config, attr.name.upper()):
                setattr(args, attr.name, getattr(args_config, attr.name.upper()))

def from_llumnix_args_config(cls, args_config):
    # Get the list of attributes of this dataclass.
    attrs = [attr.name for attr in dataclasses.fields(cls)]
    cfg_attrs = [attr for attr in attrs if hasattr(args_config, attr.upper())]
    # Set the attributes from the parsed arguments.
    # The defalut values of attributes are defined in default.py.
    args = cls(**{attr: getattr(args_config, attr.upper()) for attr in cfg_attrs})

    return args

def check_args_choices(args, parser):
    # pylint: disable=protected-access
    for action in parser._optionals._actions:
        if hasattr(action, 'choices') and action.choices is not None and hasattr(args, action.dest):
            cur_arg = getattr(args, action.dest)
            assert cur_arg in action.choices, f"{action.dest} should be one of {action.choices}, but {cur_arg} is set."


@dataclass
class EntrypointsArgs:
    host: str = None
    port: int = None
    ssl_keyfile: str = None
    ssl_certfile: str = None
    server_log_level: str = None
    launch_ray_cluster: bool = None
    ray_cluster_port: int = None
    disable_log_to_driver: bool = None
    request_output_queue_type: str = None
    disable_log_requests_server: bool = None
    log_request_timestamps: bool = None
    config_file: str = None
    disable_keep_serve_process_alive: bool = None

    def __post_init__(self):
        ensure_args_default_none(self)
        init_from_default_args_config(self, _C.SERVER)

    @classmethod
    def from_llumnix_config(cls, cfg: LlumnixConfig = get_llumnix_config()) -> 'EntrypointsArgs':
        entrypoint_args = from_llumnix_args_config(cls, cfg.SERVER)
        return entrypoint_args

    @classmethod
    def check_args(cls, args: 'EntrypointsArgs', parser: argparse.ArgumentParser) -> None:
        check_args_choices(args, parser)

    def init_from_engine_args(self, engine_args, backend_type: BackendType):
        if backend_type == BackendType.BLADELLM:
            self.host = engine_args.host
            self.port = engine_args.port

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # TODO(KuilongCui): add a checker to ensure that for arguments configured with store_true has its default value
        # set to False in the configuration settings, otherwise, it cannot be set to False.
        parser.add_argument('--launch-ray-cluster',
                            action='store_true',
                            help='if launch ray cluster')
        parser.add_argument("--ray-cluster-port",
                            type=int,
                            help='ray cluster port')
        parser.add_argument('--disable-log-to-driver',
                            action='store_true',
                            help='disable redirecting all worker logs to driver')
        parser.add_argument("--request-output-queue-type",
                            type=str,
                            choices=['rayqueue', 'zmq'],
                            help='queue type for request output queue')
        parser.add_argument('--disable-log-requests-server',
                            action='store_true',
                            help='disable logging requests in server')
        parser.add_argument("--log-request-timestamps",
                            action='store_true',
                            help='if log request timestamps')
        parser.add_argument("--config-file",
                            type=str,
                            help="path to config file of arguments")
        return parser


@dataclass
class VLLMV1EntrypointsArgs(EntrypointsArgs):
    ssl_ca_certs: str = None
    ssl_cert_reqs: int = None
    allowed_origins: List[str] = None
    allow_credentials: bool = None
    allowed_methods: List[str] = None
    allowed_headers: List[str] = None
    api_server_count: int = None
    tool_call_parser: str = None
    tool_parser_plugin: str = None
    log_config_file: str = None
    root_path: str = None
    disable_fastapi_docs: bool = None
    api_key: str = None
    enable_request_id_headers: bool = None
    middleware: List[str] = None
    enable_ssl_refresh: bool = None
    uvicorn_log_level: str = None
    disable_uvicorn_access_log: bool = None
    disable_frontend_multiprocessing: bool = None
    max_log_len: int = None
    chat_template: str = None
    lora_modules: str = None
    prompt_adapters: str = None
    enable_auto_tool_choice: bool = None
    enable_prompt_tokens_details: bool = None
    return_tokens_as_token_ids: bool = None
    response_role: str = None
    chat_template_content_format: str = None
    enable_server_load_tracking: bool = None

    def __post_init__(self):
        ensure_args_default_none(self)
        init_from_default_args_config(self, _C.SERVER)

    @classmethod
    def from_llumnix_config(cls, cfg: LlumnixConfig = get_llumnix_config()) -> 'VLLMV1EntrypointsArgs':
        v1_entrypoint_args = from_llumnix_args_config(cls, cfg.SERVER)
        return v1_entrypoint_args

    @classmethod
    def check_args(cls, args: 'VLLMV1EntrypointsArgs', parser: argparse.ArgumentParser) -> None:
        check_args_choices(args, parser)

    def init_from_engine_args(self, engine_args, backend_type: BackendType):
        pass

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        parser = make_arg_parser(parser)

        parser = EntrypointsArgs.add_cli_args(parser)

        return parser


@dataclass
class ManagerArgs:
    initial_instances: int = None

    polling_interval: float = None
    dispatch_policy: str = None
    scaling_load_metric: str = None
    topk_random_dispatch: int = None

    enable_migration: bool = None
    pair_migration_frequency: int = None
    pair_migration_policy: str = None
    migrate_out_threshold: float = None

    enable_scaling: bool = None
    min_instances: int = None
    max_instances: int = None
    scaling_interval: int = None
    scaling_policy: str = None
    scale_up_threshold: float = None
    scale_down_threshold: float = None

    log_instance_info: bool = None
    log_filename: str = None
    enable_port_increment: bool = None
    enable_port_offset_store: bool = None

    enable_adaptive_pd: bool = None
    enable_pd_disagg: bool = None
    pd_ratio: Union[str, List[int]] = None
    load_registered_service: bool = None
    load_registered_service_path: str = None
    enable_pdd_node_affinity_scheduling: bool = None

    # init from instance args
    is_group_kind_migration_backend: bool = None
    enable_engine_pd_disagg: bool = None

    def __post_init__(self):
        ensure_args_default_none(self)
        init_from_default_args_config(self, _C.MANAGER)

        def parse_ratio(ratio_str):
            parts = ratio_str.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid format for --pd-ratio : '{ratio_str}'. Expected format 'a:b'.")
            num_prefill_instances, num_decode_instances = int(parts[0].strip()), int(parts[1].strip())
            assert num_prefill_instances > 0 and num_decode_instances > 0, "Both parts of --pd-ratio must be non-negative."
            return [num_prefill_instances, num_decode_instances]

        self.pd_ratio = parse_ratio(self.pd_ratio)

    @classmethod
    def from_llumnix_config(cls, cfg: LlumnixConfig = get_llumnix_config()) -> 'ManagerArgs':
        return from_llumnix_args_config(cls, cfg.MANAGER)

    @classmethod
    def check_args(cls, args: 'ManagerArgs', launch_mode: LaunchMode, parser: argparse.ArgumentParser) -> None:
        check_args_choices(args, parser)

        assert not args.enable_port_offset_store or args.enable_port_increment, \
            "Set enable_port_increment when enable_port_offset_store"

        assert not args.enable_scaling, "Proactive auto-scaling is deprecated now, " \
            "all auto-scaling related args will not take effects."

        if args.load_registered_service:
            assert args.load_registered_service_path and launch_mode == LaunchMode.GLOBAL, \
            "Only load registered service when enabling pd-disaggregation in global launch mode, " \
            "and the path of loading registered service is required to be specified when loading registered service from path."

        if args.enable_pdd_node_affinity_scheduling:
            assert (args.enable_pd_disagg or args.enable_engine_pd_disagg) and launch_mode == LaunchMode.GLOBAL, \
                "Prefill-decode disaggregation node affinity scheduling can only be used when enabling prefill-decode disaggregation " \
                "in global launch mode."

        if args.enable_pd_disagg:
            assert args.enable_migration, "Migration must be enabled when enabling prefill-decode disaggregation (not engine-based)."

        if args.enable_adaptive_pd:
            assert args.enable_pd_disagg or args.enable_engine_pd_disagg, \
                "Adaptive prefill-decode disaggregation is only supported when prefill-decode disaggregation, " \
                "set --enable-pd-disagg or --enable-engine-pd-disagg to enable prefill-decode disaggregation."

        assert not (args.enable_engine_pd_disagg and args.enable_adaptive_pd), "Adaptive prefill-decode disaggregation \
            is not supported when engine-based prefill-decode disaggregation is enabled."

        assert not (args.enable_engine_pd_disagg and args.enable_pd_disagg), "Engine-based prefill-decode disaggregation and \
            Llumnix-based prefill-decode disaggregation are mutually exclusive."

    def init_from_instance_args(self, instance_args: 'InstanceArgs'):
        self.is_group_kind_migration_backend = instance_args.migration_backend in ['gloo', 'nccl']

    def create_global_scheduler_config(self) -> Tuple[GlobalSchedulerConfig]:
        # Create the GlobalScheduler Configuration.
        global_scheduler_config = GlobalSchedulerConfig(self.initial_instances,
                                                        self.dispatch_policy,
                                                        self.topk_random_dispatch,
                                                        self.pair_migration_policy,
                                                        self.migrate_out_threshold,
                                                        self.scaling_policy,
                                                        self.scaling_load_metric,
                                                        self.scale_up_threshold,
                                                        self.scale_down_threshold,
                                                        self.enable_pd_disagg,
                                                        self.enable_engine_pd_disagg,
                                                        self.enable_adaptive_pd,
                                                        self.is_group_kind_migration_backend)
        return global_scheduler_config

    def create_pdd_config(self) -> PDDConfig:
        pdd_config = PDDConfig(self.enable_pd_disagg,
                               self.enable_engine_pd_disagg,
                               self.pd_ratio,
                               self.enable_pdd_node_affinity_scheduling)
        return pdd_config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--initial-instances',
                            type=int,
                            help='number of instances created at initialzation')
        parser.add_argument('--polling-interval',
                            type=float,
                            help='time interval(s) to update instance info and pair migration')
        parser.add_argument('--scaling-load-metric',
                            type=str,
                            choices=['remaining_steps', 'kv_blocks_ratio'],
                            help='instance scaling load metric')
        parser.add_argument('--dispatch-policy',
                            type=str,
                            choices=['balanced', 'load', 'queue', 'flood', 'rr'],
                            help='The request dispatch policy.\n\n'
                            '* "balanced" dispatch request to the instance with minimum requests dispatched.\n'
                            '* "load" dispatch request to the instance with lowest instance load.\n'
                            '* "queue" dispatch request to the instance with minimum waiting request queue length.\n'
                            '* "flood" dispatch request to the instance with maximum requests dispatched.\n'
                            '* "rr" dispatch requests with round-robin policy.\n')
        parser.add_argument('--topk-random-dispatch',
                            type=int,
                            help='number of candidate random dispatch instances for dispatch policy.\n\n'
                            'The candidate instances are first selected according to the load'
                            '(including factors such as load, queue size, etc.) based on the dispatch policy,'
                            'and then one of them is randomly chosen to receive the request for better load balancing.')
        parser.add_argument('--enable-migration',
                            action='store_true',
                            help='enable migrate requests between instances')
        parser.add_argument('--pair-migration-frequency',
                            type=int,
                            help='pair migration frequency')
        parser.add_argument('--pair-migration-policy',
                            type=str,
                            choices=['balanced', 'defrag'],
                            help='The pair migration policy.\n\n'
                            '* "balanced" pair migration to make the instance load of instance more balanced.\n'
                            '* "defrag" pair migration without balanced constraint to '
                            'achieve defragmentation thoroughly (with instance constraints).\n')
        parser.add_argument('--migrate-out-threshold',
                            type=float,
                            help='migrate out instance load threshold')
        parser.add_argument('--enable-scaling',
                            action='store_true',
                            help='enable auto scaling')
        parser.add_argument('--min-instances',
                            type=int,
                            help='minimum number of instances')
        parser.add_argument('--max-instances',
                            type=int,
                            help='maximum number of instances')
        parser.add_argument('--scaling-interval',
                            type=int,
                            help='interval time of check scaling')
        parser.add_argument('--scaling-policy',
                            type=str,
                            choices=['max_load', 'avg_load'],
                            help='scaling policy')
        parser.add_argument('--scale-up-threshold',
                            type=float,
                            help='scale up threshold')
        parser.add_argument('--scale-down-threshold',
                            type=float,
                            help='scale down threshold')
        parser.add_argument('--log-instance-info',
                            action='store_true',
                            help='enable logging instance info')
        parser.add_argument('--log-filename',
                            type=str,
                            help='log filename')
        parser.add_argument('--enable-port-increment',
                            action='store_true',
                            help='enable port increment when desploying multiple servers')
        parser.add_argument('--enable-port-offset-store',
                            action='store_true',
                            help='enable store port offset when desploying multiple servers')
        parser.add_argument('--enable-pd-disagg',
                            action='store_true',
                            help='enable prefill-decode disaggregation')
        parser.add_argument('--enable-adaptive-pd',
                            action='store_true',
                            help='[Experimental] enable adaptive prefill-decode disaggregation')
        parser.add_argument('--enable-engine-pd-disagg',
                            action='store_true',
                            help='enable engine-based prefill-decode disaggregation')
        parser.add_argument('--pd-ratio',
                            type=str,
                            help='the prefill decode ratio used in gloabl launch mode e.g. "1:1"')
        parser.add_argument('--load-registered-service',
                            action='store_true',
                            help="load registered service.\n When this argument is passed, "
                                 "--load-registered-service-path is required at the same time.")
        parser.add_argument('--load-registered-service-path',
                            type=str,
                            help="path of loading registered service.\n"
                                 "Registered service is generated by running register_service.py, "
                                 "Currently the register_service.py is mainly used to saving the engine arguments of engine, "
                                 "the save path of engine arguments file is decided by the save-path "
                                 "and save-key aruguments of register_service.py. "
                                 "You can specify the load path through this argument, and registered service (all the engine arguments "
                                 "files) under this path will be loaded. The Llumnix will initialize instance based on "
                                 "the engine type (no_constraints, prefill, decode) and the corresponding engine arguments "
                                 "loaded from the path.")
        parser.add_argument('--enable-pdd-node-affinity-scheduling',
                            action='store_true',
                            help="Enable prefill-decode disaggregation (abbreviated as PDD) node affinity scheduling.\n "
                                 "For PDD ray cluster, each node can be annotated with prefill/decode gpu resources. "
                                 "When enabling PDD node affinity scheduling, Llumnix will schedule prefill/decode instance to "
                                 "the node with correspoinding prefill/decode gpu resources.")
        return parser


@dataclass
class LaunchArgs:
    launch_mode: LaunchMode = None
    backend_type: BackendType = None


@dataclass
class InstanceArgs:
    instance_type: str = None

    simulator_mode: bool = None
    profiling_result_file_path: str = None

    dispatch_load_metric: str = None
    dispatch_prefill_load_metric: str = None
    dispatch_prefill_as_decode_load_metric: str = None
    dispatch_decode_load_metric: str = None
    dispatch_decode_as_prefill_load_metric: str = None
    migration_load_metric: str = None
    enable_defrag: bool = None
    request_migration_policy: str = None

    migration_backend: str = None
    migration_buffer_blocks: int = None
    migration_num_layers: int = None
    migration_backend_init_timeout: float = None
    kvtransfer_migration_backend_transfer_type: str = None
    kvtransfer_migration_backend_naming_url: str = None
    migration_last_stage_max_blocks: int = None
    migration_max_stages: int = None
    engine_disagg_inst_id_env_var: str = None

    request_output_forwarding_mode: str = None

    # init from engine args
    enable_engine_pd_disagg: bool = None

    # init from manager args
    enable_migration: bool = None
    enable_adaptive_pd: bool = None

    def __post_init__(self):
        ensure_args_default_none(self)
        init_from_default_args_config(self, _C.INSTANCE)

    @classmethod
    def from_llumnix_config(cls, cfg: LlumnixConfig = get_llumnix_config()) -> 'InstanceArgs':
        return from_llumnix_args_config(cls, cfg.INSTANCE)

    @classmethod
    def check_args(cls, args: 'InstanceArgs', manager_args: ManagerArgs,
                   launch_mode: LaunchMode, parser: argparse.ArgumentParser) -> None:
        check_args_choices(args, parser)

        assert not args.simulator_mode or args.profiling_result_file_path is not None, \
            "Set profiling_result_file_path args when enable simulator mode"

        # instance_type check
        if manager_args.enable_pd_disagg and launch_mode == LaunchMode.LOCAL:
            assert args.instance_type in ['prefill', 'decode'], \
                "instance_type should be prefill or decode if enable_pd_disagg is set."

    def init_from_engine_args(self, engine_args, backend_type: BackendType):
        if backend_type == BackendType.BLADELLM:
            self.enable_engine_pd_disagg = engine_args.enable_disagg
            # for local launch mode
            if self.enable_engine_pd_disagg:
                self.instance_type = engine_args.disagg_options.inst_role
        else:
            self.enable_engine_pd_disagg = False

    def init_from_manager_args(self, manager_args: ManagerArgs):
        self.enable_migration = manager_args.enable_migration
        self.enable_adaptive_pd = manager_args.enable_adaptive_pd

    def create_migration_config(self) -> MigrationConfig:
        migration_config = MigrationConfig(self.enable_migration,
                                           self.request_migration_policy,
                                           self.migration_backend,
                                           self.migration_buffer_blocks,
                                           self.migration_num_layers,
                                           self.migration_last_stage_max_blocks,
                                           self.migration_max_stages,
                                           self.migration_backend_init_timeout,
                                           self.kvtransfer_migration_backend_transfer_type,
                                           self.kvtransfer_migration_backend_naming_url)
        return migration_config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--instance-type',
                            type=str,
                            choices=['prefill', 'decode', 'no_constraints'],
                            help="instance type of the engine.\n When not setting --enable-pd-disagg, set --instance-type \
                                to no_constraints.\n When setting --enable-pd-disagg, pd-disaggregation is fully \
                                (launch instance + migrate kv cache) implemented via LLuminx (vLLM). In local launch mode, \
                                set --instance-type as either prefill or decode. In global launch mode, do not set \
                                --instance-type as manager will automatically determine the type and number of instance.\n \
                                For engine specifying pd-disaggregation by its own and migrating kv cache internally (BladeLLM), \
                                don't set --enable-pd-disagg. Llumnix will decide if enabling engine based pd-disaggregation by \
                                checking engine arguments, Engine based pd-disaggregation means that pd-disaggregation is \
                                partially implemented via Llumnix (launch instance), and partially implemented by engine \
                                (migrate kv cache). In local launch mode, the instance type will be assigned according to \
                                engine arguments. In global launch mode, do not set --instance-type as explained above.")
        parser.add_argument('--profiling-result-file-path',
                            type=str,
                            help='profiling result file path when using simulator')
        parser.add_argument('--simulator-mode',
                            action='store_true',
                            help='enable simulator mode')
        parser.add_argument('--dispatch-load-metric',
                            type=str,
                            choices=['remaining_steps', 'kv_blocks_ratio'],
                            help='instance dispatch load metric.\n\n'
                            '* "remaining_steps" refers to the number of steps the remaining KV cache can support for all \
                            running and some waiting requests to proceed.\n'
                            '* "kv_blocks_ratio" refers to the total number of KV cache blocks required by all running \
                            and waiting requests.\n'
                            )
        parser.add_argument('--dispatch-prefill-load-metric',
                            type=str,
                            choices=['remaining_steps', 'kv_blocks_ratio'],
                            help='prefill instance dispatch load metric')
        parser.add_argument('--dispatch-prefill-as-decode-load-metric',
                            type=str,
                            choices=['remaining_steps', 'kv_blocks_ratio', 'adaptive_decode'],
                            help='[Experimental] prefill instance dispatch load metric when decoding')
        parser.add_argument('--dispatch-decode-load-metric',
                            type=str,
                            choices=['remaining_steps', 'kv_blocks_ratio'],
                            help='decode instance dispatch load metric')
        parser.add_argument('--dispatch-decode-as-prefill-load-metric',
                            type=str,
                            choices=['remaining_steps', 'kv_blocks_ratio'],
                            help='[Experimental] decode instance dispatch load metric when prefilling')
        parser.add_argument('--migration-load-metric',
                            type=str,
                            choices=['remaining_steps', 'kv_blocks_ratio'],
                            help='instance migration load metric')
        parser.add_argument('--enable-defrag',
                            type=bool,
                            help='enable defragmentation through migration based on virtual usage')
        parser.add_argument('--request-migration-policy',
                            type=str,
                            default=None,
                            choices=['LCR', 'SR', 'LR', 'FCW', 'FCWSR'],
                            help='The request migration policy.\n\n'
                            '* "LCR" migrate the running request last come.\n'
                            '* "SR" migrate the running request shortest.\n'
                            '* "LR" migrate the running request longest.\n'
                            '* "FCW" migrate the waiting request first come.\n'
                            '* "FCWSR" migrate the waiting request first come and running request shortest.\n')
        parser.add_argument('--migration-backend',
                            type=str,
                            choices=['gloo','nccl','rayrpc','grpc','kvtransfer'],
                            help='communication backend of migration, [gloo, rayrpc, nccl] are available for vllm \
                                and [grpc, kvtransfer] are available for bladellm')
        parser.add_argument('--migration-buffer-blocks',
                            type=int,
                            help='number of buffer blocks in migration')
        parser.add_argument('--migration-num-layers',
                            type=int,
                            help='number of kv-cache layers to transfer in each round during migration')
        parser.add_argument('--migration-backend-init-timeout',
                            type=float,
                            help='timeout(s) for initializing migration backend')
        parser.add_argument('--migration-backend-transfer-type',
                            type=str,
                            choices=['ipc','rdma'],
                            help='transfer type for migration backend kvTransfer')
        parser.add_argument('--grpc-migration-backend-server-port',
                            type=int,
                            help='port of grpc server for migration backend')
        parser.add_argument('--kvtransfer-migration-backend-naming-url',
                            type=str,
                            help='url of naming server for kvtransfer migration backend')
        parser.add_argument('--migration-max-stages',
                            type=int,
                            help='drop migration if the number of stages > migration_max_stages')
        parser.add_argument('--migration-last-stage-max-blocks',
                            type=int,
                            help='if the number pf remain blocks < migration_last_stage_max_blocks, do last stage migration')
        parser.add_argument('--engine-disagg-inst-id-env-var',
                            type=str,
                            help='environment variable used as engine instance id')
        parser.add_argument('--request-output-forwarding-mode',
                            type=str,
                            choices=["actor", "thread"],
                            help='mode of forwarding request output. When in actor/thread mode, request outputs generated by '
                                 'engine step are fowarded to a seperate actor/thread, and the seperate actor/thread transfers the '
                                 'request outputs to api servers using zmq.')
        return parser


@dataclass
class RegisterServiceArgs:
    engine_type: str = None
    save_key: str = None
    save_path: str = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--engine-type',
                            type=str,
                            choices=['prefill', 'decode', 'no_constraints'],
                            default='no_constraints',
                            help="Engine type of the engine arguments. The actual save filename is generated according to "
                                "the engine type, following the format f\"engine_args_{engine_type}.pkl\".")
        parser.add_argument('--save-key',
                            type=str,
                            help="Save key of the engine arguments. The actual save filepath is generated according to "
                                "the save path and save key, following the organization f\"{save_path}/{save_key}/\".")
        parser.add_argument('--save-path',
                            type=str,
                            default='.',
                            help="Save path of the engine arguments.")
        return parser


def init_llumnix_args(llumnix_config: LlumnixConfig):
    instance_args = InstanceArgs.from_llumnix_config(llumnix_config)
    manager_args = ManagerArgs.from_llumnix_config(llumnix_config)
    entrypoints_args = EntrypointsArgs.from_llumnix_config(llumnix_config)

    return entrypoints_args, manager_args, instance_args

def init_llumnix_args_v1(llumnix_config: LlumnixConfig):
    instance_args = InstanceArgs.from_llumnix_config(llumnix_config)
    manager_args = ManagerArgs.from_llumnix_config(llumnix_config)
    entrypoints_args_v1 = VLLMV1EntrypointsArgs.from_llumnix_config(llumnix_config)

    return entrypoints_args_v1, manager_args, instance_args

def post_init_llumnix_args(
    engine_args,
    instance_args: InstanceArgs,
    manager_args: ManagerArgs,
    entrypoints_args: EntrypointsArgs,
    backend_type: BackendType,
    launch_mode: LaunchMode,
    parser: LlumnixArgumentParser
):
    # bottom-up to ensure the correctness of the args
    instance_args.init_from_engine_args(engine_args, backend_type)
    instance_args.init_from_manager_args(manager_args)
    manager_args.init_from_instance_args(instance_args)
    entrypoints_args.init_from_engine_args(engine_args, backend_type)

    EntrypointsArgs.check_args(entrypoints_args, parser)
    ManagerArgs.check_args(manager_args, launch_mode, parser)
    InstanceArgs.check_args(instance_args, manager_args, launch_mode, parser)

def load_registered_engine_args(manager_args: ManagerArgs):
    if not manager_args.enable_pd_disagg and not manager_args.enable_engine_pd_disagg:
        instance_type_list = ['no_constraints']
    else:
        instance_type_list = ['prefill', 'decode']
    for instance_type in instance_type_list:
        engine_args_registered: LlumnixEngineArgs = load_engine_args(instance_type, manager_args.load_registered_service_path)
        engine_args = engine_args_registered.load_engine_args()
    return engine_args

def _get_engine_args_filename(engine_type: str) -> str:
    return f"engine_args_{engine_type}.pkl"

def _get_engine_args_filepath(save_path: str, save_key: str = None) -> str:
    if save_key is not None:
        save_filepath = os.path.join(save_path, save_key)
    else:
        save_filepath = save_path
    return save_filepath

def save_engine_args(engine_type: str, save_path: str, engine_args: LlumnixEngineArgs, save_key: str = None) -> None:
    engine_args_filename = _get_engine_args_filename(engine_type)
    save_filepath = _get_engine_args_filepath(save_path, save_key)
    save_filename = os.path.join(save_filepath, engine_args_filename)
    os.makedirs(save_filepath, exist_ok=True)
    with open(save_filename, 'wb') as file:
        pickle.dump(engine_args, file)
    logger.info("Save engine arguments of {} engine type as file: {}".format(engine_type, save_filename))

def load_engine_args(engine_type: str, load_path: str) -> LlumnixEngineArgs:
    engine_args_filename = _get_engine_args_filename(engine_type)
    load_filename = os.path.join(load_path, engine_args_filename)
    with open(load_filename, 'rb') as file:
        engine_args =  pickle.load(file)
    logger.info("Load engine arguments of {} engine type from path: {}".format(engine_type, load_path))
    return engine_args
