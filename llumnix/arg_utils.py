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

import dataclasses
from dataclasses import dataclass
import argparse
from typing import Tuple

from llumnix.internal_config import GlobalSchedulerConfig, MigrationConfig
from llumnix.config import LlumnixConfig, get_llumnix_config
from llumnix.config.default import _C
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode


class LlumnixArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.cur_namespace = "llumnix"
        super().__init__(*args, **kwargs)

    def set_namespace(self, namespace: str):
        self.cur_namespace = namespace

    def add_argument(self, *args, **kwargs):
        if self.cur_namespace == 'llumnix' and "--help" not in args:
            assert 'default' not in kwargs or kwargs['default'] is None, \
                f"Do not set the default value for '{args[0]}' in CLI, or set default value to None. " \
                f"The default value will be retrieved from config/default.py in get_llumnix_config."
            if kwargs.get('action') == 'store_true':
                kwargs['default'] = None
        super().add_argument(*args, **kwargs)


# All the default values of llumnix arguments are set in default.py. So all the arguments here are set to None.

@dataclass
class EntrypointsArgs:
    host: str = None
    port: int = None
    ssl_keyfile: str = None
    ssl_certfile: str = None
    log_level: str = None
    launch_ray_cluster: bool = None
    ray_cluster_port: int = None
    disable_log_to_driver: bool = None
    request_output_queue_type: str = None
    request_output_queue_port: int = None
    disable_log_requests_server: bool = None
    log_request_timestamps: bool = None
    config_file: str = None
    disable_keep_serve_process_alive: bool = None

    def __post_init__(self):
        for attr in dataclasses.fields(self):
            if getattr(self, attr.name) is None:
                setattr(self, attr.name, getattr(_C.SERVER, attr.name.upper()))

    @classmethod
    def from_llumnix_config(cls, cfg: LlumnixConfig = get_llumnix_config()) -> 'EntrypointsArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        # The defalut values of attributes are defined in default.py.
        entrypoints_args = cls(**{attr: getattr(cfg.SERVER, attr.upper()) for attr in attrs})
        return entrypoints_args

    @classmethod
    def check_args(cls, args: 'EntrypointsArgs', parser: argparse.ArgumentParser):
        # pylint: disable=protected-access
        for action in parser._optionals._actions:
            if hasattr(action, 'choices') and action.choices is not None and hasattr(args, action.dest):
                assert getattr(args, action.dest) in action.choices, f"{action.dest} should be one of {action.choices}."

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
        parser.add_argument("--request-output-queue-port",
                            type=int,
                            help='port number for the zmq request output queue')
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
class ManagerArgs:
    initial_instances: int = None

    load_metric: str = None
    polling_interval: float = None

    dispatch_policy: str = None
    power_of_k_choice: int = None

    enable_migration: bool = None
    enable_defrag: bool = None
    pair_migration_frequency: int = None
    pair_migration_policy: str = None
    migrate_out_threshold: float = None
    request_migration_policy: str = None

    enable_scaling: bool = None
    min_instances: int = None
    max_instances: int = None
    scaling_interval: int = None
    scaling_policy: str = None
    scale_up_threshold: float = None
    scale_down_threshold: float = None

    disable_log_requests_manager: bool = None
    log_instance_info: bool = None
    log_filename: str = None
    simulator_mode: bool = None
    profiling_result_file_path: str = None

    migration_backend: str = None
    migration_buffer_blocks: int = None
    migration_num_layers: int = None
    migration_backend_init_timeout: float = None
    migration_backend_transfer_type: str = None
    grpc_migration_backend_server_address: str = None
    kvtransfer_migration_backend_naming_url: str = None
    last_stage_max_blocks: int = None
    max_stages: int = None

    enable_pd_disagg: bool = None
    num_dispatch_instances: int = None

    enable_port_increment: bool = None
    enable_port_offset_store: bool = None


    def __post_init__(self):
        # Check if all fields default to None
        for field_info in dataclasses.fields(self):
            if field_info.default is not None:
                raise ValueError(f"The default value of '{field_info.name}' should be None")

        for attr in dataclasses.fields(self):
            if getattr(self, attr.name) is None:
                setattr(self, attr.name, getattr(_C.MANAGER, attr.name.upper()))

    def create_global_scheduler_config(
        self,
    ) -> Tuple[GlobalSchedulerConfig]:

        # Create the GlobalScheduler Configuration.
        global_scheduler_config = GlobalSchedulerConfig(self.initial_instances,
                                                        self.load_metric,
                                                        self.dispatch_policy,
                                                        self.power_of_k_choice,
                                                        self.pair_migration_policy,
                                                        self.migrate_out_threshold,
                                                        self.enable_defrag,
                                                        self.scaling_policy,
                                                        self.scale_up_threshold,
                                                        self.scale_down_threshold,
                                                        self.enable_pd_disagg,
                                                        self.num_dispatch_instances,
                                                        self.migration_backend)
        return global_scheduler_config

    def create_migration_config(self) -> MigrationConfig:
        migration_config = MigrationConfig(self.request_migration_policy,
                                           self.migration_backend,
                                           self.migration_buffer_blocks,
                                           self.migration_num_layers,
                                           self.last_stage_max_blocks,
                                           self.max_stages,
                                           self.migration_backend_init_timeout,
                                           self.migration_backend_transfer_type,
                                           self.grpc_migration_backend_server_address,
                                           self.kvtransfer_migration_backend_naming_url)
        return migration_config

    @classmethod
    def from_llumnix_config(cls, cfg: LlumnixConfig = get_llumnix_config()) -> 'ManagerArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        # The defalut values of attributes are defined in default.py.
        manager_args = cls(**{attr: getattr(cfg.MANAGER, attr.upper()) for attr in attrs})
        return manager_args

    @classmethod
    def check_args(cls, args: 'ManagerArgs', parser: argparse.ArgumentParser):
        # pylint: disable=protected-access
        for action in parser._optionals._actions:
            if hasattr(action, 'choices') and action.choices is not None and hasattr(args, action.dest):
                cur_arg = getattr(args, action.dest)
                assert cur_arg in action.choices, f"{action.dest} should be one of {action.choices}, but {cur_arg} is set."

        # bladellm only
        assert args.migration_backend not in ['kvtransfer'] or (args.migration_backend == 'kvtransfer' \
            and args.migration_backend_transfer_type), \
            ("When using kvTransfer as migration backend, "
             "do not set --migration-backend-transfer-type as empty.")

        assert not args.simulator_mode or args.profiling_result_file_path is not None, \
            "Set profiling_result_file_path args when enable simulator mode"

        assert not args.enable_port_offset_store or args.enable_port_increment, \
            "Set enable_port_increment when enable_port_offset_store"

        assert not args.enable_scaling, "Proactive auto-scaling is deprecated now, all auto-scaling related args will not take effects."

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--initial-instances',
                            type=int,
                            help='number of instances created at initialzation')

        parser.add_argument('--load-metric',
                            type=str,
                            choices=['remaining_steps', 'usage_ratio'],
                            help='instance load metric')
        parser.add_argument('--polling-interval',
                            type=float,
                            help='time interval(s) to update instance info and pair migration')

        parser.add_argument('--dispatch-policy',
                            type=str,
                            choices=['balanced', 'load', 'queue', 'flood', 'rr'],
                            help='The request dispatch policy.\n\n'
                            '* "balanced" dispatch request to the instance with minimum requests dispatched.\n'
                            '* "load" dispatch request to the instance with lowest instance load.\n'
                            '* "queue" dispatch request to the instance with minimum waiting request queue length.\n'
                            '* "flood" dispatch request to the instance with maximum requests dispatched.\n'
                            '* "rr" dispatch requests with round-robin policy.\n')
        parser.add_argument('--power-of-k-choice',
                            type=int,
                            help='number of candidate instances for dispatch policy.\n\n'
                            'The candidate instances are first selected according to the load'
                            '(including factors such as load, queue size, etc.) based on the dispatch policy,'
                            'and then one of them is randomly chosen to receive the request for better load balancing.')

        parser.add_argument('--enable-migration',
                            action='store_true',
                            help='enable migrate requests between instances')
        parser.add_argument('--enable-defrag',
                            type=bool,
                            help='enable defragmentation through migration based on virtual usage')
        parser.add_argument('--pair-migration-frequency',
                            type=int,
                            help='pair migration frequency')
        parser.add_argument('--pair-migration-policy',
                            type=str,
                            choices=['balanced', 'defrag_constrained', 'defrag_relaxed'],
                            help='The pair migration policy.\n\n'
                            '* "balanced" pair migration to make the instance load of instance more balanced.\n'
                            '* "defrag_constrained" pair migration without balanced constraint to '
                            'achieve defragmentation thoroughly (with instance constraints).\n'
                            '* "defrag_relaxed" pair migration to without balanced constraint '
                            'to achieve defragmentation thoroughly (without instance constraints).\n')
        parser.add_argument('--migrate-out-threshold',
                            type=float,
                            help='migrate out instance load threshold')
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

        parser.add_argument('--disable-log-requests-manager',
                            action='store_true',
                            help='disable logging requests in manager')
        parser.add_argument('--log-instance-info',
                            action='store_true',
                            help='enable logging instance info')
        parser.add_argument('--log-filename',
                            type=str,
                            help='log filename')
        parser.add_argument('--profiling-result-file-path',
                            type=str,
                            help='profiling result file path when using simulator')
        parser.add_argument('--simulator-mode',
                            action='store_true',
                            help='enable simulator mode')
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
                            choices=['cuda_ipc','rdma', ''],
                            help='transfer type for migration backend grpc and kvTransfer')
        parser.add_argument('--grpc-migration-backend-server-address',
                            type=str,
                            help='address of grpc server for migration backend')
        parser.add_argument('--kvtransfer-migration-backend-naming-url',
                            type=str,
                            help='url of naming server for kvtransfer migration backend')
        parser.add_argument('--max-stages',
                            type=int,
                            help='drop migration if the number of stages > max_stages')
        parser.add_argument('--last-stage-max-blocks',
                            type=int,
                            help='if the number pf remain blocks < last_stage_max_blocks, do last stage migration')

        parser.add_argument('--enable-pd-disagg',
                            action='store_true',
                            help='enable prefill decoding disaggregation')
        parser.add_argument('--num-dispatch-instances',
                            type=int,
                            help='number of available instances for dispatch')

        parser.add_argument('--enable-port-increment',
                            action='store_true',
                            help='enable port increment when desploying multiple servers')
        parser.add_argument('--enable-port-offset-store',
                            action='store_true',
                            help='enable store port offset when desploying multiple servers')

        return parser

@dataclass
class LaunchArgs:
    launch_mode: LaunchMode = None
    backend_type: BackendType = None
