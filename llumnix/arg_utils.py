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

@dataclass
class EngineManagerArgs:
    disable_init_instance_by_manager: bool = None
    initial_instances: int = None
    disable_fixed_node_init_instance: bool = None

    load_metric: str = None
    polling_interval: float = None

    dispatch_policy: str = None

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

    log_filename: str = None
    disable_log_requests_manager: bool = None
    log_instance_info: bool = None
    profiling_result_file_path: str = None

    gpu_type: str = None
    migration_backend_init_timeout: float = None
    migration_backend: str = None
    migration_cache_blocks: int = None
    migration_num_layers: int = None
    last_stage_max_blocks: int = None
    max_stages: int = None

    def __post_init__(self):
        for attr in dataclasses.fields(self):
            if getattr(self, attr.name) is None:
                setattr(self, attr.name, getattr(_C.MANAGER, attr.name.upper()))

    def create_global_scheduler_configs(
        self,
    ) -> Tuple[GlobalSchedulerConfig]:
        global_scheduler_config = GlobalSchedulerConfig(self.initial_instances,
                                                        self.load_metric,
                                                        self.dispatch_policy,
                                                        self.pair_migration_policy,
                                                        self.migrate_out_threshold,
                                                        self.enable_defrag,
                                                        self.scaling_policy,
                                                        self.scale_up_threshold,
                                                        self.scale_down_threshold)
        return global_scheduler_config

    def create_migration_config(self) -> MigrationConfig:
        migration_config = MigrationConfig(self.request_migration_policy,
                                           self.migration_backend,
                                           self.migration_cache_blocks,
                                           self.migration_num_layers,
                                           self.last_stage_max_blocks,
                                           self.max_stages,
                                           self.migration_backend_init_timeout)
        return migration_config

    @classmethod
    def from_llumnix_config(cls, cfg: LlumnixConfig = get_llumnix_config()) -> 'EngineManagerArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_manager_args = cls(**{attr: getattr(cfg.MANAGER, attr.upper()) for attr in attrs})
        return engine_manager_args

    @classmethod
    def check_args(cls, args: 'EngineManagerArgs'):
        assert args.load_metric in ['remaining_steps', 'usage_ratio'], \
            ("Invalid load metric: {}".format(args.load_metric))

        assert args.dispatch_policy in ['balanced', 'load', 'queue', 'flood'], \
            ("Invalid dispatch policy: {}".format(args.dispatch_policy))

        assert args.pair_migration_policy in ['balanced', 'defrag_constrained', 'defrag_relaxed'], \
            ("Invalid pair migration policy: {}".format(args.pair_migration_policy))

        assert args.request_migration_policy in ['LCFS', 'SJF', 'LJF'], \
            ("Invalid request migration policy: {}".format(args.request_migration_policy))

        assert args.migration_backend in ['rpc', 'gloo', 'nccl'], \
            ("Invalid migration backend: {}".format(args.migration_backend))

        assert args.scaling_policy in ['max_load', 'avg_load'], \
            ("Invalid scaling policy: {}".format(args.scaling_policy))

        assert args.migration_backend != 'gloo' or (args.migration_backend == 'gloo' \
            and not args.disable_init_instance_by_manager and not args.disable_fixed_node_init_instance), \
            ("When using gloo as migration backend, "
             "do not set --disable-init-instance-by-manager and --disable-fixed-node-init-instance.")

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--disable-fixed-node-init-instance',
                            action='store_true',
                            default=None,
                            help='disable fixing the placement of instance to current node')
        parser.add_argument('--disable-init-instance-by-manager',
                            action='store_true',
                            default=None,
                            help='disable initializing instance by manager')
        parser.add_argument('--initial-instances',
                            type=int,
                            default=None,
                            help='number of instances created at initialzation')

        parser.add_argument('--load-metric',
                            type=str,
                            default=None,
                            choices=['remaining_steps', 'usage_ratio'],
                            help='instance load metric')
        parser.add_argument('--polling-interval',
                            type=float,
                            default=None,
                            help='time interval(s) to update instance info and pair migration')

        parser.add_argument('--dispatch-policy',
                            type=str,
                            default=None,
                            choices=['balanced', 'load', 'queue', 'flood'],
                            help='request dispatch policy')

        parser.add_argument('--enable-migration',
                            action='store_true',
                            default=None,
                            help='enable migrate requests between instances')
        parser.add_argument('--pair-migration-frequency',
                            type=int,
                            default=None,
                            help='pair migration frequency')
        parser.add_argument('--pair-migration-policy',
                            type=str,
                            default=None,
                            choices=['balanced', 'defrag_constrained', 'defrag_relaxed'],
                            help='pair migration policy')
        parser.add_argument('--migrate-out-threshold',
                            type=float,
                            default=None,
                            help='migrate out instance load threshold')
        parser.add_argument('--request-migration-policy',
                            type=str,
                            default=None,
                            choices=['LCFS', 'SJF', 'LJF'],
                            help='request migration policy')
        parser.add_argument('--enable-defrag',
                            type=bool,
                            default=None,
                            help='enable defragmentation through migration based on virtual usage')

        parser.add_argument('--enable-scaling',
                            action='store_true',
                            default=None,
                            help='enable auto scaling')
        parser.add_argument('--min-instances',
                            type=int,
                            default=None,
                            help='minimum number of instances')
        parser.add_argument('--max-instances',
                            type=int,
                            default=None,
                            help='maximum number of instances')
        parser.add_argument('--scaling-interval',
                            type=int,
                            default=None,
                            help='interval time of check scaling')
        parser.add_argument('--scaling-policy',
                            type=str,
                            default=None,
                            choices=['max_load', 'avg_load'],
                            help='scaling policy')
        parser.add_argument('--scale-up-threshold',
                            type=float,
                            default=None,
                            help='scale up threshold')
        parser.add_argument('--scale-down-threshold',
                            type=float,
                            default=None,
                            help='scale down threshold')

        parser.add_argument('--disable-log-requests-manager',
                            action='store_true',
                            default=None,
                            help='disable logging requests in manager')
        parser.add_argument('--log-instance-info',
                            action='store_true',
                            default=None,
                            help='enable logging instance info')
        parser.add_argument('--log-filename',
                            type=str,
                            default=None,
                            help='log filename')
        parser.add_argument('--profiling-result-file-path',
                            type=str,
                            default=None,
                            help='profiling result file path')

        parser.add_argument('--gpu-type',
                            type=str,
                            default=None,
                            help='gpu type specified when using simulator')

        parser.add_argument('--migration-backend',
                            type=str,
                            default=None,
                            choices=['gloo','nccl','rpc'],
                            help='communication backend of migration')
        parser.add_argument('--migration-backend-init-timeout',
                            type=float,
                            default=None,
                            help='timeout(s) for initializing migration backend')
        parser.add_argument('--migration-cache-blocks',
                            type=int,
                            default=None,
                            help='number of cache blocks in migration')
        parser.add_argument('--migration-num-layers',
                            type=int,
                            default=None,
                            help='number of kv-cache layers to transfer in each round during migration')
        parser.add_argument('--last-stage-max-blocks',
                            type=int,
                            default=None,
                            help='if the number pf remain blocks < last_stage_max_blocks, do last stage migration')
        parser.add_argument('--max-stages',
                            type=int,
                            default=None,
                            help='drop migration if the number of stages > max_stages')

        return parser
