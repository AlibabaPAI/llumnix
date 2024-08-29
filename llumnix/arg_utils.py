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

from llumnix.config import GlobalSchedulerConfig, MigrationConfig


@dataclass
class EngineManagerArgs:
    launch_ray_cluster: bool = True
    disable_init_instance_by_manager: bool = False
    initial_instances: int = 1
    disable_fixed_node_init_instance: bool = False

    load_metric: str = 'remaining_steps'
    polling_interval: float = 0.05

    dispatch_policy: str = 'load'

    enable_migration: bool = True
    enable_defrag: bool = True
    pair_migration_frequency: int = 1
    pair_migration_policy: str = 'defrag_constrained'
    migrate_out_threshold: float = 3.0
    request_migration_policy: str = 'SJF'

    enable_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 1
    scaling_interval: int = 10
    scaling_policy: str = 'avg_load'
    scale_up_threshold: float = 10
    scale_down_threshold: float = 60

    log_filename: str = "server.log"
    disable_log_requests_manager: bool = False
    log_instance_info: bool = True
    profiling_result_file_path: str = ""

    gpu_type: str = "a10"
    migration_backend_init_timeout: float = 10.0
    migration_backend: str = "rpc"
    migration_cache_blocks: int = 512
    migration_num_layers: int = 1
    last_stage_max_blocks: int = 16
    max_stages: int = 3

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
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineManagerArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_manager_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        cls._check_args(engine_manager_args)
        return engine_manager_args

    @classmethod
    def _check_args(cls, args):
        assert args.migration_backend != 'gloo' or (args.migration_backend == 'gloo' \
            and not args.disable_init_instance_by_manager and not args.disable_fixed_node_init_instance), \
            ("When using gloo as migration backend, "
             "do not set --disable-init-instance-by-manager and --disable-fixed-node-init-instance.")

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--disable-fixed-node-init-instance',
                            action='store_true',
                            help='disable fixing the placement of instance to current node')
        parser.add_argument('--disable-init-instance-by-manager',
                            action='store_true',
                            help='disable initializing instance by manager')
        parser.add_argument('--initial-instances',
                            type=int,
                            default=EngineManagerArgs.initial_instances,
                            help='number of instances created at initialzation')

        parser.add_argument('--load-metric',
                            type=str,
                            default=EngineManagerArgs.load_metric,
                            choices=['remaining_steps', 'usage_ratio'],
                            help='instance load metric')
        parser.add_argument('--polling-interval',
                            type=float,
                            default=EngineManagerArgs.polling_interval,
                            help='time interval(s) to update instance info and pair migration')

        parser.add_argument('--dispatch-policy',
                            type=str,
                            default=EngineManagerArgs.dispatch_policy,
                            choices=['balanced', 'load', 'queue', 'flood'],
                            help='request dispatch policy')

        parser.add_argument('--enable-migration',
                            action='store_true',
                            help='enable migrate requests between instances')
        parser.add_argument('--pair-migration-frequency',
                            type=int,
                            default=EngineManagerArgs.pair_migration_frequency,
                            help='pair migration frequency')
        parser.add_argument('--pair-migration-policy',
                            type=str,
                            default=EngineManagerArgs.pair_migration_policy,
                            choices=['balanced', 'defrag_constrained', 'defrag_relaxed'],
                            help='pair migration policy')
        parser.add_argument('--migrate-out-threshold',
                            type=float,
                            default=EngineManagerArgs.migrate_out_threshold,
                            help='migrate out instance load threshold')
        parser.add_argument('--request-migration-policy',
                            type=str,
                            default=EngineManagerArgs.request_migration_policy,
                            choices=['LCFS', 'SJF', 'LJF'],
                            help='request migration policy')
        parser.add_argument('--enable-defrag',
                            type=bool,
                            default=EngineManagerArgs.enable_defrag,
                            help='enable defragmentation through migration based on virtual usage')

        parser.add_argument('--enable-scaling',
                            action='store_true',
                            help='enable auto scaling')
        parser.add_argument('--min-instances',
                            type=int,
                            default=EngineManagerArgs.min_instances,
                            help='minimum number of instances')
        parser.add_argument('--max-instances',
                            type=int,
                            default=EngineManagerArgs.max_instances,
                            help='maximum number of instances')
        parser.add_argument('--scaling-interval',
                            type=int,
                            default=EngineManagerArgs.scaling_interval,
                            help='interval time of check scaling')
        parser.add_argument('--scaling-policy',
                            type=str,
                            default=EngineManagerArgs.scaling_policy,
                            choices=['max_load', 'avg_load'],
                            help='scaling policy')
        parser.add_argument('--scale-up-threshold',
                            type=float,
                            default=EngineManagerArgs.scale_up_threshold,
                            help='scale up threshold')
        parser.add_argument('--scale-down-threshold',
                            type=float,
                            default=EngineManagerArgs.scale_down_threshold,
                            help='scale down threshold')

        parser.add_argument('--disable-log-requests-manager',
                            action='store_true',
                            help='disable logging requests in manager')
        parser.add_argument('--log-instance-info',
                            action='store_true',
                            help='enable logging instance info')
        parser.add_argument('--log-filename',
                            type=str,
                            default=EngineManagerArgs.log_filename,
                            help='log filename')
        parser.add_argument('--profiling-result-file-path',
                            type=str,
                            default=EngineManagerArgs.profiling_result_file_path,
                            help='profiling result file path')

        parser.add_argument('--gpu-type',
                            type=str,
                            default=EngineManagerArgs.gpu_type,
                            help='gpu type specified when using simulator')

        parser.add_argument('--migration-backend',
                            type=str,
                            default=EngineManagerArgs.migration_backend,
                            choices=['gloo','nccl','rpc'],
                            help='communication backend of migration')
        parser.add_argument('--migration-backend-init-timeout',
                            type=float,
                            default=EngineManagerArgs.migration_backend_init_timeout,
                            help='timeout(s) for initializing migration backend')
        parser.add_argument('--migration-cache-blocks',
                            type=int,
                            default=EngineManagerArgs.migration_cache_blocks,
                            help='number of cache blocks in migration')
        parser.add_argument('--migration-num-layers',
                            type=int,
                            default=EngineManagerArgs.migration_num_layers,
                            help='number of kv-cache layers to transfer in each round during migration')
        parser.add_argument('--last-stage-max-blocks',
                            type=int,
                            default=EngineManagerArgs.last_stage_max_blocks,
                            help='if the number pf remain blocks < last_stage_max_blocks, do last stage migration')
        parser.add_argument('--max-stages',
                            type=int,
                            default=EngineManagerArgs.max_stages,
                            help='drop migration if the number of stages > max_stages')

        return parser
