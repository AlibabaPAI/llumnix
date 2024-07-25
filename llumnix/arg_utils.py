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
    fixed_node_init: bool = False

    initial_instances: int = 1
    load_metric: str = 'consumed_speed'
    polling_interval: float = 0.05

    dispatch_policy: str = 'load'

    enable_migrate: bool = False
    check_migrate_frequency: int = 1
    check_migrate_policy: str = 'prefill_constrained'
    migrate_out_threshold: float = 3.0
    migrate_policy: str = 'LCFS'
    enable_prefill_migrate: bool = True

    enable_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 1
    scaling_interval: int = 10
    scale_policy: str = 'avg_load'
    scale_up_threshold: float = 10
    scale_down_threshold: float = 60

    disable_log_requests_manager: bool = False
    results_filename: str = "server.log"
    record_instance_info: bool = False
    profiling_result_file_path: str = ""
    gpu_type: str = "a10"
    migration_backend: str = "rpc"
    migration_cache_blocks: int = 512
    last_stage_max_blocks: int = 4
    max_stages: int = 3

    def create_engine_manager_configs(
        self,
    ) -> Tuple[GlobalSchedulerConfig]:
        global_scheduler_config = GlobalSchedulerConfig(self.initial_instances,
                                                        self.load_metric,
                                                        self.dispatch_policy,
                                                        self.check_migrate_policy,
                                                        self.migrate_out_threshold,
                                                        self.enable_prefill_migrate,
                                                        self.scale_policy,
                                                        self.scale_up_threshold,
                                                        self.scale_down_threshold)
        return global_scheduler_config

    def create_migration_configs(
        self, instance_rank_map, pp_or_tp_enabled
    ) -> MigrationConfig:
        migration_config = MigrationConfig(self.migrate_policy,
                                           self.migration_backend,
                                           self.migration_cache_blocks,
                                           self.last_stage_max_blocks,
                                           self.max_stages,
                                           instance_rank_map,
                                           pp_or_tp_enabled)
        return migration_config

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineManagerArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--fixed-node-init',
                            action='store_true',
                            help='place llumlet and workers in current node.')

        parser.add_argument('--initial-instances',
                            type=int,
                            default=EngineManagerArgs.initial_instances,
                            help='number of model instance')
        parser.add_argument('--load-metric',
                            type=str,
                            default=EngineManagerArgs.load_metric,
                            choices=['consumed_speed', 'used_ratio'],
                            help='load metric')

        parser.add_argument('--dispatch-policy',
                            type=str,
                            default=EngineManagerArgs.dispatch_policy,
                            choices=['balanced', 'load', 'queue', 'flood'],
                            help='dispatch policy')

        parser.add_argument('--enable-migrate',
                            action='store_true',
                            help='enable migrate request between instance')
        parser.add_argument('--check-migrate-frequency',
                            type=int,
                            default=EngineManagerArgs.check_migrate_frequency,
                            help='check migrate frequency')
        parser.add_argument('--check-migrate-policy',
                            type=str,
                            default=EngineManagerArgs.check_migrate_policy,
                            choices=['balanced', 'prefill_constrained', 'prefill_relaxed'],
                            help='check migrate policy')
        parser.add_argument('--migrate-out-threshold',
                            type=float,
                            default=EngineManagerArgs.migrate_out_threshold,
                            help='migrate out load threshold')
        parser.add_argument('--migrate-policy',
                            type=str,
                            default=EngineManagerArgs.migrate_policy,
                            choices=['LCFS', 'SJF', 'LJF'],
                            help='migrate policy')
        parser.add_argument('--enable-prefill-migrate',
                            type=bool,
                            default=EngineManagerArgs.enable_prefill_migrate,
                            help='enable prefill migrate')

        parser.add_argument('--enable-scaling',
                            action='store_true',
                            help='enable auto scaline')
        parser.add_argument('--min-instances',
                            type=int,
                            default=EngineManagerArgs.min_instances,
                            help='min instances num')
        parser.add_argument('--max-instances',
                            type=int,
                            default=EngineManagerArgs.max_instances,
                            help='max instances num')
        parser.add_argument('--scaling-interval',
                            type=int,
                            default=EngineManagerArgs.scaling_interval,
                            help='interval time of check scaling')
        parser.add_argument('--scale-policy',
                            type=str,
                            default=EngineManagerArgs.scale_policy,
                            choices=['max_load', 'avg_load'],
                            help='scale policy')
        parser.add_argument('--scale-up-threshold',
                            type=float,
                            default=EngineManagerArgs.scale_up_threshold,
                            help='scaling up threshold')
        parser.add_argument('--scale-down-threshold',
                            type=float,
                            default=EngineManagerArgs.scale_down_threshold,
                            help='scaling down threshold')

        parser.add_argument('--disable-log-requests-manager',
                            action='store_true',
                            default=EngineManagerArgs.disable_log_requests_manager,
                            help='disable logging requests in manager')
        parser.add_argument('--record-instance-info',
                            action='store_true',
                            help='if record instance info')
        parser.add_argument('--results-filename',
                            type=str,
                            default=EngineManagerArgs.results_filename,
                            help='results filename')
        parser.add_argument('--profiling-result-file-path',
                            type=str,
                            default=EngineManagerArgs.profiling_result_file_path,
                            help='profiling result file path')
        parser.add_argument('--gpu-type',
                            type=str,
                            default=EngineManagerArgs.gpu_type,
                            help='gpu type specified when using simulator')
        parser.add_argument('--polling-interval',
                            type=float,
                            default=EngineManagerArgs.polling_interval,
                            help='time interval(s) to update instance info/migration')

        parser.add_argument('--migration-backend',
                            type=str,
                            default=EngineManagerArgs.migration_backend,
                            choices=['gloo','nccl','rpc'],
                            help='communication backend during migration')
        parser.add_argument('--migration-cache-blocks',
                            type=int,
                            default=EngineManagerArgs.migration_cache_blocks,
                            help='cache blocks num during migration')
        parser.add_argument('--last-stage-max-blocks',
                            type=int,
                            default=EngineManagerArgs.last_stage_max_blocks,
                            help='if the remain blocks num < last_stage_max_blocks, do last stage migration')
        parser.add_argument('--max-stages',
                            type=int,
                            default=EngineManagerArgs.max_stages,
                            help='drop migration if stage num > max_stages')

        return parser
