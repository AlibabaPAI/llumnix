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

class MigrationConfig:
    def __init__(
            self,
            request_migration_policy: str,
            migration_backend: str,
            migration_cache_blocks: int,
            migration_num_layers: int,
            last_stage_max_blocks: int,
            max_stages: int,
            migration_backend_init_timeout: float) -> None:
        self.request_migration_policy = request_migration_policy
        self.migration_backend = migration_backend
        self.migration_num_layers = migration_num_layers
        self.migration_cache_blocks = migration_cache_blocks
        self.last_stage_max_blocks = last_stage_max_blocks
        self.max_stages = max_stages
        self.migration_backend_init_timeout = migration_backend_init_timeout

class GlobalSchedulerConfig:
    def __init__(
            self,
            initial_instances: int,
            load_metric: str,
            dispatch_policy: str,
            pair_migration_policy: str,
            migrate_out_threshold: float,
            enable_defrag: bool,
            scaling_policy: str,
            scale_up_threshold: float,
            scale_down_threshold: float) -> None:
        self.initial_instances = initial_instances
        self.load_metric = load_metric

        self.dispatch_policy = dispatch_policy

        self.pair_migration_policy = pair_migration_policy
        self.migrate_out_load_threshold = migrate_out_threshold*(-1)
        self.enable_defrag = enable_defrag

        self.scaling_policy = scaling_policy
        self.scale_up_threshold = scale_up_threshold*(-1)
        self.scale_down_threshold = scale_down_threshold*(-1)
