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
            migrate_policy: str,
            migration_backend: str,
            migration_cache_blocks: int,
            migration_num_layers: int,
            last_stage_max_blocks: int,
            max_stages: int,
            migration_backend_init_timeout: float,
            model_parallelism_enabled: bool) -> None:
        self.migrate_policy = migrate_policy
        self.migration_backend = migration_backend
        self.migration_num_layers = migration_num_layers
        self.migration_cache_blocks = migration_cache_blocks
        self.last_stage_max_blocks = last_stage_max_blocks
        self.max_stages = max_stages
        self.migration_backend_init_timeout = migration_backend_init_timeout
        self.model_parallelism_enabled = model_parallelism_enabled

class GlobalSchedulerConfig:
    def __init__(
            self,
            initial_instances: int,
            load_metric: str,
            dispatch_policy: str,
            check_migirate_policy: str,
            migrate_out_threshold: float,
            enable_prefill_migrate: bool,
            scale_policy: str,
            scale_up_threshold: float,
            scale_down_threshold: float) -> None:
        self.initial_instances = initial_instances
        self.load_metric = load_metric

        self.dispatch_policy = dispatch_policy

        self.check_migrate_policy = check_migirate_policy
        self.migrate_out_load_threshold = migrate_out_threshold*(-1)
        self.enable_prefill_migrate = enable_prefill_migrate

        self.scale_policy = scale_policy
        self.scale_up_threshold = scale_up_threshold*(-1)
        self.scale_down_threshold = scale_down_threshold*(-1)
