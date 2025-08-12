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

from typing import Union, List

from dataclasses import dataclass

@dataclass
class DispatchLoadMetricConfig:
    dispatch_load_metric: str
    dispatch_prefill_load_metric: str
    dispatch_decode_load_metric: str
    dispatch_prefill_as_decode_load_metric: bool
    dispatch_decode_as_prefill_load_metric: bool

class GlobalSchedulerConfig:
    def __init__(
            self,
            initial_instances: int,
            dispatch_policy: str,
            topk_random_dispatch: int,
            pair_migration_policy: str,
            migrate_out_threshold: float,
            scaling_policy: str,
            scaling_load_metric: str,
            scale_up_threshold: float,
            scale_down_threshold: float,
            enable_pd_disagg: bool,
            enable_vllm_v1_engine_pd_disagg: bool,
            enable_bladellm_engine_pd_disagg: bool,
            enable_bladellm_engine_semi_pd_disagg: bool,
            enable_adaptive_pd: bool,
            is_group_kind_migration_backend: bool,
            dispatch_load_metric_config: DispatchLoadMetricConfig,
            cache_meta_client_config_path: str=None,
            enable_pre_step_migration: bool=False,
            ) -> None:
        self.initial_instances = initial_instances
        self.dispatch_policy = dispatch_policy
        self.topk_random_dispatch = topk_random_dispatch

        self.pair_migration_policy = pair_migration_policy
        self.migrate_out_load_threshold = migrate_out_threshold

        self.scaling_policy = scaling_policy
        self.scaling_load_metric = scaling_load_metric
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

        self.enable_pd_disagg = enable_pd_disagg
        self.enable_vllm_v1_engine_pd_disagg = enable_vllm_v1_engine_pd_disagg
        self.enable_bladellm_engine_pd_disagg = enable_bladellm_engine_pd_disagg
        self.enable_bladellm_engine_semi_pd_disagg = enable_bladellm_engine_semi_pd_disagg
        self.enable_adaptive_pd = enable_adaptive_pd
        self.is_group_kind_migration_backend = is_group_kind_migration_backend

        self.dispatch_load_metric_config = dispatch_load_metric_config
        self.cache_meta_client_config_path = cache_meta_client_config_path

        self.enable_pre_step_migration = enable_pre_step_migration


class MigrationConfig:
    def __init__(
            self,
            enable_routine_migration: bool,
            enable_pre_stop_migration: bool,
            request_migration_policy: str,
            migration_backend: str,
            migration_buffer_blocks: int,
            migration_num_layers: int,
            migration_last_stage_max_blocks: int,
            migration_max_stages: int,
            migration_backend_init_timeout: float,
            kvtransfer_migration_backend_transfer_type: str = "",
            kvtransfer_migration_backend_naming_url: str = "",
            ) -> None:
        self.enable_routine_migration = enable_routine_migration
        self.enable_pre_stop_migration = enable_pre_stop_migration
        self.request_migration_policy = request_migration_policy
        self.migration_backend = migration_backend
        self.kvtransfer_migration_backend_transfer_type = kvtransfer_migration_backend_transfer_type
        self.migration_num_layers = migration_num_layers
        self.migration_buffer_blocks = migration_buffer_blocks
        self.migration_last_stage_max_blocks = migration_last_stage_max_blocks
        self.migration_max_stages = migration_max_stages
        self.migration_backend_init_timeout = migration_backend_init_timeout
        self.kvtransfer_migration_backend_naming_url = kvtransfer_migration_backend_naming_url

        # lazy init in MigrationLocalWorker constructor
        self.grpc_migration_server_port: List = None

    @property
    def enable_migration(self):
        return self.enable_routine_migration or self.enable_pre_stop_migration


class PDDConfig:
    def __init__(
            self,
            enable_pd_disagg: bool,
            enable_vllm_v1_engine_pd_disagg: bool,
            enable_bladellm_engine_pd_disagg: bool,
            enable_bladellm_engine_semi_pd_disagg: bool,
            pd_ratio: Union[str, List[int]],
            enable_pdd_node_affinity_scheduling: bool) -> None:
        self.enable_pd_disagg = enable_pd_disagg
        self.enable_vllm_v1_engine_pd_disagg = enable_vllm_v1_engine_pd_disagg
        self.enable_bladellm_engine_pd_disagg = enable_bladellm_engine_pd_disagg
        self.enable_bladellm_engine_semi_pd_disagg = enable_bladellm_engine_semi_pd_disagg
        self.pd_ratio = pd_ratio
        self.enable_pdd_node_affinity_scheduling = enable_pdd_node_affinity_scheduling

    @property
    def enable_pd(self):
        return self.enable_pd_disagg \
            or self.enable_vllm_v1_engine_pd_disagg \
            or self.enable_bladellm_engine_pd_disagg \
            or self.enable_bladellm_engine_semi_pd_disagg
