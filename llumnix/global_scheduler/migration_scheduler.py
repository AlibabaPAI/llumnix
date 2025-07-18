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

from typing import Dict, List, Optional, Tuple

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.load_computation import DummyLoad
from llumnix.global_scheduler.migration_filter import (MigrationFilterPipeline, MigrationFilterConfig,
                                                       CustomFilter, MigrationFilterFactory)
from llumnix.global_scheduler.migration_policy import MigrationPolicyFactory, MigrationPolicy
from llumnix.utils import MigrationType, InstanceType
from llumnix.internal_config import DispatchLoadMetricConfig

logger = init_logger(__name__)


class MigrationScheduler:
    def __init__(self,
                 pair_migration_policy: str,
                 migrate_out_load_threshold: float,
                 is_group_kind_migration_backend: bool,
                 enable_pd_disagg: bool,
                 enable_engine_pd_disagg: bool,
                 enable_engine_semi_pd_disagg: bool,
                 enable_adaptive_pd: bool,
                 dispatch_load_metric_config: DispatchLoadMetricConfig) -> None:
        self.pair_migration_policy = pair_migration_policy
        self.enable_pd_disagg = enable_pd_disagg
        self.enable_engine_pd_disagg = enable_engine_pd_disagg
        self.enable_engine_semi_pd_disagg = enable_engine_semi_pd_disagg
        self.enable_adaptive_pd = enable_adaptive_pd
        self.dispatch_load_metric_config = dispatch_load_metric_config

        self.filter_config = MigrationFilterConfig(migrate_out_load_threshold=migrate_out_load_threshold)
        self.migration_base_filter = MigrationFilterPipeline(self.filter_config)
        self._register_migration_backend_init_filter(is_group_kind_migration_backend)
        self._register_new_instance_filter()
        self._register_has_migration_slot_filter()

        self._set_migration_filter()
        self._set_migration_policy()

    def _enable_pd(self):
        return self.enable_engine_pd_disagg or self.enable_engine_semi_pd_disagg or self.enable_pd_disagg

    def _register_migration_backend_init_filter(self, is_group_kind_migration_backend: bool) -> None:
        # some migration backends require init_process_group before passing the KV cache. Here, we add a filter
        # to prevent instances of migration backends that have not been initialized from participating in migration.
        migration_backend_init_filter = CustomFilter()
        migration_backend_init_filter.set_filter_condtition(
            src_filter=lambda _: not is_group_kind_migration_backend,
            dst_filter=lambda _: not is_group_kind_migration_backend)
        self.migration_base_filter.add_filter("migration_backend_init_filter", migration_backend_init_filter)

    def _register_new_instance_filter(self) -> None:
        # instances that have just been launched should be refused for migration due to the absence of load information.
        new_instance_filter = CustomFilter()
        new_instance_filter.set_filter_condtition(
            src_filter=lambda instance_info: not isinstance(instance_info.migration_load_metric, DummyLoad),
            dst_filter=lambda instance_info: not isinstance(instance_info.migration_load_metric, DummyLoad))
        self.migration_base_filter.add_filter("new_instance_filter", new_instance_filter)

    def _register_has_migration_slot_filter(self) -> None:
        # has_migration_slot is used to filter the instances that have reached the max concurrent parallelism.
        has_migration_slot_filter = CustomFilter()
        has_migration_slot_filter.set_filter_condtition(
            src_filter=lambda instance_info: instance_info.has_migration_slot,
            dst_filter=lambda instance_info: instance_info.has_migration_slot
        )
        self.migration_base_filter.add_filter("has_migration_slot_filter", has_migration_slot_filter)

    def _set_migration_filter(self):
        self.load_filter = MigrationFilterFactory.get_filter("load")

        if not self._enable_pd():
            self.no_constraints_load_balance_filter_pipeline = MigrationFilterPipeline(self.filter_config)
            self.no_constraints_load_balance_filter_pipeline.add_filter(
                "no_constraints_load_balance_filter", self.load_filter)

        if self.enable_pd_disagg:
            self.p_d_filter: CustomFilter = MigrationFilterFactory.get_filter("custom")
            self.p_d_filter.set_filter_condtition(
                src_filter=lambda instance_info: instance_info.instance_type == InstanceType.PREFILL,
                dst_filter=lambda instance_info: instance_info.instance_type == InstanceType.DECODE \
                    and instance_info.num_killed_requests == 0
            )
            self.p2d_transfer_filter_pipeline = MigrationFilterPipeline(self.filter_config)
            self.p2d_transfer_filter_pipeline.add_filter("pd_instance_filter", self.p_d_filter)

        if self._enable_pd() and not self.enable_engine_pd_disagg:
            self.d_d_filter: CustomFilter = MigrationFilterFactory.get_filter("custom")
            self.d_d_filter.set_filter_condtition(
                src_filter=lambda instance_info: instance_info.instance_type == InstanceType.DECODE,
                dst_filter=lambda instance_info: instance_info.instance_type == InstanceType.DECODE
            )
            self.decode_load_balance_filter_pipeline = MigrationFilterPipeline(self.filter_config)
            self.decode_load_balance_filter_pipeline.add_filter("dd_intance_filter", self.d_d_filter)
            self.decode_load_balance_filter_pipeline.add_filter("decode_load_balance_filter", self.load_filter)

        if self.enable_adaptive_pd:
            self.dynamic_p_free_d_filter: CustomFilter = MigrationFilterFactory.get_filter("custom")
            self.dynamic_p_free_d_filter.set_filter_condtition(
                src_filter=lambda instance_info: instance_info.instance_type == InstanceType.PREFILL \
                    and instance_info.decode_batch_size > 0,
                dst_filter=lambda instance_info: instance_info.instance_type == InstanceType.DECODE \
                    and not getattr(instance_info, self.dispatch_load_metric_config.dispatch_decode_load_metric).is_busy()
            )
            self.dynamic_p2d_filter_pipeline = MigrationFilterPipeline(self.filter_config)
            self.dynamic_p2d_filter_pipeline.add_filter("dynamic_p_d_instance_filter", self.dynamic_p_free_d_filter)

            self.dynamic_p_filter: CustomFilter = MigrationFilterFactory.get_filter("custom")
            self.dynamic_p_filter.set_filter_condtition(
                src_filter=lambda instance_info: instance_info.instance_type == InstanceType.PREFILL \
                    and instance_info.decode_batch_size > 0
                    and not getattr(instance_info, self.dispatch_load_metric_config.dispatch_prefill_as_decode_load_metric).is_busy(),
                dst_filter=lambda instance_info: instance_info.instance_type == InstanceType.PREFILL \
                    and instance_info.decode_batch_size > 0
                    and not getattr(instance_info, self.dispatch_load_metric_config.dispatch_prefill_load_metric).is_busy()
                    and not getattr(instance_info, self.dispatch_load_metric_config.dispatch_prefill_as_decode_load_metric).is_busy()
            )
            self.aggrate_dynamic_p_filter_pipeline = MigrationFilterPipeline(self.filter_config)
            self.aggrate_dynamic_p_filter_pipeline.add_filter("aggrate_dynamic_p_filter", self.dynamic_p_filter)

            self.d_p_filter: CustomFilter = MigrationFilterFactory.get_filter("custom")
            self.d_p_filter.set_filter_condtition(
                src_filter=lambda instance_info: instance_info.instance_type == InstanceType.DECODE \
                    and getattr(instance_info, self.dispatch_load_metric_config.dispatch_decode_load_metric).is_busy(),
                dst_filter=lambda instance_info: instance_info.instance_type == InstanceType.PREFILL \
                    and instance_info.num_running_requests == 0
            )
            self.ease_d_with_empty_p_filter_pipeline = MigrationFilterPipeline(self.filter_config)
            self.ease_d_with_empty_p_filter_pipeline.add_filter("busy_d_empty_p_filter", self.d_p_filter)

    def _set_migration_policy(self):
        self.pair_migration_policy = MigrationPolicyFactory.get_policy(
            self.pair_migration_policy,
            migrate_out_load_threshold=self.filter_config.migrate_out_load_threshold)

        if self.enable_pd_disagg or self.enable_adaptive_pd:
            self.defrag_policy = MigrationPolicyFactory.get_policy(
                "defrag", migrate_out_load_threshold=self.filter_config.migrate_out_load_threshold)

        if self.enable_adaptive_pd:
            self.aggrate_dynamic_p_policy = MigrationPolicyFactory.get_policy(
                "aggrate_dynamic_prefill", migrate_out_load_threshold=self.filter_config.migrate_out_load_threshold)

    def push_migrations(self, instance_info: Dict[str, InstanceInfo]) -> List[List[Tuple[str, str]]]:
        migration_tasks = []

        if self.enable_pd_disagg:
            migration_tasks.append((
                MigrationType.PD_MIGRATION,
                self._pair_migration(instance_info, self.p2d_transfer_filter_pipeline, self.defrag_policy)
            ))

        if not self._enable_pd():
            migration_tasks.append((
                MigrationType.NO_CONSTRAINTS_LOAD_BALANCE,
                self._pair_migration(instance_info, self.no_constraints_load_balance_filter_pipeline,
                                     self.pair_migration_policy)
            ))

        elif not self.enable_engine_pd_disagg:
            migration_tasks.append((
                MigrationType.DD_LOAD_BALANCE,
                self._pair_migration(instance_info, self.decode_load_balance_filter_pipeline, self.pair_migration_policy)
            ))

        if self.enable_adaptive_pd:
            exist_free_d = lambda instance_infos: any(
                instance_info.instance_type == InstanceType.DECODE
                and not getattr(instance_info, self.dispatch_load_metric_config.dispatch_decode_load_metric).is_busy()
                for instance_info in instance_infos.values()
            )
            if exist_free_d(instance_info):
                migration_tasks.append((
                    MigrationType.DYNAMIC_P_TO_D,
                    self._pair_migration(instance_info, self.dynamic_p2d_filter_pipeline, self.defrag_policy)
                ))
            else:
                migration_tasks.append((
                    MigrationType.AGGREGATE_DYNAMIC_P,
                    self._pair_migration(instance_info, self.aggrate_dynamic_p_filter_pipeline, self.aggrate_dynamic_p_policy)
                ))

            migration_tasks.append((
                MigrationType.EASE_D_WITH_P_BUBBLE,
                self._pair_migration(instance_info, self.ease_d_with_empty_p_filter_pipeline, self.defrag_policy)
            ))

        return migration_tasks

    # migration_filter must ensure that the specific instance_info does not appear in both src and dst simultaneously
    def _pair_migration(self,
                        instance_info: Dict[str, InstanceInfo],
                        migration_filter_pipeline: Optional[MigrationFilterPipeline],
                        migration_policy: MigrationPolicy) -> List[Tuple[str, str]]:
        src_instance_infos, dst_instance_infos = self.migration_base_filter.filter_instances(instance_info.values())
        if migration_filter_pipeline:
            src_instance_infos = migration_filter_pipeline.filter_src_instances(src_instance_infos)
            dst_instance_infos = migration_filter_pipeline.filter_dst_instances(dst_instance_infos)
        pair_instance_ids = migration_policy.pair_migration(src_instance_infos, dst_instance_infos)

        for src_instance_id, dst_instance_id in pair_instance_ids:
            assert src_instance_id != dst_instance_id, f"migration src and dst instance should not be the same, but got" \
                f"{src_instance_id} and {dst_instance_id}."

        return pair_instance_ids
