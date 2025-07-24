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

import random
from typing import Dict, List, Optional, Tuple
import pytest

from llumnix.instance_info import InstanceInfo
from llumnix.load_computation import KvBlocksRatioLoad, AdaptiveDecodeBatchLoad, RemainingStepsLoad
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler
from llumnix.global_scheduler.migration_filter import (MigrationFilterPipeline, MigrationFilterConfig,
                                                       MigrationFilterFactory, CustomFilter)
from llumnix.global_scheduler.migration_policy import MigrationPolicyFactory, Balanced, MigrationPolicy, Failover
from llumnix.internal_config import DispatchLoadMetricConfig
from llumnix.utils import InstanceType, UnitStatus

MIGRATE_OUT_LOAD_THRESHOLD = -3.0
INSTANCE_NUM = 15
assert INSTANCE_NUM % 2 != 0, "Some test need an odd number of instances."

def init_migration_scheduler(
        policy: str = 'balanced',
        enable_pd_disagg: bool = False,
        enable_engine_pd_disagg: bool = False,
        enable_engine_semi_pd_disagg: bool = False,
        enable_adaptive_pd: bool = False
    ):
    migration_scheduler = MigrationScheduler(
        pair_migration_policy=policy,
        migrate_out_load_threshold=MIGRATE_OUT_LOAD_THRESHOLD,
        is_group_kind_migration_backend=False,
        enable_pd_disagg=enable_pd_disagg,
        enable_engine_pd_disagg=enable_engine_pd_disagg,
        enable_engine_semi_pd_disagg=enable_engine_semi_pd_disagg,
        enable_adaptive_pd=enable_adaptive_pd,
        dispatch_load_metric_config = DispatchLoadMetricConfig(
            dispatch_load_metric='remaining_steps',
            dispatch_prefill_load_metric='kv_blocks_ratio',
            dispatch_decode_load_metric='remaining_steps',
            dispatch_prefill_as_decode_load_metric='adaptive_decode',
            dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
        )
    )
    return migration_scheduler


def test_load_migration_filter():
    num_tests = 1000

    migration_filter_pipeline = MigrationFilterPipeline(MigrationFilterConfig(MIGRATE_OUT_LOAD_THRESHOLD))
    load_filter = MigrationFilterFactory.get_filter('load')
    migration_filter_pipeline.add_filter('load_filter', load_filter)

    for _ in range(num_tests):
        instance_infos = []

        for instance_id in range(0, INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-1, 1)
            instance_info.num_killed_requests = random.randint(0, 1)
            instance_infos.append(instance_info)

        src_instance_infos, dst_instance_infos = migration_filter_pipeline.filter_instances(instance_infos)

        for instance_info in src_instance_infos:
            assert instance_info.num_killed_requests > 0  or MIGRATE_OUT_LOAD_THRESHOLD < instance_info.migration_load_metric

        for instance_info in dst_instance_infos:
            assert instance_info.num_killed_requests == 0 and instance_info.migration_load_metric < MIGRATE_OUT_LOAD_THRESHOLD

def test_custom_migration_filter():
    migration_filter_pipeline = MigrationFilterPipeline(MigrationFilterConfig(MIGRATE_OUT_LOAD_THRESHOLD))
    custom_filter: CustomFilter = MigrationFilterFactory.get_filter('custom')
    custom_src_filter = lambda instance_info: instance_info.instance_id != '0'
    custom_dst_filter = lambda instance_info: instance_info.instance_id != '1'
    custom_filter.set_filter_condtition(src_filter=custom_src_filter, dst_filter=custom_dst_filter)
    migration_filter_pipeline.add_filter('custom_filter', custom_filter)

    instance_infos = []
    for instance_id in range(0, INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_id = instance_id

    src_instance_infos, dst_instance_infos = migration_filter_pipeline.filter_instances(instance_infos)

    for instance_info in src_instance_infos:
        assert custom_src_filter(instance_info)

    for instance_info in dst_instance_infos:
        assert custom_dst_filter(instance_info)

def test_balanced_migration_policy():
    num_tests = 1000
    exist_migration = False
    balanced_migration_policy: Balanced = MigrationPolicyFactory.get_policy(
        'balanced', migrate_out_load_threshold=MIGRATE_OUT_LOAD_THRESHOLD)

    for _ in range(num_tests):
        all_instance_infos: Dict[str, InstanceInfo] = {}

        src_instance_infos = []
        for idx in range(INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_id = f"src_instance_id_{idx}"
            all_instance_infos[instance_info.instance_id] = instance_info
            instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)
            instance_info.migration_load_metric_after_migrate_out = instance_info.migration_load_metric - random.uniform(0, 1)
            instance_info.migration_load_metric_after_migrate_in = instance_info.migration_load_metric + random.uniform(0, 1)
            instance_info.num_killed_requests = random.randint(0, 1)
            src_instance_infos.append(instance_info)

        dst_instance_infos = []
        for idx in range(INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_id = f"dst_instance_id_{idx}"
            all_instance_infos[instance_info.instance_id] = instance_info
            instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)
            instance_info.migration_load_metric_after_migrate_out = instance_info.migration_load_metric - random.uniform(0, 1)
            instance_info.migration_load_metric_after_migrate_in = instance_info.migration_load_metric + random.uniform(0, 1)
            instance_info.num_killed_requests = random.randint(0, 1)
            dst_instance_infos.append(instance_info)

        migrate_instance_pairs = balanced_migration_policy.pair_migration(src_instance_infos, dst_instance_infos)
        exist_migration = exist_migration or len(migrate_instance_pairs) > 0

        for migrate_out_instance_id, migrate_in_instance_id in migrate_instance_pairs:
            migrate_out_instance = all_instance_infos[migrate_out_instance_id]
            migrate_in_instance = all_instance_infos[migrate_in_instance_id]
            assert migrate_in_instance.migration_load_metric < migrate_out_instance.migration_load_metric
            assert migrate_in_instance.migration_load_metric_after_migrate_in < MIGRATE_OUT_LOAD_THRESHOLD
            assert 0 < migrate_out_instance.migration_load_metric_after_migrate_out - migrate_in_instance.migration_load_metric_after_migrate_in \
                < migrate_out_instance.migration_load_metric - migrate_in_instance.migration_load_metric

    assert exist_migration

def test_aggrate_dynamic_prefill_migration_policy():
    aggrate_migration_policy: Balanced = MigrationPolicyFactory.get_policy(
        'aggrate_dynamic_prefill', migrate_out_load_threshold=MIGRATE_OUT_LOAD_THRESHOLD)

    all_instance_infos: Dict[str, InstanceInfo] = {}
    instance_infos = []
    for idx in range(INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_id = f"instance_id_{idx}"
        all_instance_infos[instance_info.instance_id] = instance_info
        instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)
        instance_infos.append(instance_info)

    migrate_instance_pairs = aggrate_migration_policy.pair_migration(instance_infos, instance_infos)
    assert len(migrate_instance_pairs) > 0
    assert len(migrate_instance_pairs) == INSTANCE_NUM//2

    for migrate_out_instance_id, migrate_in_instance_id in migrate_instance_pairs:
        migrate_out_instance = all_instance_infos[migrate_out_instance_id]
        migrate_in_instance = all_instance_infos[migrate_in_instance_id]
        assert migrate_out_instance.instance_id != migrate_in_instance.instance_id

def test_defrag_migration_policy():
    defrag_migration_policy: Balanced = MigrationPolicyFactory.get_policy(
        'defrag', migrate_out_load_threshold=MIGRATE_OUT_LOAD_THRESHOLD)
    all_instance_infos: Dict[str, InstanceInfo] = {}

    src_instance_infos = []
    for idx in range(INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_id = f"src_instance_id_{idx}"
        all_instance_infos[instance_info.instance_id] = instance_info
        instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)
        src_instance_infos.append(instance_info)

    dst_instance_infos = []
    for idx in range(INSTANCE_NUM-1):
        instance_info = InstanceInfo()
        instance_info.instance_id = f"dst_instance_id_{idx}"
        all_instance_infos[instance_info.instance_id] = instance_info
        instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)
        dst_instance_infos.append(instance_info)

    migrate_instance_pairs = defrag_migration_policy.pair_migration(src_instance_infos, dst_instance_infos)
    assert len(migrate_instance_pairs) == INSTANCE_NUM-1

    prev_src_migration_load, prev_dst_migration_load = None, None
    for migrate_out_instance_id, migrate_in_instance_id in migrate_instance_pairs:
        migrate_out_instance = all_instance_infos[migrate_out_instance_id]
        migrate_in_instance = all_instance_infos[migrate_in_instance_id]
        assert prev_src_migration_load is None or prev_src_migration_load > migrate_out_instance.migration_load_metric
        prev_src_migration_load = migrate_out_instance.migration_load_metric
        assert prev_dst_migration_load is None or prev_dst_migration_load < migrate_in_instance.migration_load_metric
        prev_dst_migration_load = migrate_in_instance.migration_load_metric

def test_failover_migration_policy():
    defrag_migration_policy: Failover = MigrationPolicyFactory.get_policy(
        'failover', migrate_out_load_threshold=MIGRATE_OUT_LOAD_THRESHOLD)
    all_instance_infos: Dict[str, InstanceInfo] = {}

    src_instance_infos = []
    for idx in range(INSTANCE_NUM//2):
        instance_info = InstanceInfo()
        instance_info.instance_id = f"src_instance_id_{idx}"
        all_instance_infos[instance_info.instance_id] = instance_info
        instance_info.unit_id = idx
        instance_info.unit_status = UnitStatus.FAILOVER_MIGRATING
        src_instance_infos.append(instance_info)

    dst_instance_infos = []
    for idx in range(INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_id = f"dst_instance_id_{idx}"
        all_instance_infos[instance_info.instance_id] = instance_info
        instance_info.unit_id = idx
        instance_info.unit_status = UnitStatus.HEALTH
        dst_instance_infos.append(instance_info)

    migrate_instance_pairs = defrag_migration_policy.pair_migration(src_instance_infos, dst_instance_infos)
    assert len(migrate_instance_pairs) == INSTANCE_NUM//2

    for _, migrate_in_instance_id in migrate_instance_pairs:
        assert all_instance_infos[migrate_in_instance_id].unit_id >= INSTANCE_NUM//2


class MockMigrationScheduler(MigrationScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_constraints_pairs = []
        self.pd_migration_pairs = []
        self.dd_migration_pairs = []
        self.dynamicp_d_migration_pairs = []
        self.dynamicp_dynamicp_migration_pairs = []
        self.dp_migration_pairs = []
        self.unit_failover_migration_pairs = []
        self.prefill_unit_failover_migration_pairs = []
        self.decode_unit_failover_migration_pairs = []

    def reset_store(self):
        self.no_constraints_pairs = []
        self.pd_migration_pairs = []
        self.dd_migration_pairs = []
        self.dynamicp_d_migration_pairs = []
        self.dynamicp_dynamicp_migration_pairs = []
        self.dp_migration_pairs = []
        self.unit_failover_migration_pairs = []
        self.prefill_unit_failover_migration_pairs = []
        self.decode_unit_failover_migration_pairs = []

    def _pair_migration(self,
                        instance_info: Dict[str, InstanceInfo],
                        migration_filter_pipeline: Optional[MigrationFilterPipeline],
                        migration_policy: MigrationPolicy,
                        skip_broken_unit: bool = True) -> List[Tuple[str, str]]:
        target_store = None
        if hasattr(self, 'p2d_transfer_filter_pipeline') \
            and migration_filter_pipeline == self.p2d_transfer_filter_pipeline:
            target_store = self.pd_migration_pairs
        elif hasattr(self, 'no_constraints_load_balance_filter_pipeline') \
            and migration_filter_pipeline == self.no_constraints_load_balance_filter_pipeline:
            target_store = self.no_constraints_pairs
        elif hasattr(self, 'decode_load_balance_filter_pipeline') \
            and migration_filter_pipeline == self.decode_load_balance_filter_pipeline:
            target_store = self.dd_migration_pairs
        elif hasattr(self, 'dynamic_p2d_filter_pipeline') \
            and migration_filter_pipeline == self.dynamic_p2d_filter_pipeline:
            target_store = self.dynamicp_d_migration_pairs
        elif hasattr(self, 'aggrate_dynamic_p_filter_pipeline') \
            and migration_filter_pipeline == self.aggrate_dynamic_p_filter_pipeline:
            target_store = self.dynamicp_dynamicp_migration_pairs
        elif hasattr(self, 'ease_d_with_empty_p_filter_pipeline') \
            and migration_filter_pipeline == self.ease_d_with_empty_p_filter_pipeline:
            target_store = self.dp_migration_pairs
        elif hasattr(self, 'unit_failover_pipeline') \
            and migration_filter_pipeline == self.unit_failover_pipeline:
            target_store = self.unit_failover_migration_pairs
        elif hasattr(self, 'prefill_unit_failover_pipeline') \
            and migration_filter_pipeline == self.prefill_unit_failover_pipeline:
            target_store = self.prefill_unit_failover_migration_pairs
        elif hasattr(self, 'decode_unit_failover_pipeline') \
            and migration_filter_pipeline == self.decode_unit_failover_pipeline:
            target_store = self.decode_unit_failover_migration_pairs
        migrate_instance_pairs = super()._pair_migration(instance_info, migration_filter_pipeline,
                                                         migration_policy, skip_broken_unit)
        target_store.extend(migrate_instance_pairs)

@pytest.mark.parametrize("enable_pd_disagg, enable_engine_pd_disagg, enable_engine_semi_pd_disagg",
                         [(True, False, False), (False, True, False), (False, False, True), (False, False, False)])
def test_migration_scheduler(enable_pd_disagg, enable_engine_pd_disagg, enable_engine_semi_pd_disagg):
    dispatch_load_metric_config = DispatchLoadMetricConfig(
            dispatch_load_metric='remaining_steps',
            dispatch_prefill_load_metric='kv_blocks_ratio',
            dispatch_decode_load_metric='remaining_steps',
            dispatch_prefill_as_decode_load_metric='adaptive_decode',
            dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
        )
    migration_scheduler = MockMigrationScheduler('defrag', -3.0, False, enable_pd_disagg,
                                                 enable_engine_pd_disagg, enable_engine_semi_pd_disagg, False, dispatch_load_metric_config)
    all_instance_infos: Dict[str, InstanceInfo] = {}
    if not migration_scheduler._enable_pd():
        for idx in range(INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_type = InstanceType.NEUTRAL
            instance_info.instance_id = f"instance_id_{idx}"
            all_instance_infos[instance_info.instance_id] = instance_info
            instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)
    else:
        for idx in range(INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_type = InstanceType.PREFILL
            instance_info.instance_id = f"prefill_instance_id_{idx}"
            all_instance_infos[instance_info.instance_id] = instance_info
            instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)

        for idx in range(INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_type = InstanceType.DECODE
            instance_info.instance_id = f"decode_instance_id_{idx}"
            all_instance_infos[instance_info.instance_id] = instance_info
            instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)

    migration_scheduler.push_migrations(all_instance_infos)

    if enable_pd_disagg:
        assert len(migration_scheduler.pd_migration_pairs) > 0
        for src_instance_id, dst_instance_id in migration_scheduler.pd_migration_pairs:
            src_instance = all_instance_infos[src_instance_id]
            dst_instance = all_instance_infos[dst_instance_id]
            assert src_instance.instance_type == InstanceType.PREFILL
            assert dst_instance.instance_type == InstanceType.DECODE

    if enable_pd_disagg or enable_engine_semi_pd_disagg:
        assert len(migration_scheduler.dd_migration_pairs) > 0
        for src_instance_id, dst_instance_id in migration_scheduler.dd_migration_pairs:
            src_instance = all_instance_infos[src_instance_id]
            dst_instance = all_instance_infos[dst_instance_id]
            assert src_instance.instance_type == InstanceType.DECODE
            assert dst_instance.instance_type == InstanceType.DECODE

    if enable_engine_pd_disagg:
        assert len(migration_scheduler.pd_migration_pairs) == 0
        assert len(migration_scheduler.dd_migration_pairs) == 0

    if not migration_scheduler._enable_pd():
        assert len(migration_scheduler.no_constraints_pairs) > 0
        assert len(migration_scheduler.pd_migration_pairs) == 0
        assert len(migration_scheduler.dd_migration_pairs) == 0

@pytest.mark.parametrize("enable_pd_disagg, enable_engine_semi_pd_disagg", [(True, False), (False, True)])
def test_adaptive_migration_scheduler(enable_pd_disagg, enable_engine_semi_pd_disagg):
    dispatch_load_metric_config = DispatchLoadMetricConfig(
            dispatch_load_metric='remaining_steps',
            dispatch_prefill_load_metric='kv_blocks_ratio',
            dispatch_decode_load_metric='remaining_steps',
            dispatch_prefill_as_decode_load_metric='adaptive_decode',
            dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
        )
    migration_scheduler = MockMigrationScheduler('defrag', -3.0, False, enable_pd_disagg,
                                                 False, enable_engine_semi_pd_disagg, True, dispatch_load_metric_config)
    KvBlocksRatioLoad.BUSY_THRESHOLD = 5
    RemainingStepsLoad.BUSY_THRESHOLD = 5
    AdaptiveDecodeBatchLoad.DECODE_COMPUTE_BOUND_BATCH_SIZE = 5

    normal_prefill_instance_infos = {}
    for idx in range(INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_type = InstanceType.PREFILL
        instance_info.instance_id = f"normal_prefill_instance_id_{idx}"
        normal_prefill_instance_infos[instance_info.instance_id] = instance_info
        instance_info.kv_blocks_ratio = KvBlocksRatioLoad(2)
        instance_info.migration_load_metric = RemainingStepsLoad(10)
        instance_info.num_running_requests = 10
        instance_info.decode_batch_size = 0
        instance_info.dispatch_prefill_as_decode_load_metric = AdaptiveDecodeBatchLoad(0)

    dynamic_prefill_instance_infos = {}
    for idx in range(INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_type = InstanceType.PREFILL
        instance_info.instance_id = f"dynamic_prefill_instance_id_{idx}"
        dynamic_prefill_instance_infos[instance_info.instance_id] = instance_info
        instance_info.kv_blocks_ratio = KvBlocksRatioLoad(idx)
        instance_info.migration_load_metric = RemainingStepsLoad(10)
        instance_info.num_running_requests = 10
        instance_info.decode_batch_size = 1
        instance_info.dispatch_prefill_as_decode_load_metric = AdaptiveDecodeBatchLoad(idx)

    normal_decode_instance_infos = {}
    for idx in range(INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_type = InstanceType.DECODE
        instance_info.instance_id = f"normal_decode_instance_id_{idx}"
        normal_decode_instance_infos[instance_info.instance_id] = instance_info
        instance_info.remaining_steps = RemainingStepsLoad(20)
        instance_info.migration_load_metric = RemainingStepsLoad(10)
        instance_info.num_running_requests = 10
        instance_info.decode_batch_size = 1

    busy_decode_instance_infos = {}
    for idx in range(INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_type = InstanceType.DECODE
        instance_info.instance_id = f"busy_decode_instance_id_{idx}"
        busy_decode_instance_infos[instance_info.instance_id] = instance_info
        instance_info.remaining_steps = RemainingStepsLoad(1)
        instance_info.migration_load_metric = RemainingStepsLoad(10)
        instance_info.num_running_requests = 10
        instance_info.decode_batch_size = 1

    # dynamic p and normal d
    all_instance: Dict[str, InstanceInfo] = {}
    all_instance.update(normal_prefill_instance_infos)
    all_instance.update(dynamic_prefill_instance_infos)
    all_instance.update(normal_decode_instance_infos)
    all_instance.update(busy_decode_instance_infos)
    migration_scheduler.push_migrations(all_instance)

    assert len(migration_scheduler.dynamicp_d_migration_pairs) > 0
    for migrate_out_instance_id, migrate_in_instance_id in migration_scheduler.dynamicp_d_migration_pairs:
        migrate_out_instance = all_instance[migrate_out_instance_id]
        migrate_in_instance = all_instance[migrate_in_instance_id]
        assert migrate_out_instance.decode_batch_size > 0 \
            and migrate_out_instance.instance_type == InstanceType.PREFILL
        assert not getattr(migrate_in_instance, migration_scheduler.dispatch_load_metric_config.dispatch_decode_load_metric).is_busy() \
            and migrate_in_instance.instance_type == InstanceType.DECODE
    migration_scheduler.reset_store()

    # dynamic p and busy d
    all_instance: Dict[str, InstanceInfo] = {}
    all_instance.update(normal_prefill_instance_infos)
    all_instance.update(dynamic_prefill_instance_infos)
    all_instance.update(busy_decode_instance_infos)
    migration_scheduler.push_migrations(all_instance)

    assert len(migration_scheduler.dynamicp_dynamicp_migration_pairs) > 0
    for migrate_out_instance_id, migrate_in_instance_id in migration_scheduler.dynamicp_dynamicp_migration_pairs:
        migrate_out_instance = all_instance[migrate_out_instance_id]
        migrate_in_instance = all_instance[migrate_in_instance_id]
        assert migrate_out_instance.decode_batch_size > 0 \
            and migrate_out_instance.instance_type == InstanceType.PREFILL
        assert migrate_in_instance.decode_batch_size > 0 \
            and migrate_in_instance.instance_type == InstanceType.PREFILL
        assert migrate_out_instance.decode_batch_size <= migrate_in_instance.decode_batch_size
    migration_scheduler.reset_store()

    # free p and busy d
    free_prefill_instance_infos = {}
    for idx in range(INSTANCE_NUM):
        instance_info = InstanceInfo()
        instance_info.instance_type = InstanceType.PREFILL
        instance_info.instance_id = f"free_prefill_instance_id_{idx}"
        free_prefill_instance_infos[instance_info.instance_id] = instance_info
        instance_info.kv_blocks_ratio = KvBlocksRatioLoad(0)
        instance_info.migration_load_metric = RemainingStepsLoad(10)
        instance_info.num_running_requests = 0
        instance_info.decode_batch_size = 0
        instance_info.dispatch_prefill_as_decode_load_metric = AdaptiveDecodeBatchLoad(0)

    all_instance: Dict[str, InstanceInfo] = {}
    all_instance.update(normal_prefill_instance_infos)
    all_instance.update(dynamic_prefill_instance_infos)
    all_instance.update(free_prefill_instance_infos)
    all_instance.update(busy_decode_instance_infos)
    migration_scheduler.push_migrations(all_instance)

    assert len(migration_scheduler.dp_migration_pairs) > 0
    for migrate_out_instance_id, migrate_in_instance_id in migration_scheduler.dp_migration_pairs:
        migrate_out_instance = all_instance[migrate_out_instance_id]
        migrate_in_instance = all_instance[migrate_in_instance_id]
        assert getattr(migrate_out_instance, migration_scheduler.dispatch_load_metric_config.dispatch_decode_load_metric).is_busy() \
            and migrate_out_instance.instance_type == InstanceType.DECODE
        assert migrate_in_instance.num_running_requests == 0 \
            and migrate_in_instance.instance_type == InstanceType.PREFILL

@pytest.mark.parametrize("enable_pd_disagg", [True, False])
def test_unit_failover_migration_scheduler(enable_pd_disagg):
    dispatch_load_metric_config = DispatchLoadMetricConfig(
        dispatch_load_metric='remaining_steps',
        dispatch_prefill_load_metric='kv_blocks_ratio',
        dispatch_decode_load_metric='remaining_steps',
        dispatch_prefill_as_decode_load_metric='adaptive_decode',
        dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
    )
    migration_scheduler = MockMigrationScheduler('defrag', -3.0, False, enable_pd_disagg,
                                                 False, False, False, dispatch_load_metric_config)
    all_instance_infos: Dict[str, InstanceInfo] = {}
    if not migration_scheduler._enable_pd():
        for idx in range(INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_type = InstanceType.NEUTRAL
            instance_info.instance_id = f"instance_id_{idx}"
            instance_info.unit_id = idx
            instance_info.unit_status = UnitStatus.HEALTH if idx % 2 == 0 else UnitStatus.BROKEN
            all_instance_infos[instance_info.instance_id] = instance_info
    else:
        for idx in range(INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_type = InstanceType.PREFILL
            instance_info.instance_id = f"prefill_instance_id_{idx}"
            instance_info.unit_id = idx
            instance_info.unit_status = UnitStatus.HEALTH if idx % 2 == 0 else UnitStatus.BROKEN
            all_instance_infos[instance_info.instance_id] = instance_info

        for idx in range(INSTANCE_NUM):
            instance_info = InstanceInfo()
            instance_info.instance_type = InstanceType.DECODE
            instance_info.instance_id = f"decode_instance_id_{idx}"
            instance_info.unit_id = idx
            instance_info.unit_status = UnitStatus.HEALTH if idx % 2 == 0 else UnitStatus.BROKEN
            all_instance_infos[instance_info.instance_id] = instance_info

    migration_scheduler.push_migrations(all_instance_infos)

    if enable_pd_disagg:
        assert len(migration_scheduler.unit_failover_migration_pairs) > 0
    else:
        assert len(migration_scheduler.prefill_unit_failover_migration_pairs) > 0
        assert len(migration_scheduler.decode_unit_failover_migration_pairs) > 0
