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
import pytest
import numpy as np

from llumnix.instance_info import InstanceLoadCalculator, InstanceInfo
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler, PairMigrationConstraints, InstanceType

MIGRATE_OUT_LOAD_THRESHOLD = 3.0
INSTANCE_NUM = 4

def init_migration_scheduler(policy='balanced', constraint_prefill_instance_num=-1):
    instance_load_calculator = InstanceLoadCalculator('remaining_steps', True)
    migration_scheduler = MigrationScheduler(policy, MIGRATE_OUT_LOAD_THRESHOLD, instance_load_calculator, constraint_prefill_instance_num)
    return migration_scheduler

@pytest.fixture
def migration_scheduler():
    migration_scheduler = init_migration_scheduler()
    yield migration_scheduler

def test_add_instance_and_remove_instance(migration_scheduler):
    migration_scheduler.add_instance('instance_1')
    assert migration_scheduler.num_instances == 1
    if migration_scheduler.constraint_prefill_instance_num <= 0:
        assert len(migration_scheduler.instance_id_type_set[InstanceType.NO_CONSTRAINTS]) == 1
    else:
        assert len(migration_scheduler.instance_id_type_set[InstanceType.PREFILL]) == 1
    migration_scheduler.add_instance('instance_2')
    assert migration_scheduler.num_instances == 2
    if migration_scheduler.constraint_prefill_instance_num <= 0:
        assert len(migration_scheduler.instance_id_type_set[InstanceType.NO_CONSTRAINTS]) == 2
    else:
        assert len(migration_scheduler.instance_id_type_set[InstanceType.PREFILL]) == min(2, migration_scheduler.constraint_prefill_instance_num)
        assert len(migration_scheduler.instance_id_type_set[InstanceType.DECODE]) == max(2 - migration_scheduler.constraint_prefill_instance_num, 0)
    migration_scheduler.remove_instance('instance_1')
    assert migration_scheduler.num_instances == 1
    migration_scheduler.remove_instance('instance_2')
    assert migration_scheduler.num_instances == 0

@pytest.mark.parametrize("pair_migration_type", ['NO_CONSTRAINTS','DECODING_2_DECODING','PREFILL_2_DECODING'])
def test_pair_migration(pair_migration_type):
    num_tests = 1000
    for _ in range(num_tests):
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, INSTANCE_NUM + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.instance_load_migrate = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-1, 1)
            instance_info.num_killed_requests = random.randint(0, 1)
            instance_info_dict[instance_id] = instance_info
            if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
                constraint_prefill_instance_num = -1
            else:
                constraint_prefill_instance_num = random.randint(-1, INSTANCE_NUM)
            migration_scheduler = init_migration_scheduler(constraint_prefill_instance_num=constraint_prefill_instance_num)
            if migration_scheduler.constraint_prefill_instance_num > 0:
                if len(migration_scheduler.instance_id_type_set[InstanceType.PREFILL]) < migration_scheduler.constraint_prefill_instance_num:
                    migration_scheduler.instance_id_type_set[InstanceType.PREFILL].add(instance_id)
                else:
                    migration_scheduler.instance_id_type_set[InstanceType.DECODE].add(instance_id)
            else:
                migration_scheduler.instance_id_type_set[InstanceType.NO_CONSTRAINTS].add(instance_id)
    migration_scheduler.instance_info = instance_info_dict
    sorted_src_instance_infos, sorted_dst_instance_infos = migration_scheduler._get_migration_settings(pair_migration_type)
    for instance in sorted_src_instance_infos:
        if pair_migration_type != PairMigrationConstraints.PREFILL_2_DECODING:
            assert instance.num_killed_requests > 0 \
                or instance.instance_load_migrate > MIGRATE_OUT_LOAD_THRESHOLD
            if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
                migration_scheduler._sort_instance_infos([InstanceType.NO_CONSTRAINTS], descending=False)
                assert instance in migration_scheduler.instance_id_type_set[InstanceType.NO_CONSTRAINTS]
            elif migration_scheduler == PairMigrationConstraints.DECODING_2_DECODING:
                migration_scheduler._sort_instance_infos([InstanceType.DECODE], descending=False)
                assert instance in migration_scheduler.instance_id_type_set[InstanceType.DECODE]
        else:
            migration_scheduler._sort_instance_infos([InstanceType.PREFILL, InstanceType.DECODE], descending=False)
            assert instance in migration_scheduler.instance_id_type_set[InstanceType.PREFILL]
    for instance in sorted_dst_instance_infos:
        if pair_migration_type != PairMigrationConstraints.PREFILL_2_DECODING:
            assert instance.num_killed_requests == 0 and instance.instance_load_migrate < MIGRATE_OUT_LOAD_THRESHOLD
            if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
                assert instance in migration_scheduler.instance_id_type_set[InstanceType.NO_CONSTRAINTS]
            elif migration_scheduler == PairMigrationConstraints.DECODING_2_DECODING:
                assert instance in migration_scheduler.instance_id_type_set[InstanceType.DECODE]
        else:
            assert instance in migration_scheduler.instance_id_type_set[InstanceType.DECODE]
            assert instance.num_killed_requests == 0

@pytest.mark.parametrize("policy", ['balanced','defrag_constrained'])
def test_pair_migration(policy):
    num_tests = 1000
    for _ in range(num_tests):
        migration_scheduler = init_migration_scheduler(policy)
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, INSTANCE_NUM + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.instance_load_migrate = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-1, 1)
            instance_info.num_killed_requests = random.randint(0, 1)
            instance_info.num_blocks_last_running_request = random.randint(0, 1) * np.inf
            instance_info_dict[instance_id] = instance_info
            migration_scheduler.instance_id_type_set[InstanceType.NO_CONSTRAINTS].add(instance_id)
        migration_scheduler.instance_info = instance_info_dict
        migration_scheduler._sort_instance_infos(descending=False)
        sorted_src_instance_infos = [i for i in reversed(migration_scheduler.sorted_instance_infos[InstanceType.NO_CONSTRAINTS])
                                if i.num_killed_requests > 0 or i.instance_load_migrate > migration_scheduler.migrate_out_load_threshold]
        sorted_dst_instance_infos = [i for i in migration_scheduler.sorted_instance_infos[InstanceType.NO_CONSTRAINTS]
                                         if i.num_killed_requests == 0 and i.instance_load_migrate < migration_scheduler.migrate_out_load_threshold]
        migrate_instance_pairs = migration_scheduler.pair_migration_policy.pair_migration(sorted_src_instance_infos, sorted_dst_instance_infos, True)
        for migrate_out_instance, migrate_in_instance in migrate_instance_pairs:
            assert migrate_out_instance != migrate_in_instance
            if policy == 'balanced':
                assert instance_info_dict[migrate_out_instance].num_blocks_last_running_request == 0
            if instance_info_dict[migrate_out_instance].num_killed_requests == 0:
                assert instance_info_dict[migrate_out_instance].instance_load_migrate > instance_info_dict[migrate_in_instance].instance_load_migrate
