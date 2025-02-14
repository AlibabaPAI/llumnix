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

import math
import random
import pytest

from llumnix.instance_info import InstanceInfo
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler
from llumnix.instance_info import InstanceType
from llumnix.global_scheduler.migration_filter import MigrationInstanceFilter, MigrationFilterConfig
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints
from llumnix.arg_utils import InstanceArgs

MIGRATE_OUT_LOAD_THRESHOLD = -3.0
INSTANCE_NUM = 16

def init_migration_scheduler(policy='balanced'):
    migration_scheduler = MigrationScheduler(policy, MIGRATE_OUT_LOAD_THRESHOLD, False)
    return migration_scheduler

def test_add_instance_and_remove_instance():
    migration_scheduler = init_migration_scheduler('balanced')
    migration_scheduler.add_instance('instance_1', InstanceArgs(instance_type="no_constraints"))
    assert migration_scheduler.num_instances == 1
    migration_scheduler.add_instance('instance_2', InstanceArgs(instance_type="no_constraints"))
    assert migration_scheduler.num_instances == 2
    migration_scheduler.remove_instance('instance_1')
    assert migration_scheduler.num_instances == 1
    migration_scheduler.remove_instance('instance_2')
    assert migration_scheduler.num_instances == 0

@pytest.mark.parametrize("pair_migration_type", ['NO_CONSTRAINTS', 'DECODING_2_DECODING', 'PREFILL_2_DECODING'])
def test_migration_filter(pair_migration_type):
    num_tests = 1000
    migration_filter = MigrationInstanceFilter(MigrationFilterConfig(MIGRATE_OUT_LOAD_THRESHOLD))

    for _ in range(num_tests):
        instance_infos = []

        total_prefill_instance_num = 0

        for instance_id in range(1, INSTANCE_NUM + 1):
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-1, 1)
            instance_info.num_killed_requests = random.randint(0, 1)

            if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
                constraint_prefill_instance_num = math.inf
            else:
                constraint_prefill_instance_num = random.randint(1, INSTANCE_NUM)

            if constraint_prefill_instance_num == math.inf:
                instance_info.instance_type = InstanceType.NO_CONSTRAINTS
            else:
                if total_prefill_instance_num < constraint_prefill_instance_num:
                    instance_info.instance_type = InstanceType.PREFILL
                    total_prefill_instance_num += 1
                else:
                    instance_info.instance_type = InstanceType.DECODE

            instance_infos.append(instance_info)

        src_instance_infos, dst_instance_infos = migration_filter.filter_instances(instance_infos, pair_migration_type)

        for instance in src_instance_infos:
            if pair_migration_type != PairMigrationConstraints.PREFILL_2_DECODING:
                assert instance.num_killed_requests > 0 \
                    or instance.migration_load_metric > MIGRATE_OUT_LOAD_THRESHOLD
                if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
                    assert instance.instance_type == InstanceType.NO_CONSTRAINTS
                elif pair_migration_type == PairMigrationConstraints.DECODING_2_DECODING:
                    assert instance.instance_type == InstanceType.DECODE
            else:
                assert instance.instance_type == InstanceType.PREFILL

        for instance in dst_instance_infos:
            if pair_migration_type != PairMigrationConstraints.PREFILL_2_DECODING:
                assert instance.num_killed_requests == 0 and instance.migration_load_metric < MIGRATE_OUT_LOAD_THRESHOLD
                if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
                    assert instance.instance_type == InstanceType.NO_CONSTRAINTS
                elif pair_migration_type == PairMigrationConstraints.DECODING_2_DECODING:
                    assert instance.instance_type == InstanceType.DECODE
            else:
                assert instance.instance_type == InstanceType.DECODE
                assert instance.num_killed_requests == 0

@pytest.mark.parametrize("policy", ['balanced', 'defrag'])
def test_pair_migration(policy):
    num_tests = 1000
    exist_migration = False

    for _ in range(num_tests):
        migration_scheduler = init_migration_scheduler(policy)
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, INSTANCE_NUM + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.migration_load_metric = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-6, 3)
            instance_info.migration_load_metric_after_migrate_out = instance_info.migration_load_metric - random.uniform(0, 1)
            instance_info.migration_load_metric_after_migrate_in = instance_info.migration_load_metric + random.uniform(0, 1)
            instance_info.num_killed_requests = random.randint(0, 1)
            instance_info.instance_type = InstanceType.NO_CONSTRAINTS
            instance_info_dict[instance_id] = instance_info
        migration_scheduler.instance_info = instance_info_dict

        migrate_instance_pairs = migration_scheduler.pair_migration(PairMigrationConstraints.NO_CONSTRAINTS)
        exist_migration = exist_migration or len(migrate_instance_pairs) > 0

        for migrate_out_instance, migrate_in_instance in migrate_instance_pairs:
            assert migrate_out_instance != migrate_in_instance
            if policy == 'balanced':
                assert instance_info_dict[migrate_out_instance].num_blocks_last_running_request == 0
            if instance_info_dict[migrate_out_instance].num_killed_requests == 0:
                assert instance_info_dict[migrate_out_instance].migration_load_metric > instance_info_dict[migrate_in_instance].migration_load_metric

    assert exist_migration
