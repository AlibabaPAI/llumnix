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
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler


MIGRATE_OUT_LOAD_THRESHOLD = 3.0


def init_migration_scheduler(policy='balanced'):
    instance_load_calculator = InstanceLoadCalculator('remaining_steps', True)
    migration_scheduler = MigrationScheduler(policy, MIGRATE_OUT_LOAD_THRESHOLD, instance_load_calculator)
    return migration_scheduler

@pytest.fixture
def migration_scheduler():
    migration_scheduler = init_migration_scheduler()
    yield migration_scheduler

def test_add_instance_and_remove_instance(migration_scheduler):
    migration_scheduler.add_instance('instance_1')
    assert migration_scheduler.num_instances == 1
    migration_scheduler.add_instance('instance_2')
    assert migration_scheduler.num_instances == 2
    migration_scheduler.remove_instance('instance_1')
    assert migration_scheduler.num_instances == 1
    migration_scheduler.remove_instance('instance_2')
    assert migration_scheduler.num_instances == 0

@pytest.mark.parametrize("policy", ['balanced', 'defrag_constrained', 'defrag_relaxed'])
def test_pair_migration(policy):
    migration_scheduler = init_migration_scheduler(policy)
    num_tests = 1000
    for _ in range(num_tests):
        instance_info_dict = {}
        for instance_id in ['instance_1', 'instance_2', 'instance_3', 'instance_4']:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.instance_load_migrate = MIGRATE_OUT_LOAD_THRESHOLD + random.uniform(-1, 1)
            instance_info.num_killed_requests = random.randint(0, 1)
            instance_info.num_blocks_last_running_request = random.randint(0, 1) * np.inf
            instance_info_dict[instance_id] = instance_info
        migration_scheduler.instance_info = instance_info_dict
        migrate_instance_pairs = migration_scheduler.pair_migration()
        for migrate_out_instance, migrate_in_instance in migrate_instance_pairs:
            assert migrate_out_instance != migrate_in_instance
            if policy != 'defrag_relaxed':
                assert instance_info_dict[migrate_out_instance].num_killed_requests > 0 \
                    or instance_info_dict[migrate_out_instance].instance_load_migrate > MIGRATE_OUT_LOAD_THRESHOLD
            assert instance_info_dict[migrate_in_instance].num_killed_requests == 0 \
                and instance_info_dict[migrate_in_instance].instance_load_migrate < MIGRATE_OUT_LOAD_THRESHOLD
            if policy == 'balanced':
                assert instance_info_dict[migrate_out_instance].num_blocks_last_running_request == 0
            if instance_info_dict[migrate_out_instance].num_killed_requests == 0:
                assert instance_info_dict[migrate_out_instance].instance_load_migrate > instance_info_dict[migrate_in_instance].instance_load_migrate
