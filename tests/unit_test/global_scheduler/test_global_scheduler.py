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
import pytest

from llumnix.internal_config import GlobalSchedulerConfig
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.instance_info import InstanceInfo
from llumnix.utils import random_uuid

from .test_manager import get_instance_info_migrate_in, get_instance_info_migrate_out


def init_global_scheduler():
    global_scheduler_config = GlobalSchedulerConfig(0, 'remaining_steps', 'load', 1, math.inf,
                                                    'defrag_constrained', 3.0, True, 'avg_load',
                                                    10, 60, False, 'rayrpc')
    global_scheduler = GlobalScheduler(global_scheduler_config)
    return global_scheduler

def init_instance_infos(initial_instances):
    instance_infos = []
    for _ in range(initial_instances):
        instance_id = random_uuid()
        instance_info = InstanceInfo()
        instance_info.instance_id = instance_id
        instance_infos.append(instance_info)
    return instance_infos

@pytest.fixture
def global_scheduler():
    global_scheduler = init_global_scheduler()
    yield global_scheduler

def test_scale_up_and_scale_down(global_scheduler):
    initial_instances = 4
    instance_infos = init_instance_infos(initial_instances)
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    num_instances = global_scheduler.scale_up(instance_ids)
    assert num_instances == initial_instances
    instance_infos = init_instance_infos(initial_instances)
    instance_ids_1 = [instance_info.instance_id for instance_info in instance_infos]
    num_instances = global_scheduler.scale_down(instance_ids_1)
    assert num_instances == initial_instances
    num_instances = global_scheduler.scale_down(instance_ids)
    assert num_instances == 0

def test_update_instance_infos(global_scheduler):
    initial_instances = 4
    instance_infos = init_instance_infos(initial_instances)
    global_scheduler.update_instance_infos(instance_infos)
    assert len(global_scheduler.instance_info) == 0
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    global_scheduler.scale_up(instance_ids)
    global_scheduler.update_instance_infos(instance_infos)
    assert len(global_scheduler.instance_info) == initial_instances

def test_dispatch(global_scheduler):
    initial_instances = 4
    instance_infos = init_instance_infos(initial_instances)
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    global_scheduler.scale_up(instance_ids)
    global_scheduler.update_instance_infos(instance_infos)
    instance_id, request_expected_steps = global_scheduler.dispatch()
    assert instance_id in instance_ids
    assert request_expected_steps == math.inf

def test_pair_migration(global_scheduler):
    instance_id = random_uuid()
    instance_id_1 = random_uuid()
    instance_ids = [instance_id, instance_id_1]
    instance_info_migrate_in = get_instance_info_migrate_in(instance_id)
    instance_info_migrate_out = get_instance_info_migrate_out(instance_id_1)
    instance_infos = [instance_info_migrate_in, instance_info_migrate_out]
    global_scheduler.scale_up(instance_ids)
    global_scheduler.update_instance_infos(instance_infos)
    migrate_instace_pairs = global_scheduler.pair_migration("NO_CONSTRAINTS")
    assert migrate_instace_pairs[0][0] == instance_id_1
    assert migrate_instace_pairs[0][1] == instance_id
