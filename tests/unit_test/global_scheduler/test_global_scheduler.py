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
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.utils import random_uuid
from llumnix.load_computation import KvBlocksRatioLoad, RemainingStepsLoad


def init_global_scheduler(
        enable_pd_disagg: bool = False,
        enable_engine_pd_disagg: bool = False,
        enable_adaptive_pd: bool = False,
        enable_engine_semi_pd_disagg = False):
    global_scheduler_config = GlobalSchedulerConfig(
        initial_instances=0,
        dispatch_policy="load",
        topk_random_dispatch=1,
        pair_migration_policy="defrag",
        migrate_out_threshold=3,
        scaling_policy="avg_load",
        scaling_load_metric="remaining_steps",
        scale_up_threshold=10,
        scale_down_threshold=60,
        enable_pd_disagg=enable_pd_disagg,
        enable_engine_pd_disagg=enable_engine_pd_disagg,
        enable_engine_semi_pd_disagg=enable_engine_semi_pd_disagg,
        enable_adaptive_pd=enable_adaptive_pd,
        is_group_kind_migration_backend=False)
    global_scheduler = GlobalScheduler(global_scheduler_config)
    return global_scheduler

def init_instance_infos(initial_instances, instance_type = InstanceType.NO_CONSTRAINTS):
    instance_infos = []
    for _ in range(initial_instances):
        instance_id = random_uuid()
        instance_info = InstanceInfo(instance_type=instance_type)
        instance_info.instance_id = instance_id
        instance_infos.append(instance_info)
    return instance_infos

@pytest.fixture
def global_scheduler():
    global_scheduler = init_global_scheduler()
    yield global_scheduler

@pytest.mark.asyncio
async def test_add_instance_and_remove_instance():
    global_scheduler = init_global_scheduler(enable_pd_disagg=True)
    # test prefill instance
    global_scheduler.scale_up('instance_1', InstanceType.NO_CONSTRAINTS)
    assert global_scheduler.num_instances == 1
    assert len(global_scheduler.instance_info) == 1
    assert len(global_scheduler.instance_id_set) == 1
    assert len(global_scheduler.prefill_instance_info) == 1
    assert len(global_scheduler.decode_instance_info) == 1
    assert global_scheduler.prefill_instance_num_requests['instance_1'] == 0
    assert global_scheduler.decode_instance_num_requests['instance_1'] == 0
    assert 'instance_1' in global_scheduler.instance_id_set

    global_scheduler.scale_down('instance_1')
    assert global_scheduler.num_instances == 0
    assert len(global_scheduler.instance_info) == 0
    assert len(global_scheduler.instance_id_set) == 0
    assert len(global_scheduler.prefill_instance_info) == 0
    assert len(global_scheduler.decode_instance_info) == 0
    assert global_scheduler.prefill_instance_num_requests.get("instance_1", None) is None
    assert global_scheduler.decode_instance_num_requests.get("instance_1", None) is None

    global_scheduler.scale_up('instance_2', InstanceType.PREFILL)
    assert len(global_scheduler.prefill_instance_num_requests) == 1
    assert global_scheduler.prefill_instance_num_requests['instance_2'] == 0
    assert len(global_scheduler.prefill_instance_info) == 1
    assert len(global_scheduler.instance_id_set) == 1
    global_scheduler.scale_up('instance_3', InstanceType.PREFILL)
    assert len(global_scheduler.prefill_instance_num_requests) == 2
    assert global_scheduler.prefill_instance_num_requests['instance_3'] == 0
    assert len(global_scheduler.prefill_instance_info) == 2
    assert len(global_scheduler.instance_id_set) == 2

    global_scheduler.scale_down('instance_2')
    assert len(global_scheduler.prefill_instance_info) == 1
    assert len(global_scheduler.instance_info) == 1
    global_scheduler.scale_down('instance_3')
    assert len(global_scheduler.prefill_instance_info) == 0
    assert len(global_scheduler.instance_info) == 0

    # test decode instance
    global_scheduler.scale_up('instance_1', InstanceType.DECODE)
    assert len(global_scheduler.decode_instance_info) == 1
    assert len(global_scheduler.prefill_instance_info) == 0
    global_scheduler.scale_down('instance_1')
    assert global_scheduler.num_instances == 0
    assert len(global_scheduler.decode_instance_num_requests) == 0
    assert len(global_scheduler.instance_info) == 0
    assert len(global_scheduler.instance_id_set) == 0
    assert len(global_scheduler.prefill_instance_info) == 0
    assert len(global_scheduler.decode_instance_info) == 0

    global_scheduler.scale_up('instance_2', InstanceType.DECODE)
    assert len(global_scheduler.decode_instance_info) == 1
    global_scheduler.scale_up('instance_3', InstanceType.DECODE)
    assert len(global_scheduler.decode_instance_info) == 2
    assert global_scheduler.num_instances == 2
    assert len(global_scheduler.decode_instance_num_requests) == 2
    assert len(global_scheduler.prefill_instance_num_requests) == 0
    assert len(global_scheduler.instance_info) == 2
    assert len(global_scheduler.instance_id_set) == 2

    global_scheduler.scale_down('instance_2')
    assert len(global_scheduler.instance_id_set) == 1
    global_scheduler.scale_down('instance_3')
    assert len(global_scheduler.instance_id_set) == 0

    initial_instances = 4
    instance_infos = init_instance_infos(initial_instances)
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    num_instances = global_scheduler.scale_up(instance_ids, [InstanceType.NO_CONSTRAINTS]*len(instance_ids))
    assert num_instances == initial_instances
    instance_infos = init_instance_infos(initial_instances)
    instance_ids_1 = [instance_info.instance_id for instance_info in instance_infos]
    num_instances = global_scheduler.scale_down(instance_ids_1)
    assert num_instances == initial_instances
    num_instances = global_scheduler.scale_down(instance_ids)
    assert num_instances == 0

@pytest.mark.asyncio
async def test_update_instance_infos():
    global_scheduler = init_global_scheduler(enable_pd_disagg=True)
    initial_instances = 4
    instance_infos = init_instance_infos(initial_instances, InstanceType.NO_CONSTRAINTS)
    global_scheduler.update_instance_infos(instance_infos)
    assert len(global_scheduler.instance_id_set) == 0
    assert len(global_scheduler.instance_info) == 0
    assert len(global_scheduler.prefill_instance_info) == 0
    assert len(global_scheduler.decode_instance_info) == 0
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    global_scheduler.scale_up(instance_ids, [InstanceType.NO_CONSTRAINTS]*len(instance_ids))
    global_scheduler.update_instance_infos(instance_infos)
    assert len(global_scheduler.instance_id_set) == initial_instances
    assert len(global_scheduler.instance_info) == initial_instances
    assert len(global_scheduler.prefill_instance_info) == initial_instances
    assert len(global_scheduler.decode_instance_info) == initial_instances
    for instance_id in instance_ids:
        assert instance_id in global_scheduler.instance_id_set and \
               instance_id in global_scheduler.instance_info and \
               instance_id in global_scheduler.instance_num_requests and \
               instance_id in global_scheduler.prefill_instance_info and \
               instance_id in global_scheduler.decode_instance_info
        instance_info_1 = global_scheduler.instance_info[instance_id]
        instance_info_2 = global_scheduler.prefill_instance_info[instance_id]
        instance_info_3 = global_scheduler.decode_instance_info[instance_id]
        assert instance_info_1 == instance_info_2 == instance_info_3

    instance_infos = init_instance_infos(initial_instances, InstanceType.PREFILL)
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    global_scheduler.scale_up(instance_ids, [InstanceType.PREFILL]*len(instance_ids))
    global_scheduler.update_instance_infos(instance_infos)
    assert len(global_scheduler.instance_id_set) == initial_instances * 2
    assert len(global_scheduler.instance_info) == initial_instances * 2
    assert len(global_scheduler.prefill_instance_info) == initial_instances * 2
    assert len(global_scheduler.decode_instance_info) == initial_instances
    for instance_id in instance_ids:
        assert instance_id in global_scheduler.instance_id_set and \
               instance_id in global_scheduler.instance_info and \
               instance_id in global_scheduler.prefill_instance_info and \
               instance_id not in global_scheduler.decode_instance_info
        instance_info_1 = global_scheduler.instance_info[instance_id]
        instance_info_2 = global_scheduler.prefill_instance_info[instance_id]
        assert instance_info_1 == instance_info_2

    instance_infos = init_instance_infos(initial_instances, InstanceType.DECODE)
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    global_scheduler.scale_up(instance_ids, [InstanceType.DECODE]*len(instance_ids))
    global_scheduler.update_instance_infos(instance_infos)
    assert len(global_scheduler.instance_id_set) == initial_instances * 3
    assert len(global_scheduler.instance_info) == initial_instances * 3
    assert len(global_scheduler.prefill_instance_info) == initial_instances * 2
    assert len(global_scheduler.decode_instance_info) == initial_instances * 2
    for instance_id in instance_ids:
        assert instance_id in global_scheduler.instance_id_set and \
               instance_id in global_scheduler.instance_info and \
               instance_id not in global_scheduler.prefill_instance_info and \
               instance_id in global_scheduler.decode_instance_info
        instance_info_1 = global_scheduler.instance_info[instance_id]
        instance_info_2 = global_scheduler.decode_instance_info[instance_id]
        assert instance_info_1 == instance_info_2

@pytest.mark.asyncio
async def test_dispatch_and_expected_steps(global_scheduler: GlobalScheduler):
    initial_instances = 4
    instance_infos = init_instance_infos(initial_instances)
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    global_scheduler.scale_up(instance_ids, [InstanceType.NO_CONSTRAINTS]*len(instance_ids))
    global_scheduler.update_instance_infos(instance_infos)
    instance_id, _, request_expected_steps = global_scheduler.dispatch(0)
    assert instance_id in instance_ids
    assert request_expected_steps == math.inf

@pytest.mark.asyncio
async def test_dispatch_pd_disagg_and_expected_steps():
    global_scheduler: GlobalScheduler = init_global_scheduler(enable_pd_disagg=True)
    initial_instances = 4
    instance_infos = init_instance_infos(initial_instances, InstanceType.NO_CONSTRAINTS)
    instance_ids = [instance_info.instance_id for instance_info in instance_infos]
    global_scheduler.scale_up(instance_ids, [InstanceType.NO_CONSTRAINTS]*len(instance_ids))
    global_scheduler.update_instance_infos(instance_infos)

    target_instance_id, _, request_expected_steps, = global_scheduler.dispatch(0)
    assert target_instance_id in instance_ids
    assert request_expected_steps == 1

@pytest.mark.asyncio
async def test_dispatch_adaptive_pd_disagg_and_expected_steps():
    global_scheduler = init_global_scheduler(enable_pd_disagg=True, enable_adaptive_pd=True)

    prefill_instance_id = random_uuid()
    prefill_instance_info = InstanceInfo(instance_id=prefill_instance_id, instance_type=InstanceType.PREFILL)
    KvBlocksRatioLoad.BUSY_THRESHOLD = 100
    prefill_instance_info.dispatch_load_metric = KvBlocksRatioLoad(10)
    global_scheduler.scale_up(prefill_instance_id, InstanceType.PREFILL)
    global_scheduler.update_instance_infos([prefill_instance_info])

    decode_instance_id = random_uuid()
    decode_instance_info = InstanceInfo(instance_id=decode_instance_id, instance_type=InstanceType.DECODE)
    RemainingStepsLoad.BUSY_THRESHOLD = 0
    decode_instance_info.dispatch_load_metric = RemainingStepsLoad(10)
    global_scheduler.scale_up(decode_instance_id, InstanceType.DECODE)
    global_scheduler.update_instance_infos([decode_instance_info])

    target_instance_id, _, request_expected_steps = global_scheduler.dispatch(0)
    assert target_instance_id == prefill_instance_id
    assert request_expected_steps == 1

    KvBlocksRatioLoad.BUSY_THRESHOLD = 0
    target_instance_id, _, request_expected_steps = global_scheduler.dispatch(0)
    assert target_instance_id == decode_instance_id
    assert request_expected_steps == math.inf
