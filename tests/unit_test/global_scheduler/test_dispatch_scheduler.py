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

from collections import defaultdict
import random
from typing import Dict

import pytest
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler
from llumnix.global_scheduler.dispatch_policy import DispatchLoadMetricConfig
from llumnix.load_computation import KvBlocksRatioLoad, RemainingStepsLoad, AdaptiveDecodeBatchLoad, MissWaitingTokensLoad


def init_dispatch_scheduler(
        policy='load',
        topk_random_dispatch=1,
        enable_pd_disagg=False,
        enable_engine_pd_disagg=False,
        enable_engine_semi_pd_disagg=False,
        enable_adaptive_pd=False,
        dispatch_load_metric_config=None,
        cache_aware_query_client_config_path=None):
    dispatch_scheduler = DispatchScheduler(
        dispatch_policy=policy,
        topk_random_dispatch=topk_random_dispatch,
        enable_pd_disagg=enable_pd_disagg,
        enable_engine_pd_disagg=enable_engine_pd_disagg,
        enable_engine_semi_pd_disagg=enable_engine_semi_pd_disagg,
        enable_adaptive_pd=enable_adaptive_pd,
        dispatch_load_metric_config=dispatch_load_metric_config,
        cache_aware_query_client_config_path=cache_aware_query_client_config_path,
    )
    return dispatch_scheduler


def init_instances(num_instance: int = 4, instance_type: InstanceType = InstanceType.NO_CONSTRAINTS):
    instance_info_dict: Dict[str, InstanceInfo] = {}
    instance_num_requests: Dict[str, int] = {}
    for i in range(num_instance):
        instance_info = InstanceInfo(
            instance_id=f"instance_{instance_type.value}_{i}",
            kv_blocks_ratio=KvBlocksRatioLoad(random.uniform(1, 10)),
            remaining_steps=RemainingStepsLoad(random.uniform(1, 10)),
            adaptive_decode=AdaptiveDecodeBatchLoad(random.uniform(1, 10)),
            miss_waiting_tokens=MissWaitingTokensLoad(random.uniform(1, 10)),
            num_waiting_requests=random.uniform(1, 10),
            instance_type=instance_type,
        )
        instance_info_dict[instance_info.instance_id] = instance_info
        instance_num_requests[instance_info.instance_id] = 0

    return instance_info_dict, instance_num_requests

def init_pd_instances(num_prefill: int = 4, num_decode: int = 4):
    instance_info_dict: Dict[str, InstanceInfo] = {}
    instance_num_requests: Dict[str, int] = {}

    prefill_instance_info_dict: Dict[str, InstanceInfo] = {}
    prefill_instance_num_requests: Dict[str, int] = {}
    for i in range(num_prefill):
        instance_info = InstanceInfo(
            instance_id=f"instance_{InstanceType.PREFILL.value}_{i}",
            kv_blocks_ratio=KvBlocksRatioLoad(random.uniform(1, 9)),
            remaining_steps=RemainingStepsLoad(random.uniform(1, 9)),
            adaptive_decode=AdaptiveDecodeBatchLoad(random.uniform(1, 9)),
            miss_waiting_tokens=MissWaitingTokensLoad(random.uniform(1, 9)),
            num_waiting_requests=random.uniform(1, 9),
            instance_type=InstanceType.PREFILL,
        )
        prefill_instance_info_dict[instance_info.instance_id] = instance_info
        prefill_instance_num_requests[instance_info.instance_id] = 0
        instance_info_dict[instance_info.instance_id] = instance_info
        instance_num_requests[instance_info.instance_id] = 0

    decode_instance_info_dict: Dict[str, InstanceInfo] = {}
    decode_instance_num_requests: Dict[str, int] = {}
    for i in range(num_decode):
        instance_info = InstanceInfo(
            instance_id=f"instance_{InstanceType.DECODE.value}_{i}",
            kv_blocks_ratio=KvBlocksRatioLoad(random.uniform(1, 9)),
            remaining_steps=RemainingStepsLoad(random.uniform(1, 9)),
            adaptive_decode=AdaptiveDecodeBatchLoad(random.uniform(1, 9)),
            miss_waiting_tokens=MissWaitingTokensLoad(random.uniform(1, 9)),
            num_waiting_requests=random.uniform(1, 9),
            instance_type=InstanceType.DECODE,
        )
        decode_instance_info_dict[instance_info.instance_id] = instance_info
        decode_instance_num_requests[instance_info.instance_id] = 0
        instance_info_dict[instance_info.instance_id] = instance_info
        instance_num_requests[instance_info.instance_id] = 0
    
    return instance_info_dict, instance_num_requests, prefill_instance_info_dict, \
        prefill_instance_num_requests, decode_instance_info_dict, decode_instance_num_requests

def test_dispatch_no_constraints():
    dispatch_load_metric_config = DispatchLoadMetricConfig(
        dispatch_load_metric='remaining_steps',
        dispatch_prefill_load_metric='kv_blocks_ratio',
        dispatch_decode_load_metric='remaining_steps',
        dispatch_prefill_as_decode_load_metric='adaptive_decode',
        dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
    )
    dispatch_scheduler = init_dispatch_scheduler(policy="rr", enable_pd_disagg=False, dispatch_load_metric_config=dispatch_load_metric_config)
    instance_num: int = 4
    instance_info_dict, instance_num_requests = init_instances(instance_num)

    expected_dispatched_instance_ids = instance_info_dict.keys()

    instance_dispatch_info = defaultdict(int)
    for _ in range(instance_num * 2):
        instance_id = dispatch_scheduler.dispatch_no_constrains(
            instance_info_dict, instance_num_requests, {}
        )
        instance_dispatch_info[instance_id] += 1

    for instance_id, num_requests in instance_dispatch_info.items():
        assert instance_id in expected_dispatched_instance_ids
        assert num_requests == instance_num_requests[instance_id]

@pytest.mark.parametrize("enable_pd_disagg", [True, False])
def test_dispatch_pd(enable_pd_disagg):
    dispatch_load_metric_config = DispatchLoadMetricConfig(
        dispatch_load_metric='remaining_steps',
        dispatch_prefill_load_metric='kv_blocks_ratio',
        dispatch_decode_load_metric='remaining_steps',
        dispatch_prefill_as_decode_load_metric='adaptive_decode',
        dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
    )
    dispatch_scheduler: DispatchScheduler = init_dispatch_scheduler(
        policy="rr", enable_pd_disagg=enable_pd_disagg, enable_engine_pd_disagg=not enable_pd_disagg, enable_adaptive_pd=False, dispatch_load_metric_config=dispatch_load_metric_config)
    instance_num: int = 4
    instance_info_dict, instance_num_requests, prefill_instance_info_dict, prefill_instance_num_requests, \
        decode_instance_info_dict, decode_instance_num_requests = init_pd_instances(instance_num, instance_num)

    expected_dispatched_instance_ids = instance_info_dict.keys()

    prefill_instance_dispatch_info = defaultdict(int)
    decode_instance_dispatch_info = defaultdict(int)
    for _ in range(instance_num * 2):
        prefill_instance_id, decode_instance_id = dispatch_scheduler.dispatch_pd(
            instance_infos=instance_info_dict,
            instance_num_requests=instance_num_requests,
            prefill_instance_infos=prefill_instance_info_dict,
            prefill_instance_num_requests=prefill_instance_num_requests,
            decode_instance_infos=decode_instance_info_dict,
            decode_instance_num_requests=decode_instance_num_requests,
            dispatch_context={}
        )
        prefill_instance_dispatch_info[prefill_instance_id] += 1
        decode_instance_dispatch_info[decode_instance_id] += 1
        assert not enable_pd_disagg or decode_instance_id is None

    for instance_id, num_requests in prefill_instance_dispatch_info.items():
        assert instance_id in expected_dispatched_instance_ids
        assert num_requests == prefill_instance_num_requests[instance_id] == instance_num_requests[instance_id]

    if not enable_pd_disagg:
        for instance_id, num_requests in decode_instance_dispatch_info.items():
            assert instance_id in expected_dispatched_instance_ids
            assert num_requests == decode_instance_num_requests[instance_id] == instance_num_requests[instance_id]

# @pytest.mark.parametrize("enable_pd_disagg", [True, False])
@pytest.mark.parametrize("enable_pd_disagg", [False])
def test_dispatch_adaptive_pd(enable_pd_disagg):
    dispatch_load_metric_config = DispatchLoadMetricConfig(
        dispatch_load_metric='remaining_steps',
        dispatch_prefill_load_metric='kv_blocks_ratio',
        dispatch_decode_load_metric='remaining_steps',
        dispatch_prefill_as_decode_load_metric='adaptive_decode',
        dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
    )
    dispatch_scheduler = init_dispatch_scheduler(
        policy='load',
        enable_pd_disagg=enable_pd_disagg,
        enable_engine_pd_disagg=not enable_pd_disagg,
        enable_adaptive_pd=True,
        dispatch_load_metric_config=dispatch_load_metric_config,
    )
    instance_num: int = 4
    instance_info_dict, instance_num_requests, prefill_instance_info_dict, prefill_instance_num_requests, \
        decode_instance_info_dict, decode_instance_num_requests = init_pd_instances(instance_num, instance_num)

    instance_dispatch_info = defaultdict(int)
    # ----- choose prefill tests -----

    # exist free prefill, choose prefill
    KvBlocksRatioLoad.BUSY_THRESHOLD = 10
    RemainingStepsLoad.BUSY_THRESHOLD = -1
    prefill_instance_id, decode_instance_id = dispatch_scheduler.dispatch_pd(
        instance_infos=instance_info_dict,
        instance_num_requests=instance_num_requests,
        prefill_instance_infos=prefill_instance_info_dict,
        prefill_instance_num_requests=prefill_instance_num_requests,
        decode_instance_infos=decode_instance_info_dict,
        decode_instance_num_requests=decode_instance_num_requests,
        dispatch_context={},
    )
    assert not enable_pd_disagg or decode_instance_id is None
    instance_dispatch_info[prefill_instance_id] += 1
    instance_dispatch_info[decode_instance_id] += 1
    lowest_load_prefill_instance_id = min(prefill_instance_info_dict,
                                          key=lambda x: getattr(prefill_instance_info_dict[x], dispatch_load_metric_config.dispatch_prefill_load_metric))
    lowest_load_decode_instance_id = min(decode_instance_info_dict,
                                          key=lambda x: getattr(decode_instance_info_dict[x], dispatch_load_metric_config.dispatch_decode_load_metric))
    assert prefill_instance_id == lowest_load_prefill_instance_id
    assert enable_pd_disagg or decode_instance_id == lowest_load_decode_instance_id
    assert instance_dispatch_info[prefill_instance_id] == instance_num_requests[prefill_instance_id]
    assert instance_dispatch_info[prefill_instance_id] == prefill_instance_num_requests[prefill_instance_id]
    if not enable_pd_disagg:
        assert instance_dispatch_info[decode_instance_id] == instance_num_requests[decode_instance_id]
        assert instance_dispatch_info[decode_instance_id] == decode_instance_num_requests[decode_instance_id]

    # no free prefill, choose decode
    KvBlocksRatioLoad.BUSY_THRESHOLD = -1
    RemainingStepsLoad.BUSY_THRESHOLD = -1
    prefill_instance_id, decode_instance_id = dispatch_scheduler.dispatch_pd(
        instance_infos=instance_info_dict,
        instance_num_requests=instance_num_requests,
        prefill_instance_infos=prefill_instance_info_dict,
        prefill_instance_num_requests=prefill_instance_num_requests,
        decode_instance_infos=decode_instance_info_dict,
        decode_instance_num_requests=decode_instance_num_requests,
        dispatch_context={},
    )
    assert not enable_pd_disagg or decode_instance_id is None
    instance_dispatch_info[prefill_instance_id] += 1
    instance_dispatch_info[decode_instance_id] += 1
    lowest_load_decode_as_prefill_instance_id = min(
        decode_instance_info_dict,key=lambda x: getattr(decode_instance_info_dict[x], dispatch_load_metric_config.dispatch_decode_as_prefill_load_metric))
    lowest_load_decode_instance_id = min(
        decode_instance_info_dict, key=lambda x: getattr(decode_instance_info_dict[x], dispatch_load_metric_config.dispatch_decode_load_metric))
    assert prefill_instance_id == lowest_load_decode_as_prefill_instance_id
    assert enable_pd_disagg or decode_instance_id == prefill_instance_id
    assert prefill_instance_id not in prefill_instance_info_dict
    assert instance_dispatch_info[prefill_instance_id] == instance_num_requests[prefill_instance_id]
    assert instance_dispatch_info[prefill_instance_id] == decode_instance_num_requests[prefill_instance_id]
    if not enable_pd_disagg:
        assert instance_dispatch_info[decode_instance_id] == instance_num_requests[decode_instance_id]
        assert instance_dispatch_info[decode_instance_id] == decode_instance_num_requests[decode_instance_id]

    # no free prefill, no free decode, choose busy prefill
    KvBlocksRatioLoad.BUSY_THRESHOLD = -1
    RemainingStepsLoad.BUSY_THRESHOLD = 10
    prefill_instance_id, decode_instance_id = dispatch_scheduler.dispatch_pd(
        instance_infos=instance_info_dict,
        instance_num_requests=instance_num_requests,
        prefill_instance_infos=prefill_instance_info_dict,
        prefill_instance_num_requests=prefill_instance_num_requests,
        decode_instance_infos=decode_instance_info_dict,
        decode_instance_num_requests=decode_instance_num_requests,
        dispatch_context={},
    )
    assert not enable_pd_disagg or decode_instance_id is None
    instance_dispatch_info[prefill_instance_id] += 1
    instance_dispatch_info[decode_instance_id] += 1
    lowest_load_prefill_instance_id = min(prefill_instance_info_dict,
                                          key=lambda x: getattr(prefill_instance_info_dict[x], dispatch_load_metric_config.dispatch_prefill_load_metric))
    lowest_load_decode_instance_id = min(decode_instance_info_dict,
                                          key=lambda x: getattr(decode_instance_info_dict[x], dispatch_load_metric_config.dispatch_decode_load_metric))
    assert prefill_instance_id == lowest_load_prefill_instance_id
    assert enable_pd_disagg or decode_instance_id == lowest_load_decode_instance_id
    assert instance_dispatch_info[prefill_instance_id] == instance_num_requests[prefill_instance_id]
    assert instance_dispatch_info[prefill_instance_id] == prefill_instance_num_requests[prefill_instance_id]
    if not enable_pd_disagg:
        assert instance_dispatch_info[decode_instance_id] == instance_num_requests[decode_instance_id]
        assert instance_dispatch_info[decode_instance_id] == decode_instance_num_requests[decode_instance_id]
    assert getattr(prefill_instance_info_dict[prefill_instance_id], dispatch_load_metric_config.dispatch_prefill_load_metric).is_busy()

    # ----- choose decode tests -----

    # exist free decode, choose decode
    KvBlocksRatioLoad.BUSY_THRESHOLD = 10
    RemainingStepsLoad.BUSY_THRESHOLD = -1
    prefill_instance_id, decode_instance_id = dispatch_scheduler.dispatch_pd(
        instance_infos=instance_info_dict,
        instance_num_requests=instance_num_requests,
        prefill_instance_infos=prefill_instance_info_dict,
        prefill_instance_num_requests=prefill_instance_num_requests,
        decode_instance_infos=decode_instance_info_dict,
        decode_instance_num_requests=decode_instance_num_requests,
        dispatch_context={},
    )
    assert not enable_pd_disagg or decode_instance_id is None
    instance_dispatch_info[prefill_instance_id] += 1
    instance_dispatch_info[decode_instance_id] += 1
    lowest_load_prefill_instance_id = min(
        prefill_instance_info_dict, key=lambda x: getattr(prefill_instance_info_dict[x], dispatch_load_metric_config.dispatch_prefill_load_metric))
    lowest_load_decode_instance_id = min(
        decode_instance_info_dict, key=lambda x: getattr(decode_instance_info_dict[x], dispatch_load_metric_config.dispatch_decode_load_metric))
    assert prefill_instance_id == lowest_load_prefill_instance_id
    assert enable_pd_disagg or decode_instance_id == lowest_load_decode_instance_id
    assert instance_dispatch_info[prefill_instance_id] == instance_num_requests[prefill_instance_id]
    assert instance_dispatch_info[prefill_instance_id] == prefill_instance_num_requests[prefill_instance_id]
    if not enable_pd_disagg:
        assert instance_dispatch_info[decode_instance_id] == instance_num_requests[decode_instance_id]
        assert instance_dispatch_info[decode_instance_id] == decode_instance_num_requests[decode_instance_id]

    # no free decode, choose prefill
    KvBlocksRatioLoad.BUSY_THRESHOLD = 10
    RemainingStepsLoad.BUSY_THRESHOLD = 10
    prefill_instance_id, decode_instance_id = dispatch_scheduler.dispatch_pd(
        instance_infos=instance_info_dict,
        instance_num_requests=instance_num_requests,
        prefill_instance_infos=prefill_instance_info_dict,
        prefill_instance_num_requests=prefill_instance_num_requests,
        decode_instance_infos=decode_instance_info_dict,
        decode_instance_num_requests=decode_instance_num_requests,
        dispatch_context={},
    )
    assert not enable_pd_disagg or decode_instance_id is None
    instance_dispatch_info[prefill_instance_id] += 1
    instance_dispatch_info[decode_instance_id] += 1
    lowest_load_prefill_instance_id = min(
        prefill_instance_info_dict, key=lambda x: getattr(prefill_instance_info_dict[x], dispatch_load_metric_config.dispatch_prefill_load_metric))
    lowest_load_prefill_as_decode_instance_id = min(
        prefill_instance_info_dict, key=lambda x: getattr(prefill_instance_info_dict[x], dispatch_load_metric_config.dispatch_prefill_as_decode_load_metric))
    assert prefill_instance_id == lowest_load_prefill_instance_id
    assert enable_pd_disagg or decode_instance_id == lowest_load_prefill_as_decode_instance_id
    assert instance_dispatch_info[prefill_instance_id] == instance_num_requests[prefill_instance_id]
    assert instance_dispatch_info[prefill_instance_id] == prefill_instance_num_requests[prefill_instance_id]
    if not enable_pd_disagg:
        assert instance_dispatch_info[decode_instance_id] == instance_num_requests[decode_instance_id]
        assert instance_dispatch_info[decode_instance_id] == prefill_instance_num_requests[decode_instance_id]

    # no free decode, no free prefill, choose busy decode
    KvBlocksRatioLoad.BUSY_THRESHOLD = -1
    RemainingStepsLoad.BUSY_THRESHOLD = 10
    prefill_instance_id, decode_instance_id = dispatch_scheduler.dispatch_pd(
        instance_infos=instance_info_dict,
        instance_num_requests=instance_num_requests,
        prefill_instance_infos=prefill_instance_info_dict,
        prefill_instance_num_requests=prefill_instance_num_requests,
        decode_instance_infos=decode_instance_info_dict,
        decode_instance_num_requests=decode_instance_num_requests,
        dispatch_context={},
    )
    assert not enable_pd_disagg or decode_instance_id is None
    instance_dispatch_info[prefill_instance_id] += 1
    instance_dispatch_info[decode_instance_id] += 1
    lowest_load_prefill_instance_id = min(
        prefill_instance_info_dict, key=lambda x: getattr(prefill_instance_info_dict[x], dispatch_load_metric_config.dispatch_prefill_load_metric))
    lowest_load_decode_instance_id = min(
        decode_instance_info_dict, key=lambda x: getattr(decode_instance_info_dict[x], dispatch_load_metric_config.dispatch_decode_load_metric))
    assert prefill_instance_id == lowest_load_prefill_instance_id
    assert enable_pd_disagg or decode_instance_id == lowest_load_decode_instance_id
    assert instance_dispatch_info[prefill_instance_id] == instance_num_requests[prefill_instance_id]
    assert instance_dispatch_info[prefill_instance_id] == prefill_instance_num_requests[prefill_instance_id]
    if not enable_pd_disagg:
        assert instance_dispatch_info[decode_instance_id] == instance_num_requests[decode_instance_id]
        assert instance_dispatch_info[decode_instance_id] == decode_instance_num_requests[decode_instance_id]
        assert getattr(decode_instance_info_dict[decode_instance_id], dispatch_load_metric_config.dispatch_decode_load_metric).is_busy()

def test_dispatch_balanced():
    num_tests = 100
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('balanced')
        instance_info_dict, instance_num_requests = init_instances()
        instance_id_set = instance_info_dict.keys()
        for instance_id in instance_id_set:
            instance_num_requests[instance_id] = random.randint(1, 10)
        min_instance_id = next(key for key, _ in sorted(instance_num_requests.items(), key=lambda item: item[1]))
        instance_id = dispatch_scheduler.dispatch_no_constrains(
            instance_infos=instance_info_dict, instance_num_requests=instance_num_requests, dispatch_context={})
        instance_num_requests[instance_id] += 1
        assert min_instance_id == instance_id

def test_dispatch_load():
    num_tests = 100
    dispatch_load_metric_config = DispatchLoadMetricConfig(
        dispatch_load_metric='remaining_steps',
        dispatch_prefill_load_metric='kv_blocks_ratio',
        dispatch_decode_load_metric='remaining_steps',
        dispatch_prefill_as_decode_load_metric='adaptive_decode',
        dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
    )
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('load', dispatch_load_metric_config=dispatch_load_metric_config)
        instance_info_dict, instance_num_requests= init_instances()
        min_instance_id = next(key for key, _ in sorted(instance_info_dict.items(),
                                                            key=lambda item: getattr(item[1], dispatch_load_metric_config.dispatch_load_metric)))
        instance_id = dispatch_scheduler.dispatch_no_constrains(
            instance_infos=instance_info_dict, instance_num_requests=instance_num_requests, dispatch_context={})
        assert min_instance_id == instance_id

def test_dispatch_queue():
    num_tests = 100
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('queue')
        instance_info_dict, instance_num_requests = init_instances()

        min_instance_id = next(key for key, _ in sorted(instance_info_dict.items(),
                                                            key=lambda item: item[1].num_waiting_requests))
        instance_id = dispatch_scheduler.dispatch_no_constrains(
            instance_infos=instance_info_dict, instance_num_requests=instance_num_requests, dispatch_context={})
        assert instance_info_dict[min_instance_id].num_waiting_requests == instance_info_dict[instance_id].num_waiting_requests

def test_dispatch_rr():
    instance_num = 7
    dispatch_scheduler = init_dispatch_scheduler('rr')
    instance_info_dict, instance_num_requests = init_instances(instance_num)

    num_request = 2 * instance_num + 1
    for idx in range(0, num_request):
        instance_id = dispatch_scheduler.dispatch_no_constrains(
            instance_infos=instance_info_dict, instance_num_requests=instance_num_requests, dispatch_context={})
        target_instance_id = idx%instance_num
        assert instance_id == f'instance_{InstanceType.NO_CONSTRAINTS.value}_{target_instance_id}'

@pytest.mark.parametrize("topk_random_dispatch", [1, 2, 3])
def test_dispatch_topk_random_dispatch(topk_random_dispatch):
    num_tests = 200
    instance_num = 4
    dispatch_load_metric_config = DispatchLoadMetricConfig(
        dispatch_load_metric='remaining_steps',
        dispatch_prefill_load_metric='kv_blocks_ratio',
        dispatch_decode_load_metric='remaining_steps',
        dispatch_prefill_as_decode_load_metric='adaptive_decode',
        dispatch_decode_as_prefill_load_metric='kv_blocks_ratio',
    )
    dispatch_scheduler = init_dispatch_scheduler(policy='load', topk_random_dispatch=topk_random_dispatch, dispatch_load_metric_config=dispatch_load_metric_config)
    instance_info_dict, instance_num_requests = init_instances(instance_num)
    instance_id_set = set()
    for _ in range(num_tests):
        target_instance_id = dispatch_scheduler.dispatch_no_constrains(
            instance_infos=instance_info_dict, instance_num_requests=instance_num_requests, dispatch_context={})
        instance_id_set.add(target_instance_id)
    assert len(instance_id_set) == topk_random_dispatch
