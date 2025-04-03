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
from typing import Dict, Set
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler


def init_dispatch_scheduler(policy='load'):
    dispatch_scheduler = DispatchScheduler(policy, 1)
    return dispatch_scheduler


def init_instances(prefill_instance_num: int = 4, decode_instance_num: int = 0):
    instance_id_set: Set[str] = set()
    instance_type_id_set: Dict[InstanceType, Set[str]] = {
        instance_type: set() for instance_type in InstanceType
    }
    instance_info_dict: Dict[str, InstanceInfo] = {}
    instance_num_requests: Dict[str, int] = {}
    for i in range(prefill_instance_num):
        instance_info = InstanceInfo(
            instance_id=f"instance_{InstanceType.PREFILL.value}_{i}",
            dispatch_load_metric=random.randint(1, 10),
            num_waiting_requests=random.randint(1, 10),
            instance_type=InstanceType.PREFILL,
        )
        instance_info_dict[instance_info.instance_id] = instance_info
        instance_type_id_set[InstanceType.PREFILL].add(instance_info.instance_id)
        instance_id_set.add(instance_info.instance_id)
        instance_num_requests[instance_info.instance_id] = 0
    for i in range(decode_instance_num):
        instance_info = InstanceInfo(
            instance_id=f"instance_{InstanceType.DECODE.value}_{i}",
            dispatch_load_metric=random.randint(1, 10),
            num_waiting_requests=random.randint(1, 10),
            instance_type=InstanceType.DECODE,
        )
        instance_info_dict[instance_info.instance_id] = instance_info
        instance_type_id_set[InstanceType.DECODE].add(instance_info.instance_id)
        instance_id_set.add(instance_info.instance_id)
        instance_num_requests[instance_info.instance_id] = 0
    return (
        instance_id_set,
        instance_info_dict,
        instance_num_requests,
        instance_type_id_set,
    )


def test_dispatch_to_no_constraints_and_prefill():
    dispatch_scheduler = init_dispatch_scheduler("rr")
    instance_num: int = 4
    _, instance_info_dict, instance_num_requests, _ = init_instances(instance_num)

    prefill_instance_info_dict = {
        instance_id: instance_info
        for instance_id, instance_info in instance_info_dict.items()
        if instance_info.instance_type == InstanceType.PREFILL
    }
    prefill_instnace_num_requests = {
        instance_id: instance_num_requests[instance_id]
        for instance_id in prefill_instance_info_dict.keys()
    }

    expected_dispatched_instance_ids = [
        instance_id
        for instance_id, instance_info in instance_info_dict.items()
        if instance_info.instance_type == InstanceType.PREFILL
    ]

    instance_dispatch_info = defaultdict(int)
    prefill_instance_num = instance_num
    for _ in range(prefill_instance_num * 2):
        instance_id = dispatch_scheduler.dispatch(
            prefill_instance_info_dict, prefill_instnace_num_requests
        )
        prefill_instnace_num_requests[instance_id] += 1
        instance_dispatch_info[instance_id] += 1

    for instance_id, num_requests in instance_dispatch_info.items():
        assert instance_id in expected_dispatched_instance_ids
        assert num_requests == 2


def test_dispatch_balanced():
    num_tests = 100
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('balanced')
        instance_id_set, instance_info_dict, instance_num_requests, _ = init_instances()
        for instance_id in instance_id_set:
            instance_num_requests[instance_id] = random.randint(1, 10)
        min_instance_id = next(key for key, _ in sorted(instance_num_requests.items(), key=lambda item: item[1]))
        instance_id = dispatch_scheduler.dispatch(instance_info=instance_info_dict,instance_num_requests=instance_num_requests)
        instance_num_requests[instance_id] += 1
        assert min_instance_id == instance_id


def test_dispatch_balanced_decode_instance():
    num_tests = 100
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler("balanced")
        _, instance_info_dict, instance_num_requests, _ = init_instances(4,4)
        decode_instance_info_dict = {
            instance_id: instance_info
            for instance_id, instance_info in instance_info_dict.items()
            if instance_info.instance_type == InstanceType.DECODE
        }
        decode_instnace_num_requests = {
            instance_id: instance_num_requests[instance_id]
            for instance_id in decode_instance_info_dict.keys()
        }
        for instance_id in decode_instance_info_dict.keys():
            decode_instnace_num_requests[instance_id] = random.randint(1, 10)
        min_instance_id = next(
            key
            for key, _ in sorted(
                decode_instnace_num_requests.items(), key=lambda item: item[1]
            )
        )
        instance_id = dispatch_scheduler.dispatch(
            instance_info=decode_instance_info_dict,
            instance_num_requests=decode_instnace_num_requests,
        )
        decode_instnace_num_requests[instance_id] += 1
        assert min_instance_id == instance_id


def test_dispatch_load():
    num_tests = 100
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('load')
        _, instance_info_dict, instance_num_requests, _ = init_instances()
        min_instance_id = next(key for key, _ in sorted(instance_info_dict.items(),
                                                            key=lambda item: item[1].dispatch_load_metric))
        instance_id = dispatch_scheduler.dispatch(instance_info=instance_info_dict, instance_num_requests=instance_num_requests)
        assert min_instance_id == instance_id

def test_dispatch_queue():
    num_tests = 100
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('queue')
        _, instance_info_dict, instance_num_requests, _ = init_instances()

        min_instance_id = next(key for key, _ in sorted(instance_info_dict.items(),
                                                            key=lambda item: item[1].num_waiting_requests))
        instance_id = dispatch_scheduler.dispatch(instance_info=instance_info_dict, instance_num_requests=instance_num_requests)
        assert instance_info_dict[min_instance_id].num_waiting_requests == instance_info_dict[instance_id].num_waiting_requests

def test_dispatch_rr():
    instance_num = 7
    dispatch_scheduler = init_dispatch_scheduler('rr')
    _, instance_info_dict, instance_num_requests, _ = init_instances(instance_num)

    num_request = 2 * instance_num + 1
    for idx in range(0, num_request):
        instance_id = dispatch_scheduler.dispatch(instance_info=instance_info_dict, instance_num_requests=instance_num_requests)
        target_instance_id = idx%instance_num
        assert instance_id == f'instance_prefill_{target_instance_id}'

def test_dispatch_topk_random_dispatch():
    num_tests = 100
    instance_num = 4
    for topk_random_dispatch in [1, 2, 3]:
        dispatch_scheduler = DispatchScheduler('load', topk_random_dispatch)
        _, instance_info_dict, instance_num_requests, _ = init_instances(instance_num)
        instance_id_set = set()
        for _ in range(num_tests):
            instance_id_set.add(dispatch_scheduler.dispatch(instance_info=instance_info_dict, instance_num_requests=instance_num_requests))
        assert len(instance_id_set) == topk_random_dispatch
