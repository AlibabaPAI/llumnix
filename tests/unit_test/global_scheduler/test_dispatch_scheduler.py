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

from llumnix.instance_info import InstanceLoadCalculator, InstanceInfo
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler


def init_dispatch_scheduler(policy='load'):
    instance_load_calculator = InstanceLoadCalculator('remaining_steps', True)
    dispatch_scheduler = DispatchScheduler(policy, 1, instance_load_calculator, 2)
    return dispatch_scheduler

@pytest.fixture
def dispatch_scheduler():
    dispatch_scheduler = init_dispatch_scheduler()
    yield dispatch_scheduler

@pytest.mark.parametrize("num_dispatch_instances", [1, 2, 3])
def test_add_instance_and_remove_instance(dispatch_scheduler, num_dispatch_instances):
    dispatch_scheduler.num_dispatch_instances = num_dispatch_instances
    dispatch_scheduler.add_instance('instance_1')
    assert dispatch_scheduler.num_instances == 1
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 1
    dispatch_scheduler.remove_instance('instance_1')
    assert dispatch_scheduler.num_instances == 0
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 0

    dispatch_scheduler.add_instance('instance_2')
    assert dispatch_scheduler.num_instances == 1
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 1
    dispatch_scheduler.add_instance('instance_3')
    assert dispatch_scheduler.num_instances == 2
    assert len(dispatch_scheduler.available_dispatch_instance_set) == min(2, dispatch_scheduler.num_dispatch_instances)

    dispatch_scheduler.remove_instance('instance_2')
    assert dispatch_scheduler.num_instances == 1
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 1
    dispatch_scheduler.remove_instance('instance_3')
    assert dispatch_scheduler.num_instances == 0

def test_dispatch_balanced():
    num_tests = 100
    instance_num = 4
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('balanced')
        instance_num_requests = {}
        for instance_id in [f'instance_{i}' for i in range(1, instance_num + 1)]:
            if len(dispatch_scheduler.available_dispatch_instance_set) < dispatch_scheduler.num_dispatch_instances:
                dispatch_scheduler.available_dispatch_instance_set.add(instance_id)
                instance_num_requests[instance_id] = random.randint(1, 10)
        dispatch_scheduler.instance_num_requests = instance_num_requests
        min_instance_id = next(key for key, value in sorted(instance_num_requests.items(), key=lambda item: item[1]))
        instance_id = dispatch_scheduler.dispatch()
        assert min_instance_id == instance_id

def test_dispatch_load():
    num_tests = 100
    instance_num = 4
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('load')
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, instance_num + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.instance_load_dispatch_scale = random.random()
            instance_info_dict[instance_id] = instance_info
            if dispatch_scheduler.num_dispatch_instances <= 0 or (dispatch_scheduler.num_dispatch_instances > 0
                and len(dispatch_scheduler.available_dispatch_instance_set) < dispatch_scheduler.num_dispatch_instances):
                dispatch_scheduler.available_dispatch_instance_set.add(instance_id)
                instance_num_requests[instance_id] = 0
        dispatch_scheduler.instance_num_requests = instance_num_requests
        dispatch_scheduler.instance_info = instance_info_dict
        available_instance_dict = {key: value for key, value in instance_info_dict.items()
                                   if key in dispatch_scheduler.available_dispatch_instance_set}
        min_instance_id = next(key for key, value in sorted(available_instance_dict.items(),
                                                            key=lambda item: item[1].instance_load_dispatch_scale))
        instance_id = dispatch_scheduler.dispatch()
        assert min_instance_id == instance_id

def test_dispatch_queue():
    num_tests = 100
    instance_num = 4
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('queue')
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, instance_num + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.num_waiting_requests = random.randint(1, 10)
            instance_info_dict[instance_id] = instance_info
            if len(dispatch_scheduler.available_dispatch_instance_set) < dispatch_scheduler.num_dispatch_instances:
                dispatch_scheduler.available_dispatch_instance_set.add(instance_id)
                instance_num_requests[instance_id] = 0
        dispatch_scheduler.instance_num_requests = instance_num_requests
        dispatch_scheduler.instance_info = instance_info_dict
        available_instance_dict = {key: value for key, value in instance_info_dict.items()
                                   if key in dispatch_scheduler.available_dispatch_instance_set}
        min_instance_id = next(key for key, value in sorted(available_instance_dict.items(),
                                                            key=lambda item: item[1].num_waiting_requests))
        instance_id = dispatch_scheduler.dispatch()
        assert instance_info_dict[min_instance_id].num_waiting_requests == instance_info_dict[instance_id].num_waiting_requests

def test_dispatch_rr():
    instance_num = 7
    instance_load_calculator = InstanceLoadCalculator('remaining_steps', True)
    dispatch_scheduler = DispatchScheduler('rr', 1, instance_load_calculator, 3)
    instance_num_requests = {}
    instance_info_dict = {}

    for instance_id in [f'instance_{i}' for i in range(instance_num)]:
        instance_info = InstanceInfo()
        instance_info.instance_id = instance_id
        instance_info.num_waiting_requests = random.randint(1, 10)
        instance_info_dict[instance_id] = instance_info
        if  len(dispatch_scheduler.available_dispatch_instance_set) < dispatch_scheduler.num_dispatch_instances:
            dispatch_scheduler.available_dispatch_instance_set.add(instance_id)
            instance_num_requests[instance_id] = 0
    dispatch_scheduler.instance_num_requests = instance_num_requests
    dispatch_scheduler.instance_info = instance_info_dict

    num_request = 2 * instance_num + 2
    for idx in range(0, num_request):
        instance_id = dispatch_scheduler.dispatch()
        target_instance_id = idx%dispatch_scheduler.num_dispatch_instances
        assert instance_id == f'instance_{target_instance_id}'

    for idx in range(instance_num):
        if idx < dispatch_scheduler.num_dispatch_instances:
            dispatch_scheduler.instance_num_requests[f'instance_{idx}'] = \
                num_request // dispatch_scheduler.num_dispatch_instances + (1 \
                    if num_request % dispatch_scheduler.num_dispatch_instances > \
                        idx % dispatch_scheduler.num_dispatch_instances else 0)
        else:
            dispatch_scheduler.instance_num_requests[f'instance_{idx}'] = 0

def test_dispatch_power_of_k_choice():
    instance_load_calculator = InstanceLoadCalculator('remaining_steps', True)
    num_tests = 100
    instance_num = 2
    for power_of_k_choice in [1, 2, 3]:
        dispatch_scheduler = DispatchScheduler('load', power_of_k_choice, instance_load_calculator, 2)
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, instance_num + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.num_waiting_requests = random.randint(1, 10)
            instance_info_dict[instance_id] = instance_info
            if len(dispatch_scheduler.available_dispatch_instance_set) < dispatch_scheduler.num_dispatch_instances:
                dispatch_scheduler.available_dispatch_instance_set.add(instance_id)
                instance_num_requests[instance_id] = 0
        dispatch_scheduler.instance_num_requests = instance_num_requests
        dispatch_scheduler.instance_info = instance_info_dict
        instance_id_set = set()
        for _ in range(num_tests):
            instance_id_set.add(dispatch_scheduler.dispatch())
    assert len(instance_id_set) == 2
