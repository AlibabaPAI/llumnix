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
from llumnix.arg_utils import InstanceArgs

INSTANCE_NUM = 4

def init_dispatch_scheduler(policy='load'):
    instance_load_calculator = InstanceLoadCalculator('remaining_steps', True)
    dispatch_scheduler = DispatchScheduler(policy, instance_load_calculator)
    return dispatch_scheduler

@pytest.fixture
def dispatch_scheduler():
    dispatch_scheduler = init_dispatch_scheduler()
    yield dispatch_scheduler

def test_add_instance_and_remove_instance(dispatch_scheduler):
    dispatch_scheduler.add_instance('instance_1', InstanceArgs(instance_type="no_constraints"))
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 1
    dispatch_scheduler.remove_instance('instance_1')
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 0

    dispatch_scheduler.add_instance('instance_2', InstanceArgs(instance_type="no_constraints"))
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 1
    dispatch_scheduler.add_instance('instance_3', InstanceArgs(instance_type="no_constraints"))
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 2

    dispatch_scheduler.remove_instance('instance_2')
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 1
    dispatch_scheduler.remove_instance('instance_3')

def test_dispatch_balanced():
    num_tests = 100
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('balanced')
        instance_num_requests = {}
        for instance_id in [f'instance_{i}' for i in range(1, INSTANCE_NUM + 1)]:
            dispatch_scheduler.available_dispatch_instance_set.add(instance_id)
            instance_num_requests[instance_id] = random.randint(1, 10)
        dispatch_scheduler.instance_num_requests = instance_num_requests
        min_instance_id = next(key for key, value in sorted(instance_num_requests.items(), key=lambda item: item[1]))
        instance_id = dispatch_scheduler.dispatch()
        assert min_instance_id == instance_id

def test_dispatch_load():
    num_tests = 100
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('load')
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, INSTANCE_NUM + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.instance_load_dispatch_scale = random.random()
            instance_info_dict[instance_id] = instance_info
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
    for _ in range(num_tests):
        dispatch_scheduler = init_dispatch_scheduler('queue')
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, INSTANCE_NUM + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.num_waiting_requests = random.randint(1, 10)
            instance_info_dict[instance_id] = instance_info
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
    dispatch_scheduler = init_dispatch_scheduler("rr")
    instance_num_requests = {}
    instance_info_dict = {}

    for instance_id in [f'instance_{i}' for i in range(instance_num)]:
        instance_info = InstanceInfo()
        instance_info.instance_id = instance_id
        instance_info.num_waiting_requests = random.randint(1, 10)
        instance_info_dict[instance_id] = instance_info
        dispatch_scheduler.available_dispatch_instance_set.add(instance_id)
        instance_num_requests[instance_id] = 0
    dispatch_scheduler.instance_num_requests = instance_num_requests
    dispatch_scheduler.instance_info = instance_info_dict

    num_request = 2 * instance_num + 2
    for idx in range(0, num_request):
        instance_id = dispatch_scheduler.dispatch()
        target_instance_id = idx%instance_num
        assert instance_id == f'instance_{target_instance_id}'
