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
    dispatch_scheduler = DispatchScheduler(policy, instance_load_calculator)
    return dispatch_scheduler

@pytest.fixture
def dispatch_scheduler():
    dispatch_scheduler = init_dispatch_scheduler()
    yield dispatch_scheduler

def test_add_instance_and_remove_instance(dispatch_scheduler):
    dispatch_scheduler.add_instance('instance_1')
    assert dispatch_scheduler.num_instances == 1
    dispatch_scheduler.add_instance('instance_2')
    assert dispatch_scheduler.num_instances == 2
    dispatch_scheduler.remove_instance('instance_1')
    assert dispatch_scheduler.num_instances == 1
    dispatch_scheduler.remove_instance('instance_2')
    assert dispatch_scheduler.num_instances == 0

def test_dispatch_balanced():
    dispatch_scheduler = init_dispatch_scheduler('balanced')
    num_tests = 100
    for _ in range(num_tests):
        instance_num_requests = {}
        for instance_id in ['instance_1', 'instance_2', 'instance_3', 'instance_4']:
            instance_num_requests[instance_id] = random.randint(1, 10)
        dispatch_scheduler.instance_num_requests = instance_num_requests
        min_instance_id = next(key for key, value in sorted(instance_num_requests.items(), key=lambda item: item[1]))
        instance_id = dispatch_scheduler.dispatch()
        assert min_instance_id == instance_id

def test_dispatch_load():
    dispatch_scheduler = init_dispatch_scheduler('load')
    num_tests = 100
    for _ in range(num_tests):
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in ['instance_1', 'instance_2', 'instance_3', 'instance_4']:
            instance_num_requests[instance_id] = 0
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.instance_load_dispatch_scale = random.random()
            instance_info_dict[instance_id] = instance_info
        dispatch_scheduler.instance_num_requests = instance_num_requests
        dispatch_scheduler.instance_info = instance_info_dict
        min_instance_id = next(key for key, value in sorted(instance_info_dict.items(),
                                                            key=lambda item: item[1].instance_load_dispatch_scale))
        instance_id = dispatch_scheduler.dispatch()
        assert min_instance_id == instance_id

def test_dispatch_queue():
    dispatch_scheduler = init_dispatch_scheduler('queue')
    num_tests = 100
    for _ in range(num_tests):
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in ['instance_1', 'instance_2', 'instance_3', 'instance_4']:
            instance_num_requests[instance_id] = 0
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.num_waiting_requests = random.randint(1, 10)
            instance_info_dict[instance_id] = instance_info
        dispatch_scheduler.instance_num_requests = instance_num_requests
        dispatch_scheduler.instance_info = instance_info_dict
        min_instance_id = next(key for key, value in sorted(instance_info_dict.items(),
                                                            key=lambda item: item[1].num_waiting_requests))
        instance_id = dispatch_scheduler.dispatch()
        assert instance_info_dict[min_instance_id].num_waiting_requests == instance_info_dict[instance_id].num_waiting_requests
