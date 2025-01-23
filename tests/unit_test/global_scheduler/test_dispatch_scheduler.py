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

from llumnix.instance_info import InstanceInfo
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler
from llumnix.arg_utils import InstanceArgs


def test_add_instance_and_remove_instance():
    dispatch_scheduler = DispatchScheduler('balanced')
def init_dispatch_scheduler(policy='load'):
    instance_load_calculator = InstanceLoadCalculator('remaining_steps', True)
    dispatch_scheduler = DispatchScheduler(policy, 1, instance_load_calculator, 2)
    return dispatch_scheduler

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
    assert len(dispatch_scheduler.available_dispatch_instance_set) == 0

def test_dispatch_to_no_constraints_and_prefill():
    dispatch_scheduler = DispatchScheduler('rr')
    instance_num_requests = {}
    instance_info_dict = {}
    for instance_id in [f'instance_{i}' for i in range(INSTANCE_NUM)]:
        instance_info = InstanceInfo(
            instance_id=instance_id,
            dispatch_load_metric=random.randint(1, 10),
        )
        instance_info_dict[instance_id] = instance_info
    dispatch_scheduler.instance_num_requests = instance_num_requests
    dispatch_scheduler.instance_info = instance_info_dict

    dispatched_instance_ids = []
    available_instance_type = ['no_constraints', 'prefill', 'decode']
    for instance_id, _ in dispatch_scheduler.instance_info.items():
        instance_type = random.choice(available_instance_type)
        dispatch_scheduler.add_instance(instance_id, InstanceArgs(instance_type=instance_type))
        if instance_type != 'decode':
            dispatched_instance_ids.append(instance_id)
        else:
            assert instance_id not in dispatch_scheduler.available_dispatch_instance_set

    instance_dispatch_info = defaultdict(int)
    for _ in range(INSTANCE_NUM * 2):
        instance_id = dispatch_scheduler.dispatch()
        instance_dispatch_info[instance_id] += 1

    for instance_id, num_requests in instance_dispatch_info.items():
        assert instance_id in dispatched_instance_ids
        assert num_requests >= 2

def test_dispatch_balanced():
    num_tests = 100
    instance_num = 4
    for _ in range(num_tests):
        dispatch_scheduler = DispatchScheduler('balanced')
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
    instance_num = 4
    for _ in range(num_tests):
        dispatch_scheduler = DispatchScheduler('load')
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, instance_num + 1)]:
            instance_info = InstanceInfo()
            instance_info.instance_id = instance_id
            instance_info.dispatch_load_metric = random.random()
            instance_info_dict[instance_id] = instance_info
            dispatch_scheduler.available_dispatch_instance_set.add(instance_id)
            instance_num_requests[instance_id] = 0
        dispatch_scheduler.instance_num_requests = instance_num_requests
        dispatch_scheduler.instance_info = instance_info_dict
        available_instance_dict = {key: value for key, value in instance_info_dict.items()
                                   if key in dispatch_scheduler.available_dispatch_instance_set}
        min_instance_id = next(key for key, value in sorted(available_instance_dict.items(),
                                                            key=lambda item: item[1].dispatch_load_metric))
        instance_id = dispatch_scheduler.dispatch()
        assert min_instance_id == instance_id

def test_dispatch_queue():
    num_tests = 100
    instance_num = 4
    for _ in range(num_tests):
        dispatch_scheduler = DispatchScheduler('queue')
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, 4 + 1)]:
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
    dispatch_scheduler = DispatchScheduler('rr', 3)
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

    num_request = 2 * instance_num + 1
    for idx in range(0, num_request):
        instance_id = dispatch_scheduler.dispatch()
        target_instance_id = idx%instance_num
        assert instance_id == f'instance_{target_instance_id}'

def test_dispatch_power_of_k_choice():
    instance_load_calculator = InstanceLoadCalculator('remaining_steps', True)
    num_tests = 100
    instance_num = 2
    for power_of_k_choice in [1, 2, 3]:
        dispatch_scheduler = DispatchScheduler('load', power_of_k_choice, instance_load_calculator, 2)
        instance_num_requests = {}
        instance_info_dict = {}
        for instance_id in [f'instance_{i}' for i in range(1, 4 + 1)]:
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
        for i in range(num_tests):
            instance_id_set.add(dispatch_scheduler.dispatch())
    assert len(instance_id_set) == 2
