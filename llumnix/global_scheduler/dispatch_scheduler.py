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

from typing import Dict, List, Set
from abc import ABC, abstractmethod
import random

from llumnix.logger import init_logger
from llumnix.instance_info import InstanceLoadCalculator, InstanceInfo
from llumnix.arg_utils import InstanceArgs
from llumnix.global_scheduler.scaling_scheduler import InstanceType

logger = init_logger(__name__)


class DispatchScheduler:
    def __init__(self,
                 dispatch_policy: str,
                 instance_load_calculator: InstanceLoadCalculator,) -> None:
        self.dispatch_policy = DispatchPolicyFactory.get_policy(dispatch_policy)
        self.instance_load_calculator = instance_load_calculator
        self.available_dispatch_instance_set: Set[str] = set()
        # instance info args
        self.instance_info: Dict[str, InstanceInfo] = {}
        self.sorted_instance_infos: List[InstanceInfo] = None
        # statistics
        self.num_requests = 0
        self.instance_num_requests: Dict[str, int] = {}

    def dispatch(self) -> str:
        self.num_requests += 1
        dispatch_instance_id = self.dispatch_policy.dispatch(self.instance_num_requests,
                                                             self.instance_info.values())
        self.instance_num_requests[dispatch_instance_id] += 1
        if self.num_requests % 100 == 0:
            logger.info("dispatch scheduler total_dispatched_requests: {}".format(self.num_requests))
            for instance_id, num_requests in self.instance_num_requests.items():
                logger.info("Instance {} num_dispatched_requests: {}".format(instance_id, num_requests))
        return dispatch_instance_id

    def update_instance_infos(self, instance_info: Dict[str, InstanceInfo]) -> None:
        self.instance_info = instance_info

    def add_instance(self, instance_id: str, instance_args: InstanceArgs) -> None:
        if instance_args.instance_type in [InstanceType.NO_CONSTRAINTS, InstanceType.PREFILL]:
            self.available_dispatch_instance_set.add(instance_id)
            self.instance_num_requests[instance_id] = 0

    def remove_instance(self, instance_id: str) -> None:
        if instance_id in self.available_dispatch_instance_set:
            self.available_dispatch_instance_set.remove(instance_id)

class DispatchPolicy(ABC):
    @abstractmethod
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo]) -> str:
        pass

# Dispatch all requests to a single instance, used only for testing
class Flood(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo]) -> str:
        instance_id = max(instance_num_requests, key=instance_num_requests.get)
        return instance_id

class Balanced(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo]) -> str:
        # dispatch request according to the number of requests dispatched to instance by manager
        instance_id = min(instance_num_requests, key=instance_num_requests.get)
        return instance_id

class Load(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo]) -> str:
        sorted_instance_infos = sorted(
            available_instance_infos,
            key=lambda instance_info: getattr(instance_info, 'instance_load_dispatch_scale'),
        )
        instance_id = sorted_instance_infos[0].instance_id
        logger.debug("dispatch to {}, load: {}".format(instance_id, sorted_instance_infos[0].instance_load_dispatch_scale))
        return instance_id

class Queue(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo]) -> str:
        sorted_instance_infos = sorted(
            available_instance_infos,
            key=lambda instance_info: getattr(instance_info, 'num_waiting_requests'),
        )
        min_queue_size = sorted_instance_infos[0].num_waiting_requests
        instance_id_list = []
        for instance_info in sorted_instance_infos:
            if instance_info.num_waiting_requests == min_queue_size:
                instance_id_list.append(instance_info.instance_id)
        instance_id = random.choice(instance_id_list)
        logger.debug("dispatch to {}, queue size: {}".format(instance_id, sorted_instance_infos[0].num_waiting_requests))
        return instance_id

class RoundRobin(DispatchPolicy):
    prev_instance_idx: int = -1

    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo]) -> str:
        all_instance_ids = sorted(instance_num_requests.keys())
        cur_instance_idx = (self.prev_instance_idx + 1) % len(all_instance_ids)

        target_instance_id = all_instance_ids[cur_instance_idx]
        self.prev_instance_idx = cur_instance_idx
        return target_instance_id

class DispatchPolicyFactory:
    _POLICY_REGISTRY = {
        'flood': Flood,
        'balanced': Balanced,
        'load': Load,
        'queue': Queue,
        'rr': RoundRobin,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> DispatchPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
