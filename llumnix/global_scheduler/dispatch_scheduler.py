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

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceLoadCalculator, InstanceInfo
from llumnix.global_scheduler.dispatch_policy import DispatchPolicyFactory

logger = init_logger(__name__)


class DispatchScheduler:
    def __init__(self,
                 dispatch_policy: str,
                 power_of_k_choice: int,
                 instance_load_calculator: InstanceLoadCalculator,
                 num_dispatch_instances: int) -> None:
        self.dispatch_policy = DispatchPolicyFactory.get_policy(dispatch_policy)
        self.power_of_k_choice = power_of_k_choice
        self.instance_load_calculator = instance_load_calculator
        self.num_instances = 0
        self.instance_id_set: Set[str] = set()
        self.available_dispatch_instance_set: Set[str] = set()
        self.num_dispatch_instances = num_dispatch_instances
        # instance info args
        self.instance_info: Dict[str, InstanceInfo] = {}
        self.available_instance_infos: List[InstanceInfo] = None
        # statistics
        self.num_requests = 0
        self.instance_num_requests: Dict[str, int] = {}

    def dispatch(self) -> str:
        self.num_requests += 1
        instance_infos: List[InstanceInfo] = list(self.instance_info.values())
        self.available_instance_infos = [info for info in instance_infos if info.instance_id in self.available_dispatch_instance_set]
        dispatch_instance_id = self.dispatch_policy.dispatch(self.instance_num_requests,
                                                             self.available_instance_infos,
                                                             self.power_of_k_choice)
        self.instance_num_requests[dispatch_instance_id] += 1
        if self.num_requests % 100 == 0:
            logger.info("num_requests: {}".format(self.num_requests))
            for instance_id, num_requests in self.instance_num_requests.items():
                logger.info("instance {} num_dispatched_requests: {}".format(instance_id, num_requests))
        return dispatch_instance_id

    def update_instance_infos(self,
                              instance_info: Dict[str, InstanceInfo]) -> None:
        self.instance_info = instance_info

    def add_instance(self, instance_id: str) -> None:
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)
        # TODO(KuilongCui): a hacky method is being used to avoid the only-decode type engine dispatched
        if "decode" not in instance_id:
            if self.num_dispatch_instances <= 0 or (self.num_dispatch_instances > 0 and
                len(self.available_dispatch_instance_set) < self.num_dispatch_instances):
                self.available_dispatch_instance_set.add(instance_id)
                self.instance_num_requests[instance_id] = 0

    def remove_instance(self, instance_id: str) -> None:
        self.instance_id_set.remove(instance_id)
        self.num_instances = len(self.instance_id_set)
        if instance_id in self.instance_num_requests:
            del self.instance_num_requests[instance_id]
        else:
            logger.warning("instance {} not in instance_num_requests".format(instance_id))
        if instance_id in self.available_dispatch_instance_set:
            self.available_dispatch_instance_set.remove(instance_id)
            # TODO(KuilongCui): Check it when there is no decode instance.
            if self.num_instances >= self.num_dispatch_instances:
                free_instance_id = next(iter(self.instance_id_set - self.available_dispatch_instance_set))
                self.available_dispatch_instance_set.add(free_instance_id)
