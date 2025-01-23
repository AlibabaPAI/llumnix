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
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.arg_utils import InstanceArgs
from llumnix.constants import DISPATCH_LOG_FREQUENCY

logger = init_logger(__name__)


class DispatchScheduler:
    def __init__(self,
                 dispatch_policy: str,
                 power_of_k_choice: int) -> None:
        self.dispatch_policy = DispatchPolicyFactory.get_policy(dispatch_policy)
        self.power_of_k_choice = power_of_k_choice
        self.available_dispatch_instance_set: Set[str] = set()
        self.instance_info: Dict[str, InstanceInfo] = {}
        # statistics
        self.total_num_requests = 0
        self.instance_num_requests: Dict[str, int] = {}

    def dispatch(self) -> str:
        self.total_num_requests += 1
        dispatch_instance_id = self.dispatch_policy.dispatch(self.instance_num_requests,
                                                             self.instance_info.values(),
                                                             self.power_of_k_choice)
        self.instance_num_requests[dispatch_instance_id] += 1
        if self.total_num_requests % DISPATCH_LOG_FREQUENCY == 0:
            logger.info("dispatch scheduler total_dispatched_requests: {}".format(self.total_num_requests))
            for instance_id, num_requests in self.instance_num_requests.items():
                logger.info("instance {} num_dispatched_requests: {}".format(instance_id, num_requests))
        return dispatch_instance_id

    def update_instance_infos(self, instance_infos: Dict[str, InstanceInfo]) -> None:
        for instance_id, instance_info in instance_infos.items():
            if instance_id not in self.available_dispatch_instance_set:
                continue
            self.instance_info[instance_id] = instance_info

    def add_instance(self, instance_id: str, instance_args: InstanceArgs) -> None:
        if instance_args.instance_type in [InstanceType.NO_CONSTRAINTS, InstanceType.PREFILL]:
            self.available_dispatch_instance_set.add(instance_id)
            self.instance_num_requests[instance_id] = 0

    def remove_instance(self, instance_id: str) -> None:
        if instance_id in self.available_dispatch_instance_set:
            self.available_dispatch_instance_set.remove(instance_id)
            self.instance_num_requests.pop(instance_id, None)
