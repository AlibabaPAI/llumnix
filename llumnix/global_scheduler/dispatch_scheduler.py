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

from typing import Dict

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.constants import DISPATCH_LOG_FREQUENCY
from llumnix.global_scheduler.dispatch_policy import DispatchPolicyFactory

logger = init_logger(__name__)


class DispatchScheduler:
    def __init__(self,
                 dispatch_policy: str,
                 topk_random_dispatch: int) -> None:
        self.dispatch_policy = DispatchPolicyFactory.get_policy(dispatch_policy)
        self.topk_random_dispatch = topk_random_dispatch

        # statistics
        self.instance_type_num_requests: Dict[str, int] = {}

    def dispatch(
        self,
        instance_info: Dict[str, InstanceInfo],
        instance_num_requests: Dict[str, int]
    ) -> str:
        instance_infos = list(instance_info.values())
        instance_type = instance_infos[0].instance_type
        num_requests = self.instance_type_num_requests.get(instance_type, 0) + 1
        self.instance_type_num_requests[instance_type] = num_requests
        dispatch_instance_id = self.dispatch_policy.dispatch(
            instance_num_requests, instance_infos, self.topk_random_dispatch
        )

        if num_requests % DISPATCH_LOG_FREQUENCY == 0:
            logger.info("dispatch scheduler total_dispatched_requests: {}".format(num_requests))
            for instance_id, num_requests in instance_num_requests.items():
                logger.info("instance {} num_dispatched_requests: {}".format(instance_id, num_requests))
        return dispatch_instance_id
