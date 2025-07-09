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

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from llumnix.instance_info import InstanceInfo


class DispatchFilter(ABC):
    @abstractmethod
    def filter(self,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        raise NotImplementedError


class MetricBasedFilter(DispatchFilter):
    def __init__(self, metric: str):
        self.metric = metric

    def filter(self,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int]
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        available_instance_infos, available_instance_num_requests = {}, {}
        for ins_info, ins_num_requests in zip(instance_infos.values(), instance_num_requests.values()):
            if not getattr(ins_info, self.metric).is_busy():
                available_instance_infos[ins_info.instance_id] = ins_info
                available_instance_num_requests[ins_info.instance_id] = ins_num_requests
        return available_instance_infos, available_instance_num_requests
