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

from typing import Dict, List
from abc import ABC, abstractmethod
import random

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceType, sort_instance_infos


logger = init_logger(__name__)


class DispatchPolicy(ABC):
    def __init__(self, topk_random_dispatch: int = 1):
        self.topk_random_dispatch: int = topk_random_dispatch

    @abstractmethod
    def dispatch(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo]) -> str:
        raise NotImplementedError

    def random_choice_from_top_k(self, sorted_instance_infos: List[InstanceInfo]):
        k = min(self.topk_random_dispatch, len(sorted_instance_infos))
        top_k_instance_infos = sorted_instance_infos[:k]
        return random.choice(top_k_instance_infos)


# Dispatch all requests to a single instance, used only for testing
class Flood(DispatchPolicy):
    def __init__(self, topk_random_dispatch: int):
        super().__init__(topk_random_dispatch)

    def dispatch(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo]) -> str:
        instance_id = max(instance_num_requests, key=instance_num_requests.get)
        return instance_id


class Balanced(DispatchPolicy):
    def __init__(self, topk_random_dispatch: int):
        super().__init__(topk_random_dispatch)

    def dispatch(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo]) -> str:
        instance_id = min(instance_num_requests, key=instance_num_requests.get)
        return instance_id


class Load(DispatchPolicy):
    instance_type_metric_map: Dict[InstanceType, str] = {
        InstanceType.NO_CONSTRAINTS: 'dispatch_load_metric',
        InstanceType.PREFILL: 'dispatch_load_metric',
        InstanceType.DECODE: 'dispatch_load_metric',
        InstanceType.PREFILL_AS_DECODE: 'dispatch_prefill_as_decode_load_metric',
        InstanceType.DECODE_AS_PREFILL: 'dispatch_decode_as_prefill_load_metric'
    }
    
    def __init__(self, topk_random_dispatch: int):
        super().__init__(topk_random_dispatch)

    def dispatch(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo]) -> str:
        sorted_instance_infos = sort_instance_infos(available_instance_infos.values(),
                                                    self.instance_type_metric_map[instance_type])
        instance_info_chosen = self.random_choice_from_top_k(sorted_instance_infos)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch {} request to {}, load: {}".format(instance_type, instance_id,
                                                                 instance_info_chosen.dispatch_load_metric))
        return instance_id


class Queue(DispatchPolicy):
    def __init__(self, topk_random_dispatch: int):
        super().__init__(topk_random_dispatch)

    def dispatch(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo]) -> str:
        sorted_instance_infos = sort_instance_infos(available_instance_infos.values(), 'num_waiting_requests')
        instance_info_chosen = self.random_choice_from_top_k(sorted_instance_infos)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch to {}, queue size: {}".format(instance_id, instance_info_chosen.num_waiting_requests))
        return instance_id


class RoundRobin(DispatchPolicy):
    def __init__(self, topk_random_dispatch: int) -> None:
        self.prev_instance_type_idx: Dict[str, int] = {}
        super().__init__(topk_random_dispatch)

    def dispatch(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo]) -> str:
        prev_idx = self.prev_instance_type_idx.get(instance_type, -1)
        all_instance_ids = sorted(instance_num_requests.keys())
        cur_idx = (prev_idx + 1) % len(all_instance_ids)
        target_instance_id = all_instance_ids[cur_idx]
        self.prev_instance_type_idx[instance_type] = cur_idx
        return target_instance_id


class DispatchPolicyFactory:
    _POLICY_REGISTRY = {
        'flood': Flood,
        'balanced': Balanced,
        'load': Load,
        'queue': Queue,
        'rr': RoundRobin
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> DispatchPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
