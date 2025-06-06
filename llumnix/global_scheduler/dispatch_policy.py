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
from llumnix.instance_info import InstanceInfo


logger = init_logger(__name__)


def sort_instance_infos(available_instance_infos: List[InstanceInfo],
                        key_attr: str,
                        descending: bool = False) -> None:
    return sorted(
        available_instance_infos,
        key=lambda instance_info: getattr(instance_info, key_attr),
        reverse=descending
    )

def random_choice_from_top_k(sorted_instance_infos: List[InstanceInfo],
                             topk_random_dispatch: int):
    k = min(topk_random_dispatch, len(sorted_instance_infos))
    top_k_instance_infos = sorted_instance_infos[:k]
    return random.choice(top_k_instance_infos)


class DispatchPolicy(ABC):
    @abstractmethod
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 topk_random_dispatch: int) -> int:
        pass


# Dispatch all requests to a single instance, used only for testing
class Flood(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 topk_random_dispatch: int) -> str:
        instance_id = max(instance_num_requests, key=instance_num_requests.get)
        return instance_id


class Balanced(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 topk_random_dispatch: int) -> str:
        # dispatch request according to the number of requests dispatched to instance by manager
        instance_id = min(instance_num_requests, key=instance_num_requests.get)
        return instance_id


class Load(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 topk_random_dispatch: int) -> str:
        sorted_instance_infos = sort_instance_infos(available_instance_infos, 'dispatch_load_metric')
        instance_info_chosen = random_choice_from_top_k(sorted_instance_infos, topk_random_dispatch)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch to {}, load: {}".format(instance_id, instance_info_chosen.dispatch_load_metric))
        return instance_id


class Queue(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 topk_random_dispatch: int) -> str:
        sorted_instance_infos = sort_instance_infos(available_instance_infos, 'num_waiting_requests')
        instance_info_chosen = random_choice_from_top_k(sorted_instance_infos, topk_random_dispatch)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch to {}, queue size: {}".format(instance_id, instance_info_chosen.num_waiting_requests))
        return instance_id


class RoundRobin(DispatchPolicy):
    def __init__(self) -> None:
        self.prev_instance_type_idx: Dict[str, int] = {}

    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 topk_random_dispatch: int) -> str:
        instance_type: str = available_instance_infos[0].instance_type
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
        'rr': RoundRobin,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> DispatchPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
