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
                             power_of_k_choice: int):
    k = min(power_of_k_choice, len(sorted_instance_infos))
    top_k_instance_infos = sorted_instance_infos[:k]
    return random.choice(top_k_instance_infos)


class DispatchPolicy(ABC):
    @abstractmethod
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 power_of_k_choice: int) -> int:
        pass


# Dispatch all requests to a single instance, used only for testing
class Flood(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 power_of_k_choice: int) -> str:
        instance_id = max(instance_num_requests, key=instance_num_requests.get)
        return instance_id


class Balanced(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 power_of_k_choice: int) -> str:
        # dispatch request according to the number of requests dispatched to instance by manager
        instance_id = min(instance_num_requests, key=instance_num_requests.get)
        return instance_id


class Load(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 power_of_k_choice: int) -> str:
        sorted_instance_infos = sort_instance_infos(available_instance_infos, 'dispatch_load_metric')
        instance_info_chosen = random_choice_from_top_k(sorted_instance_infos, power_of_k_choice)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch to {}, load: {}".format(instance_id, instance_info_chosen.dispatch_load_metric))
        return instance_id


class Queue(DispatchPolicy):
    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 power_of_k_choice: int) -> str:
        sorted_instance_infos = sort_instance_infos(available_instance_infos, 'num_waiting_requests')
        instance_info_chosen = random_choice_from_top_k(sorted_instance_infos, power_of_k_choice)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch to {}, queue size: {}".format(instance_id, instance_info_chosen.num_waiting_requests))
        return instance_id


class RoundRobin(DispatchPolicy):
    prev_instance_idx: int = -1

    def dispatch(self,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: List[InstanceInfo],
                 power_of_k_choice: int) -> str:
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
