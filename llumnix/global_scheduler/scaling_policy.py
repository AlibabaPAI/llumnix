from typing import List
from abc import ABC, abstractmethod

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator

logger = init_logger(__name__)


class ScalePolicy(ABC):
    def __init__(self,
                 instance_load_calculator: InstanceLoadCalculator) -> None:
        self.instance_load_calculator = instance_load_calculator

    @abstractmethod
    def compute_load_metric_up(self, instance_infos: List[InstanceInfo]) -> float:
        pass

    @abstractmethod
    def compute_load_metric_down(self, instance_infos: List[InstanceInfo]) -> float:
        pass

    def compute_load_metric_avg(self, instance_infos: List[InstanceInfo]) -> float:
        tot_instance_info = InstanceInfo()
        tot_instance_info.instance_id = -1
        tot_instance_info.step_id = -1
        tot_instance_info.num_running_requests = sum([i.num_running_requests for i in instance_infos])
        tot_instance_info.num_waiting_requests = sum([i.num_waiting_requests for i in instance_infos])
        tot_instance_info.num_free_gpu_blocks = sum([i.num_free_gpu_blocks for i in instance_infos])
        tot_instance_info.num_total_gpu_blocks = sum([i.num_total_gpu_blocks for i in instance_infos])
        tot_instance_info.num_watermark_blocks = sum([i.num_watermark_blocks for i in instance_infos])
        tot_instance_info.num_blocks_all_waiting_requests = sum([i.num_blocks_all_waiting_requests for i in instance_infos])
        tot_instance_info.num_available_gpu_blocks = tot_instance_info.num_free_gpu_blocks - tot_instance_info.num_watermark_blocks
        return self.instance_load_calculator.compute_instance_load(tot_instance_info, action="scale")


class MaxLoad(ScalePolicy):
    def compute_load_metric_up(self, instance_infos: List[InstanceInfo]) -> float:
        return max([i.instance_load_dispatch_scale for i in instance_infos])

    def compute_load_metric_down(self, instance_infos: List[InstanceInfo]) -> float:
        return max([i.instance_load_dispatch_scale for i in instance_infos])


class MinLoad(ScalePolicy):
    def compute_load_metric_up(self, instance_infos: List[InstanceInfo]) -> float:
        return min([i.instance_load_dispatch_scale for i in instance_infos])

    def compute_load_metric_down(self, instance_infos: List[InstanceInfo]) -> float:
        return min([i.instance_load_dispatch_scale for i in instance_infos])


class AvgLoad(ScalePolicy):
    def compute_load_metric_up(self, instance_infos: List[InstanceInfo]) -> float:
        return self.compute_load_metric_avg(instance_infos)

    def compute_load_metric_down(self, instance_infos: List[InstanceInfo]) -> float:
        num_instances = len(instance_infos)
        tot_instance_info = InstanceInfo()
        tot_instance_info.instance_id = -1
        tot_instance_info.step_id = -1
        # the average load after scale down the last instance
        tot_instance_info.num_running_requests = sum([i.num_running_requests for i in instance_infos])
        tot_instance_info.num_waiting_requests = sum([i.num_waiting_requests for i in instance_infos])
        tot_instance_info.num_free_gpu_blocks = sum([i.num_free_gpu_blocks - i.num_total_gpu_blocks
                                                    if i.instance_id + 1 == num_instances else i.num_free_gpu_blocks
                                                    for i in instance_infos])
        tot_instance_info.num_free_gpu_blocks = max(0, tot_instance_info.num_free_gpu_blocks)
        tot_instance_info.num_total_gpu_blocks = sum([0 if i.instance_id + 1 == num_instances else i.num_total_gpu_blocks
                                                    for i in instance_infos])
        tot_instance_info.num_watermark_blocks = sum([0 if i.instance_id + 1 == num_instances else i.num_watermark_blocks
                                                    for i in instance_infos])
        tot_instance_info.num_blocks_all_waiting_requests = sum([i.num_blocks_all_waiting_requests for i in instance_infos])
        tot_instance_info.num_available_gpu_blocks = tot_instance_info.num_free_gpu_blocks - tot_instance_info.num_watermark_blocks
        return self.instance_load_calculator.compute_instance_load(tot_instance_info, action='scale')


class ScalePolicyFactory:
    _POLICY_REGISTRY = {
        'max_load': MaxLoad,
        'min_load': MinLoad,
        'avg_load': AvgLoad,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> ScalePolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
