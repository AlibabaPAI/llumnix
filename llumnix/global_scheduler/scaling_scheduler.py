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

from typing import Dict, List, Tuple, Set
from abc import ABC, abstractmethod
import numpy as np

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, ScalingLoadComputation, InstanceType
from llumnix.arg_utils import InstanceArgs

logger = init_logger(__name__)


class ScalingScheduler:
    def __init__(self,
                 scale_up_threshold: float,
                 scale_down_threshold: float,
                 scaling_policy: str,
                 scaling_load_metric: str,
                 enable_pd_disagg: bool,) -> None:
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_policy = ScalePolicyFactory.get_policy(scaling_policy, scaling_load_metric=scaling_load_metric)

        self.num_instances = 0
        self.instance_id_set: Set[str] = set()
        self.enable_pd_disagg = enable_pd_disagg

        # instance info args
        self.instance_info: Dict[str, InstanceInfo] = None
        self.sorted_instance_infos: List[InstanceInfo] = None

        # TODO(Xinyi): Tag instance type for scheduler, should be extended to auto-scaling for prefill/decoding instances.
        self.instance_type_id_set: Dict[InstanceType, Set[str]] = {instance_type: set() for instance_type in InstanceType}

    def check_scale(self) -> Tuple[str, str]:
        scale_up_num = 0
        scale_down_num = 0
        # if not all instances have returned instance_info, not scale
        if len(self.instance_info.keys()) < self.num_instances:
            return scale_up_num, scale_down_num
        now_instances = [self.instance_info[instance_id] for instance_id in self.instance_id_set]
        load_metric_up = self.scaling_policy.compute_load_metric_up(now_instances)
        load_metric_down = self.scaling_policy.compute_load_metric_down(now_instances)
        if load_metric_up > self.scale_up_threshold:
            while self.scaling_policy.compute_load_metric_avg(now_instances) > self.scale_up_threshold:
                scale_up_num += 1
                now_instances.append(self.get_empty_instance_info())
        elif load_metric_down < self.scale_down_threshold:
            scale_down_num = 1
        return scale_up_num, scale_down_num

    def update_instance_infos(self, instance_info: Dict[str, InstanceInfo]) -> None:
        self.instance_info = instance_info

    def add_instance(self, instance_id: str, instance_args: InstanceArgs) -> None:
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)
        instance_type = InstanceType(instance_args.instance_type)
        self.instance_type_id_set[instance_type].add(instance_id)

    def remove_instance(self, instance_id: str) -> None:
        if instance_id in self.instance_id_set:
            self.instance_id_set.remove(instance_id)
        for instance_type in InstanceType:
            if instance_id in self.instance_type_id_set[instance_type]:
                self.instance_type_id_set[instance_type].remove(instance_id)
                break
        self.num_instances = len(self.instance_id_set)

    def get_empty_instance_info(self) -> InstanceInfo:
        dummy_intance_info = InstanceInfo()
        dummy_intance_info.instance_id = -1
        dummy_intance_info.step_id = -1
        # TODO(s5u13b): Should be changed for proactive auto-scaling.
        dummy_intance_info.num_total_gpu_blocks = np.inf
        dummy_intance_info.num_available_gpu_blocks = np.inf
        dummy_intance_info.num_free_gpu_blocks = np.inf
        dummy_intance_info.num_available_gpu_blocks_waiting = np.inf
        return dummy_intance_info


class ScalePolicy(ABC):
    def __init__(self, scaling_load_metric: str) -> None:
        self.scaling_load_calculator = ScalingLoadComputation(scaling_load_metric)

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
        return self.scaling_load_calculator.compute_instance_load(tot_instance_info)

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
        return self.scaling_load_calculator.compute_instance_load(tot_instance_info)


class ScalePolicyFactory:
    _POLICY_REGISTRY = {
        'max_load': MaxLoad,
        'min_load': MinLoad,
        'avg_load': AvgLoad,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> ScalePolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
