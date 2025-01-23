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
from enum import Enum
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
