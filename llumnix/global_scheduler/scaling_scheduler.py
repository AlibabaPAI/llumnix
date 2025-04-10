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

from typing import Dict, Tuple, Set
import numpy as np

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.global_scheduler.scaling_policy import ScalePolicyFactory

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

        self.enable_pd_disagg = enable_pd_disagg

    def check_scale(self, instance_info: Dict[str, InstanceInfo], instance_id_set: Set[str]) -> Tuple[str, str]:
        scale_up_num = 0
        scale_down_num = 0
        # if not all instances have returned instance_info, not scale
        if len(instance_info.keys()) < len(instance_id_set):
            return scale_up_num, scale_down_num
        now_instances = [instance_info[instance_id] for instance_id in instance_id_set]
        load_metric_up = self.scaling_policy.compute_load_metric_up(now_instances)
        load_metric_down = self.scaling_policy.compute_load_metric_down(now_instances)
        if load_metric_up > self.scale_up_threshold:
            while self.scaling_policy.compute_load_metric_avg(now_instances) > self.scale_up_threshold:
                scale_up_num += 1
                now_instances.append(self.get_empty_instance_info())
        elif load_metric_down < self.scale_down_threshold:
            scale_down_num = 1
        return scale_up_num, scale_down_num


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
