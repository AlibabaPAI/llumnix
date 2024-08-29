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
from typing import Dict
import numpy as np

from llumnix.logger import init_logger

logger = init_logger(__name__)


class InstanceInfo:
    def __init__(self,
                 num_total_gpu_blocks: int = 0,
                 num_watermark_blocks: int= 0,
                 num_used_gpu_blocks: int = 0,
                 num_free_gpu_blocks: int = 0,
                 gpu_cache_usage: float = 0.0,
                 num_running_requests: int = 0,
                 num_waiting_requests: int = 0,
                 num_killed_requests: int = 0,
                 num_blocks_first_waiting_request: int = 0,
                 waiting_time_first_waiting_request: int = 0,
                 num_blocks_all_waiting_requests: int = 0,
                 inference_type: str = "",
                 num_batched_tokens: int = 0) -> None:
        self.num_total_gpu_blocks = num_total_gpu_blocks
        self.num_watermark_blocks = num_watermark_blocks
        self.num_used_gpu_blocks = num_used_gpu_blocks
        self.num_free_gpu_blocks = num_free_gpu_blocks
        self.num_available_gpu_blocks = self.num_free_gpu_blocks - self.num_watermark_blocks
        self.gpu_cache_usage = gpu_cache_usage
        self.num_running_requests = num_running_requests
        self.num_waiting_requests = num_waiting_requests
        self.num_killed_requests = num_killed_requests
        self.num_blocks_first_waiting_request = num_blocks_first_waiting_request
        self.waiting_time_first_waiting_request = waiting_time_first_waiting_request
        self.num_blocks_all_waiting_requests = num_blocks_all_waiting_requests
        self.num_available_gpu_blocks_waiting = self.num_available_gpu_blocks - self.num_blocks_all_waiting_requests
        # For instance load computation before migration.
        self.num_blocks_last_running_request = 0

        # For global scheduling.
        self.instance_load_migrate = -np.inf
        self.instance_load_dispatch_scale = -np.inf

        # For record statistics, assigned in scheduler.
        self.inference_type = inference_type
        self.num_batched_tokens = num_batched_tokens
        self.running_seq_lens = []
        self.num_seqs = 0
        self.max_tot_tokens = 0
        self.finished_request_ids = None

        # For record statistics, assigned in backend engine.
        self.instance_id = None
        self.step_id = None
        self.timestamp = None
        self.latency = 0.0

class InstanceLoadInfo:
    def __init__(self, instance_info: InstanceInfo) -> None:
        self.num_total_gpu_blocks = instance_info.num_total_gpu_blocks
        self.num_watermark_blocks = instance_info.num_watermark_blocks
        self.num_used_gpu_blocks = instance_info.num_used_gpu_blocks
        self.num_free_gpu_blocks = instance_info.num_free_gpu_blocks
        self.num_available_gpu_blocks = instance_info.num_available_gpu_blocks

        self.num_waiting_requests = instance_info.num_waiting_requests
        self.num_running_requests = instance_info.num_running_requests
        self.num_killed_requests = instance_info.num_killed_requests

        self.num_blocks_first_waiting_request = instance_info.num_blocks_first_waiting_request
        self.waiting_time_first_waiting_request = instance_info.waiting_time_first_waiting_request
        self.num_blocks_all_waiting_requests = instance_info.num_blocks_all_waiting_requests

        self.instance_id = instance_info.instance_id
        self.step_id = instance_info.step_id

class InstanceLoadCalculator:
    def __init__(self,
                 load_metric: str,
                 enable_defrag: bool) -> None:
        self.load_metric = load_metric
        self.enable_defrag = enable_defrag
        self.load_computation_strategies: Dict[str, LoadComputationStrategy] = {
            'migrate': MigrationLoadComputation(load_metric, enable_defrag),
            'dispatch': DispatchAndScalingLoadComputation(load_metric, enable_defrag),
            'scale': DispatchAndScalingLoadComputation(load_metric, enable_defrag),
        }

    def compute_instance_load(self,
                              instance_info: InstanceInfo,
                              action: str = 'migrate') -> float:
        instance_load_info = InstanceLoadInfo(instance_info)
        assert action in self.load_computation_strategies
        load_computation_strategy = self.load_computation_strategies[action]
        return load_computation_strategy.compute_instance_load(instance_load_info)

class LoadComputationStrategy(ABC):
    def __init__(self,
                 load_metric: str,
                 enable_defrag: bool) -> None:
        self.load_metric = load_metric
        self.enable_defrag = enable_defrag

    @abstractmethod
    def compute_instance_load(self, i: InstanceLoadInfo) -> float:
        pass

class MigrationLoadComputation(LoadComputationStrategy):
    def compute_instance_load(self, i: InstanceLoadInfo) -> float:
        instance_load = -np.inf
        if self.load_metric == 'usage_ratio':
            instance_load = (i.num_used_gpu_blocks + i.num_blocks_first_waiting_request) / i.num_total_gpu_blocks
        elif self.load_metric == 'remaining_steps':
            if not self.enable_defrag:
                num_requests = i.num_running_requests
                num_available_gpu_blocks = i.num_available_gpu_blocks
            else:
                num_requests = i.num_running_requests
                if i.num_waiting_requests != 0:
                    num_requests += 1
                    # num_requests = i.num_running_requests + i.num_waiting_requests
                num_available_gpu_blocks = i.num_available_gpu_blocks - i.num_blocks_first_waiting_request
                # num_available_gpu_blocks = i.num_available_gpu_blocks - i.num_blocks_all_waiting_requests
            if num_requests == 0:
                return -np.inf
            instance_load = (num_available_gpu_blocks / num_requests)*(-1)
        return instance_load

class DispatchAndScalingLoadComputation(LoadComputationStrategy):
    def compute_instance_load(self, i: InstanceLoadInfo) -> float:
        instance_load = -np.inf
        if self.load_metric == 'usage_ratio':
            instance_load = (i.num_used_gpu_blocks + i.num_blocks_all_waiting_requests) / i.num_total_gpu_blocks
        elif self.load_metric == 'remaining_steps':
            num_requests = i.num_running_requests + i.num_waiting_requests
            num_available_gpu_blocks = i.num_available_gpu_blocks - i.num_blocks_all_waiting_requests
            if num_requests == 0:
                return -np.inf
            instance_load = (num_available_gpu_blocks / num_requests)*(-1)
        return instance_load
