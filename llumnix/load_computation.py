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

import numpy as np

import llumnix.envs as llumnix_envs


class BaseLoad(ABC):
    @abstractmethod
    def is_busy(self) -> bool:
        """
        Returns True if the load is busy, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def __lt__(self, other: "BaseLoad") -> bool:
        """
        Returns True if the load is less than the other load, False otherwise.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def compute_instance_load(cls, instance_info: 'InstanceInfo') -> "BaseLoad":
        """
        Compute the load of an instance.
        """
        raise NotImplementedError


class DummyLoad(BaseLoad):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def is_busy(self) -> bool:
        return False

    def __lt__(self, other: "DummyLoad") -> bool:
        return True

    def __repr__(self):
        return "DummyLoad"

    @classmethod
    def compute_instance_load(cls, instance_info: 'InstanceInfo') -> "DummyLoad":
        return DummyLoad()


class KvBlocksRatioLoad(BaseLoad):
    BUSY_THRESHOLD = float(llumnix_envs.KVBLOCKSRATIO_BUSY_THRESHOLD)

    def __init__(self, demand_factor: float = 0.0) -> None:
        self.demand_factor = demand_factor

    def is_busy(self) -> bool:
        return self.demand_factor >= KvBlocksRatioLoad.BUSY_THRESHOLD

    def __lt__(self, other: "KvBlocksRatioLoad") -> bool:
        if isinstance(other, DummyLoad):
            return False
        return self.demand_factor < other.demand_factor

    def __repr__(self) -> str:
        return f"KvBlocksRatioLoad(demand_factor={self.demand_factor},is_busy={self.is_busy()})"

    @classmethod
    def compute_instance_load(cls, instance_info: 'InstanceInfo') -> "KvBlocksRatioLoad":
        all_wanted_blocks = instance_info.num_used_gpu_blocks + instance_info.num_blocks_all_waiting_requests
        if instance_info.num_total_gpu_blocks == 0:
            return KvBlocksRatioLoad(np.inf)
        demand_factor = all_wanted_blocks / instance_info.num_total_gpu_blocks
        return KvBlocksRatioLoad(demand_factor)


class RemainingStepsLoad(BaseLoad):
    BUSY_THRESHOLD = float(llumnix_envs.REMAININGSTEPS_BUSY_THRESHOLD)

    def __init__(self, remaining_steps: float = 0.0) -> None:
        self.remaining_steps = remaining_steps

    def is_busy(self) -> bool:
        return self.remaining_steps < RemainingStepsLoad.BUSY_THRESHOLD

    def __lt__(self, other: "RemainingStepsLoad") -> bool:
        if isinstance(other, DummyLoad):
            return False
        return self.remaining_steps >= other.remaining_steps

    def __repr__(self) -> str:
        return f"RemainingStepsLoad(remaining_steps={self.remaining_steps},is_busy={self.is_busy()})"

    @classmethod
    def compute_instance_load(cls, instance_info: 'InstanceInfo') -> "RemainingStepsLoad":
        if instance_info.enable_defrag:
            num_requests = instance_info.num_running_requests
            if instance_info.num_waiting_requests != 0:
                num_requests += 1
            num_available_gpu_blocks = instance_info.num_available_gpu_blocks - \
                instance_info.num_blocks_first_waiting_request
        else:
            num_requests = instance_info.num_running_requests + instance_info.num_waiting_requests
            num_available_gpu_blocks = instance_info.num_available_gpu_blocks - instance_info.num_blocks_all_waiting_requests

        if num_requests == 0:
            return RemainingStepsLoad(np.inf)
        return RemainingStepsLoad(num_available_gpu_blocks / num_requests)


class AdaptiveDecodeBatchLoad(BaseLoad):
    DECODE_COMPUTE_BOUND_BATCH_SIZE = float(llumnix_envs.DECODE_COMPUTE_BOUND_BATCH_SIZE)

    def __init__(self, decode_batch_size: float = 0) -> None:
        self.decode_batch_size = decode_batch_size

        if self.decode_batch_size == 0:
            self.decode_load = self.DECODE_COMPUTE_BOUND_BATCH_SIZE
        elif self.decode_batch_size >= self.DECODE_COMPUTE_BOUND_BATCH_SIZE:
            self.decode_load = self.decode_batch_size
        else:
            self.decode_load = self.DECODE_COMPUTE_BOUND_BATCH_SIZE - self.decode_batch_size

    def is_busy(self) -> bool:
        return self.decode_batch_size > self.DECODE_COMPUTE_BOUND_BATCH_SIZE

    def __lt__(self, other: "AdaptiveDecodeBatchLoad") -> bool:
        if isinstance(other, DummyLoad):
            return False
        return self.decode_load < other.decode_load

    def __repr__(self) -> str:
        return f"AdaptiveDecodeBatchLoad(decode_batch_size={self.decode_batch_size},is_busy={self.is_busy()})"

    @classmethod
    def compute_instance_load(cls, instance_info: 'InstanceInfo') -> "AdaptiveDecodeBatchLoad":
        return AdaptiveDecodeBatchLoad(instance_info.decode_batch_size)


class MissWaitingTokensLoad(BaseLoad):
    BUSY_THRESHOLD = float(llumnix_envs.MISSWAITINGTOKENS_BUSY_THRESHOLD)

    def __init__(self, miss_waiting_tokens: float = 0.0) -> None:
        self.miss_waiting_tokens = miss_waiting_tokens

    def is_busy(self) -> bool:
        return self.miss_waiting_tokens >= MissWaitingTokensLoad.BUSY_THRESHOLD

    def __lt__(self, other: "MissWaitingTokensLoad") -> bool:
        if isinstance(other, DummyLoad):
            return False
        return self.miss_waiting_tokens < other.miss_waiting_tokens

    def __repr__(self) -> str:
        return f"MissWaitingTokensLoad(miss_waiting_tokens={self.miss_waiting_tokens},is_busy={self.is_busy()})"

    @classmethod
    def compute_instance_load(cls, instance_info: 'InstanceInfo') -> "MissWaitingTokensLoad":
        return MissWaitingTokensLoad(instance_info.num_miss_tokens_all_waiting_requests)


class LoadCalculatorFactory:
    _LOAD_REGISTRY = {
        'kv_blocks_ratio': KvBlocksRatioLoad,
        'remaining_steps': RemainingStepsLoad,
        'adaptive_decode': AdaptiveDecodeBatchLoad,
        'miss_waiting_tokens': MissWaitingTokensLoad
    }

    @classmethod
    def get_load_calculator(cls, load_name: str, **kwargs) -> BaseLoad:
        return cls._LOAD_REGISTRY[load_name](**kwargs)
