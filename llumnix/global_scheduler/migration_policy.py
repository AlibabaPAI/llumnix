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

from typing import List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import copy
import numpy as np

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator

logger = init_logger(__name__)


class PairMigrationConstraints(str, Enum):
    """Target of Migration."""
    NO_CONSTRAINTS = "NO_CONSTRAINTS"
    # Enable the prefill-decoding disaggregration.
    DECODING_2_DECODING = "DECODING_2_DECODING"
    PREFILL_2_DECODING = "PREFILL_2_DECODING"


class PairMigrationPolicy(ABC):
    def __init__(self,
                 migrate_out_load_threshold: float,
                 instance_load_calculator: InstanceLoadCalculator) -> None:
        self.migrate_out_load_threshold = migrate_out_load_threshold
        self.instance_load_calculator = instance_load_calculator

    @abstractmethod
    def pair_migration(self,
                       src_instance_infos: List[InstanceInfo],
                       dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def sort_instance_infos(self, instance_infos: List[InstanceInfo], descending: bool = True) -> None:
        key_attr = 'instance_load_migrate'
        sorted_instance_infos = sorted(
            instance_infos,
            key=lambda instance_info: getattr(instance_info, key_attr),
            reverse=descending
        )
        return sorted_instance_infos


class Balanced(PairMigrationPolicy):
    def pair_migration(self,
                       src_instance_infos: List[InstanceInfo],
                       dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
        sorted_src_instance_infos = self.sort_instance_infos(src_instance_infos, descending=True)
        sorted_dst_instance_infos = self.sort_instance_infos(dst_instance_infos, descending=False)
        migrate_instance_pairs = []
        for i in range(min(len(sorted_src_instance_infos), len(sorted_dst_instance_infos))):
            load_diff_before_mig = sorted_src_instance_infos[i].instance_load_migrate - sorted_dst_instance_infos[i].instance_load_migrate

            left_load_after_mig = self._compute_instance_load_after_migrate(sorted_src_instance_infos[i], is_migrate_in=False)
            right_load_after_mig = self._compute_instance_load_after_migrate(sorted_dst_instance_infos[i], is_migrate_in=True)

            # Add some constrains to reduce unnecessary migrations
            if right_load_after_mig > self.migrate_out_load_threshold:
                continue
            load_diff_after_mig = left_load_after_mig - right_load_after_mig
            if (0 < load_diff_after_mig < load_diff_before_mig) or (sorted_dst_instance_infos[i].instance_load_migrate == -np.inf):
                migrate_instance_pairs.append((sorted_src_instance_infos[i].instance_id,
                                               sorted_dst_instance_infos[i].instance_id))
        return migrate_instance_pairs

    def _compute_instance_load_after_migrate(self, instance_info: InstanceInfo, is_migrate_in: bool) -> float:
        instance_info_after_migrate = copy.deepcopy(instance_info)
        num_blocks_last_running_request = instance_info_after_migrate.num_blocks_last_running_request

        if is_migrate_in:
            instance_info_after_migrate.num_running_requests += 1
            instance_info_after_migrate.num_free_gpu_blocks -= num_blocks_last_running_request
        else:
            instance_info_after_migrate.num_running_requests -= 1
            instance_info_after_migrate.num_free_gpu_blocks += num_blocks_last_running_request

        return self.instance_load_calculator.compute_instance_load(instance_info_after_migrate, action='migrate')


class DefragConstrained(PairMigrationPolicy):
    def pair_migration(self,
                       src_instance_infos: List[InstanceInfo],
                       dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
        sorted_src_instance_infos = self.sort_instance_infos(src_instance_infos, descending=True)
        sorted_dst_instance_infos = self.sort_instance_infos(dst_instance_infos, descending=False)
        migrate_instance_pairs = []
        for i in range(min(len(sorted_src_instance_infos), len(sorted_dst_instance_infos))):
            # without any constrain in order to make prefill migrate happens as soon as possible
            migrate_instance_pairs.append((sorted_src_instance_infos[i].instance_id, sorted_dst_instance_infos[i].instance_id))
        return migrate_instance_pairs


class PairMigrationPolicyFactory:
    _POLICY_REGISTRY = {
        'balanced': Balanced,
        'defrag_constrained': DefragConstrained,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> PairMigrationPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
