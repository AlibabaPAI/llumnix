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
import numpy as np

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo

logger = init_logger(__name__)


class PairMigrationConstraints(str, Enum):
    """Target of Migration."""
    NO_CONSTRAINTS = "NO_CONSTRAINTS"
    # Enable the prefill-decoding disaggregration.
    DECODING_2_DECODING = "DECODING_2_DECODING"
    PREFILL_2_DECODING = "PREFILL_2_DECODING"


class PairMigrationPolicy(ABC):
    def __init__(self, migrate_out_load_threshold: float) -> None:
        self.migrate_out_load_threshold = migrate_out_load_threshold

    @abstractmethod
    def pair_migration(self,
                       src_instance_infos: List[InstanceInfo],
                       dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def sort_instance_infos(self, instance_infos: List[InstanceInfo], descending: bool = True) -> None:
        key_attr = 'migration_load_metric'
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
            load_diff_before_mig = sorted_src_instance_infos[i].migration_load_metric - sorted_dst_instance_infos[i].migration_load_metric
            left_load_after_mig = sorted_src_instance_infos[i].migration_load_metric_after_migrate_out
            right_load_after_mig = sorted_dst_instance_infos[i].migration_load_metric_after_migrate_in
            # Add some constrains to reduce unnecessary migrations
            if right_load_after_mig > self.migrate_out_load_threshold:
                continue
            load_diff_after_mig = left_load_after_mig - right_load_after_mig
            if (0 < load_diff_after_mig < load_diff_before_mig) or (sorted_dst_instance_infos[i].migration_load_metric == -np.inf):
                migrate_instance_pairs.append((sorted_src_instance_infos[i].instance_id,
                                               sorted_dst_instance_infos[i].instance_id))
        return migrate_instance_pairs


class Defrag(PairMigrationPolicy):
    def pair_migration(self,
                       src_instance_infos: List[InstanceInfo],
                       dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
        sorted_src_instance_infos = self.sort_instance_infos(src_instance_infos, descending=True)
        sorted_dst_instance_infos = self.sort_instance_infos(dst_instance_infos, descending=False)
        migrate_instance_pairs = []
        for i in range(min(len(sorted_src_instance_infos), len(sorted_dst_instance_infos))):
            migrate_instance_pairs.append((sorted_src_instance_infos[i].instance_id, sorted_dst_instance_infos[i].instance_id))
        return migrate_instance_pairs


class PairMigrationPolicyFactory:
    _POLICY_REGISTRY = {
        'balanced': Balanced,
        'defrag': Defrag,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> PairMigrationPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
