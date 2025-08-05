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

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo

logger = init_logger(__name__)


class PairMigrationConstraints(str, Enum):
    """Target of Migration."""
    NO_CONSTRAINTS = "NO_CONSTRAINTS"
    # Enable the prefill-decode disaggregration.
    DECODE_2_DECODE = "DECODE_2_DECODE"
    PREFILL_2_DECODE = "PREFILL_2_DECODE"
    PREFILL_2_PREFILL = "PREFILL_2_PREFILL"


class MigrationPolicy(ABC):
    def __init__(self, migrate_out_load_threshold: float = 0.0) -> None:
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


class Balanced(MigrationPolicy):
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
            if 0 < load_diff_after_mig < load_diff_before_mig:
                migrate_instance_pairs.append((sorted_src_instance_infos[i].instance_id,
                                               sorted_dst_instance_infos[i].instance_id))
        return migrate_instance_pairs


class Defrag(MigrationPolicy):
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


class AggrateDynamicPrefill(MigrationPolicy):
    def pair_migration(self,
                       src_instance_infos: List[InstanceInfo],
                       dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
        if min(len(src_instance_infos), len(dst_instance_infos)) <= 1:
            return []

        sorted_src_instance_infos = sorted(src_instance_infos, reverse=False,
                                           key=lambda instance_info: getattr(instance_info, 'instance_id'))
        sorted_dst_instance_infos = sorted(dst_instance_infos, reverse=True,
                                           key=lambda instance_info: getattr(instance_info, 'instance_id'))
        migrate_instance_pairs = []
        for i in range(min(len(sorted_src_instance_infos), len(sorted_dst_instance_infos))//2):
            if sorted_src_instance_infos[i].instance_id != sorted_dst_instance_infos[i].instance_id:
                migrate_instance_pairs.append((sorted_src_instance_infos[i].instance_id, sorted_dst_instance_infos[i].instance_id))

        return migrate_instance_pairs


class Failover(MigrationPolicy):
    def pair_migration(self,
                       src_instance_infos: List[InstanceInfo],
                       dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
        broken_unit_id = [src_instance_info.unit_id for src_instance_info in src_instance_infos if not src_instance_info.is_unit_healthy()]
        available_dst_instance_ids = [dst_instance_info.instance_id for dst_instance_info in dst_instance_infos
                                      if dst_instance_info.unit_id not in broken_unit_id]
        migrate_instance_pairs = []

        if len(dst_instance_infos) > 0:
            cur_dst_idx = 0
            for src_instance_info in src_instance_infos:
                dst_instance_id = available_dst_instance_ids[cur_dst_idx]
                migrate_instance_pairs.append((src_instance_info.instance_id, dst_instance_id))
                cur_dst_idx = (cur_dst_idx + 1) % len(available_dst_instance_ids)

        return migrate_instance_pairs


class MigrationPolicyFactory:
    _POLICY_REGISTRY = {
        'balanced': Balanced,
        'defrag': Defrag,
        'aggrate_dynamic_prefill': AggrateDynamicPrefill,
        'failover': Failover,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> MigrationPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
