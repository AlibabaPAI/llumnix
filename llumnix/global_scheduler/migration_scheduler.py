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
from enum import Enum
import copy
import numpy as np

from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator

logger = init_logger(__name__)

class PairMigrationConstraints(str, Enum):
    """Target of Migration."""
    NO_CONSTRAINTS = "NO_CONSTRAINTS"

    # Enable the prefill-decoding disaggregration.
    DECODING_2_DECODING = "DECODING_2_DECODING"
    PREFILL_2_DECODING = "PREFILL_2_DECODING"

class InstanceType(str, Enum):
    NO_CONSTRAINTS = "NO_CONSTRAINTS"

    # Specific to Prefill-Decoding disaggregation.
    PREFILL = "prefill"
    DECODE = "decode"

class MigrationScheduler:
    def __init__(self,
                 pair_migration_policy: str,
                 migrate_out_load_threshold: float,
                 instance_load_calculator: InstanceLoadCalculator,
                 constraint_prefill_instance_num: int) -> None:
        self.migrate_out_load_threshold = migrate_out_load_threshold
        self.instance_load_calculator = instance_load_calculator
        self.enable_defrag = instance_load_calculator.enable_defrag
        if not self.enable_defrag:
            self.pair_migration_policy \
                = PairMigrationPolicyFactory.get_policy("balanced",
                                                        migrate_out_load_threshold=migrate_out_load_threshold,
                                                        instance_load_calculator=instance_load_calculator)
        else:
            self.pair_migration_policy \
                = PairMigrationPolicyFactory.get_policy(pair_migration_policy,
                                                        migrate_out_load_threshold=migrate_out_load_threshold,
                                                        instance_load_calculator=instance_load_calculator)

        self.num_instances = 0
        self.instance_id_set: Set[str] = set()
        self.instance_id_type_set: Dict[InstanceType, Set[str]] = {instance_type: set() for instance_type in InstanceType}
        self.constraint_prefill_instance_num = constraint_prefill_instance_num
        # instance info args
        self.instance_info: Dict[str, InstanceInfo] = None
        self.sorted_instance_infos: Dict[str, List[InstanceInfo]] = {instance_type: list() for instance_type in InstanceType}

    def pair_migration(self, pair_migration_type:str) -> List[Tuple[str, str]]:
        sorted_src_instance_infos, sorted_dst_instance_infos = self._get_migration_settings(pair_migration_type)
        return self.pair_migration_policy.pair_migration(sorted_src_instance_infos, sorted_dst_instance_infos)

    def _get_migration_settings(self, pair_migration_type:str) -> Dict[str, InstanceInfo]:
        if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
            # migrate out instances
            self._sort_instance_infos([InstanceType.NO_CONSTRAINTS])
            sorted_src_instance_infos = [i for i in reversed(self.sorted_instance_infos[InstanceType.NO_CONSTRAINTS])
                                if i.num_killed_requests > 0 or i.instance_load_migrate > self.migrate_out_load_threshold]
            # migrate in instances
            sorted_dst_instance_infos = [i for i in self.sorted_instance_infos[InstanceType.NO_CONSTRAINTS]
                               if i.num_killed_requests == 0 and i.instance_load_migrate < self.migrate_out_load_threshold]
        elif pair_migration_type == PairMigrationConstraints.PREFILL_2_DECODING:
            self._sort_instance_infos([InstanceType.PREFILL, InstanceType.DECODE])
            sorted_src_instance_infos = list(reversed(self.sorted_instance_infos[InstanceType.PREFILL]))
            sorted_dst_instance_infos = [i for i in self.sorted_instance_infos[InstanceType.DECODE]
                                            if i.num_killed_requests == 0]
            # TODO[xinyi]: Considering decoding instances load, try to decode on the prefill instance.
        elif pair_migration_type == PairMigrationConstraints.DECODING_2_DECODING:
            self._sort_instance_infos([InstanceType.DECODE])
            sorted_src_instance_infos = [i for i in reversed(self.sorted_instance_infos[InstanceType.DECODE])
                                if i.num_killed_requests > 0 or i.instance_load_migrate > self.migrate_out_load_threshold]
            sorted_dst_instance_infos = [i for i in self.sorted_instance_infos[InstanceType.DECODE]
                               if i.num_killed_requests == 0 and i.instance_load_migrate < self.migrate_out_load_threshold]
        return sorted_src_instance_infos, sorted_dst_instance_infos

    def update_instance_infos(self,
                              instance_info: Dict[str, InstanceInfo]) -> None:
        self.instance_info = instance_info

    def add_instance(self, instance_id: str) -> None:
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)
        if self.constraint_prefill_instance_num > 0:
            if len(self.instance_id_type_set[InstanceType.PREFILL]) < self.constraint_prefill_instance_num:
                self.instance_id_type_set[InstanceType.PREFILL].add(instance_id)
            else:
                self.instance_id_type_set[InstanceType.DECODE].add(instance_id)
        else:
            self.instance_id_type_set[InstanceType.NO_CONSTRAINTS].add(instance_id)

    def remove_instance(self, instance_id: str) -> None:
        self.instance_id_set.remove(instance_id)
        self.num_instances = len(self.instance_id_set)

    def _sort_instance_infos(self, instance_types_list: str,
                             descending: bool = False) -> None:
        instance_infos: List[InstanceInfo] = list(self.instance_info.values())
        filtered_instance_infos  = {inst_type: set() for inst_type in instance_types_list}
        key_attr = 'instance_load_migrate'
        for inst_type in instance_types_list:
            filtered_instance_infos[inst_type] = [info for info in instance_infos if info.instance_id in self.instance_id_type_set[inst_type]]
            self.sorted_instance_infos[inst_type] = sorted(
                filtered_instance_infos[inst_type],
                key=lambda instance_info: getattr(instance_info, key_attr),
                reverse=descending
            )

class PairMigrationPolicy(ABC):
    def __init__(self,
                 migrate_out_load_threshold: float,
                 instance_load_calculator: InstanceLoadCalculator) -> None:
        self.migrate_out_load_threshold = migrate_out_load_threshold
        self.instance_load_calculator = instance_load_calculator

    @abstractmethod
    def pair_migration(self,
                       sorted_src_instance_infos: List[InstanceInfo],
                       sorted_dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
        raise NotImplementedError

class Balanced(PairMigrationPolicy):
    def pair_migration(self,
                       sorted_src_instance_infos: List[InstanceInfo],
                       sorted_dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
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
                       sorted_src_instance_infos: List[InstanceInfo],
                       sorted_dst_instance_infos: List[InstanceInfo],
                       ) -> List[Tuple[str, str]]:
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
