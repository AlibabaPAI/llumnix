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
from llumnix.global_scheduler.scaling_scheduler import InstanceType

logger = init_logger(__name__)

class PairMigrationConstraints(str, Enum):
    """Target of Migration."""
    NO_CONSTRAINTS = "NO_CONSTRAINTS"

    # Enable the prefill-decoding disaggregration.
    DECODING_2_DECODING = "DECODING_2_DECODING"
    PREFILL_2_DECODING = "PREFILL_2_DECODING"

class MigrationScheduler:
    def __init__(self,
                 pair_migration_policy: str,
                 migrate_out_load_threshold: float,
                 instance_load_calculator: InstanceLoadCalculator) -> None:
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
        # instance info args
        self.instance_info: Dict[str, InstanceInfo] = None
        self.sorted_instance_infos: List[InstanceInfo] = None

    def pair_migration(self, pair_migration_type: PairMigrationConstraints) -> List[Tuple[str, str]]:
        self._sort_instance_infos(descending=False)
        sorted_src_instance_infos, sorted_dst_instance_infos = self._get_migration_instance_infos(pair_migration_type)
        return self.pair_migration_policy.pair_migration(sorted_src_instance_infos, sorted_dst_instance_infos)

    def _get_migration_instance_infos(self, pair_migration_type: PairMigrationConstraints) -> Dict[str, InstanceInfo]:
        filter_instance_infos_policy = FilteringInstanceInfosPolicyFactory.get_policy(pair_migration_type,
                                                        migrate_out_load_threshold=self.migrate_out_load_threshold)
        return filter_instance_infos_policy.filter_instances(self.sorted_instance_infos,pair_migration_type)

    def update_instance_infos(self,
                              instance_info: Dict[str, InstanceInfo]) -> None:
        self.instance_info = instance_info

    def add_instance(self, instance_id: str) -> None:
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)

    def remove_instance(self, instance_id: str) -> None:
        self.instance_id_set.remove(instance_id)
        self.num_instances = len(self.instance_id_set)

    def _sort_instance_infos(self,
                             descending: bool = True) -> None:
        instance_infos: List[InstanceInfo] = list(self.instance_info.values())
        key_attr = 'instance_load_migrate'
        self.sorted_instance_infos = sorted(
            instance_infos,
            key=lambda instance_info: getattr(instance_info, key_attr),
            reverse=descending
        )

class FilteringInstanceInfosPolicy(ABC):
    def __init__(self,
                 migrate_out_load_threshold: float) -> None:
        self.migrate_out_load_threshold = migrate_out_load_threshold
        self.filter_instances_rules = {
            PairMigrationConstraints.NO_CONSTRAINTS: (InstanceType.NO_CONSTRAINTS, InstanceType.NO_CONSTRAINTS),
            PairMigrationConstraints.DECODING_2_DECODING: (InstanceType.DECODE, InstanceType.DECODE),
            PairMigrationConstraints.PREFILL_2_DECODING: (InstanceType.PREFILL, InstanceType.DECODE),
        }

    def filter_instances(self, sorted_instance_infos: List[InstanceInfo],
                         pair_migration_type: PairMigrationConstraints = None) -> Dict[str, InstanceInfo]:
        src_type, dst_type = self.filter_instances_rules[pair_migration_type]
        filtered_src_instance_infos = [info for info in sorted_instance_infos if info.instance_type == src_type]
        filtered_dst_instance_infos = [info for info in sorted_instance_infos if info.instance_type == dst_type]
        src_instance_infos = self.filter_src_instances(filtered_src_instance_infos)
        dst_instance_infos = self.filter_dst_instances(filtered_dst_instance_infos)
        return src_instance_infos, dst_instance_infos

    @abstractmethod
    def filter_src_instances(self, filtered_instance_infos) -> Dict[str, InstanceInfo]:
        raise NotImplementedError

    @abstractmethod
    def filter_dst_instances(self, filtered_instance_infos) -> Dict[str, InstanceInfo]:
        raise NotImplementedError

class FilterConstrained(FilteringInstanceInfosPolicy):
    def filter_src_instances(self, filtered_instance_infos: List[InstanceInfo]) -> Dict[str, InstanceInfo]:
        src_instance_infos = [i for i in reversed(filtered_instance_infos)
                            if i.num_killed_requests > 0 or i.instance_load_migrate > self.migrate_out_load_threshold]
        return src_instance_infos

    def filter_dst_instances(self, filtered_instance_infos: List[InstanceInfo]) -> Dict[str, InstanceInfo]:
        dst_instance_infos = [i for i in filtered_instance_infos
                            if i.num_killed_requests == 0 and i.instance_load_migrate < self.migrate_out_load_threshold]
        return dst_instance_infos

class FilterRelaxed(FilteringInstanceInfosPolicy):
    # The policy is currently used to select the decoding instances to migrate requests from the prefill instances.
    def filter_src_instances(self, filtered_instance_infos: List[InstanceInfo]) -> Dict[str, InstanceInfo]:
        src_instance_infos = list(reversed(filtered_instance_infos))
        return src_instance_infos

    def filter_dst_instances(self, filtered_instance_infos: List[InstanceInfo]) -> Dict[str, InstanceInfo]:
        dst_instance_infos = [i for i in filtered_instance_infos
                                        if i.num_killed_requests == 0]
        return dst_instance_infos

class FilteringInstanceInfosPolicyFactory:
    _POLICY_REGISTRY = {
        PairMigrationConstraints.NO_CONSTRAINTS: FilterConstrained,
        PairMigrationConstraints.DECODING_2_DECODING: FilterConstrained,
        PairMigrationConstraints.PREFILL_2_DECODING: FilterRelaxed,
    }

    @classmethod
    def get_policy(cls, policy_name: PairMigrationConstraints, **kwargs) -> FilteringInstanceInfosPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)

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
