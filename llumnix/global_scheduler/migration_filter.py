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

from typing import Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo

logger = init_logger(__name__)


class MigrationFilterConfig:
    def __init__(self, migrate_out_load_threshold):
        self.migrate_out_load_threshold: float = migrate_out_load_threshold


class MigrationFilter(ABC):
    @abstractmethod
    def filter_src_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        raise NotImplementedError

    @abstractmethod
    def filter_dst_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        raise NotImplementedError


class MigrationFilterPipeline:
    def __init__(self, filter_config: MigrationFilterConfig) -> None:
        self.filter_config = filter_config
        self.registered_filters: Dict[str, MigrationFilter] = {}

    def __repr__(self):
        return "MigrationFilterPipeline(filters={})".format(self.registered_filters.keys())

    def add_filter(self, filter_name: str, migration_filter: MigrationFilter) -> bool:
        if filter_name in self.registered_filters:
            logger.warning("Migration filter {} has been registered.".format(filter_name))
            return False

        self.registered_filters[filter_name] = migration_filter
        return True

    def remove_filter(self, filter_name: str) -> None:
        self.registered_filters.pop(filter_name, None)

    def get_filter(self, filter_name: str) -> Optional[MigrationFilter]:
        return self.registered_filters.get(filter_name, None)

    def filter_instances(self, instance_infos: List[InstanceInfo]) -> Tuple[List[InstanceInfo], List[InstanceInfo]]:
        src_filter_conditions = [filter.filter_src_condition(self.filter_config)
                                 for filter in self.registered_filters.values()]
        dst_filter_conditions = [filter.filter_dst_condition(self.filter_config)
                                 for filter in self.registered_filters.values()]
<<<<<<< HEAD
=======

        if pair_migration_type == PairMigrationConstraints.NEUTRAL:
            policy_filter = MigrationFilterPolicyFactory.get_policy("load")
        elif pair_migration_type in [PairMigrationConstraints.PREFILL_2_DECODE, PairMigrationConstraints.DECODE_2_DECODE]:
            policy_filter = MigrationFilterPolicyFactory.get_policy('pdd')
        else:
            raise ValueError(f"Unsupported pair migration type: {pair_migration_type}")
        src_filter_conditions.append(policy_filter.filter_src_condition(self.filter_config, pair_migration_type))
        dst_filter_conditions.append(policy_filter.filter_dst_condition(self.filter_config, pair_migration_type))

>>>>>>> 624c758 (Address review comments)
        filtered_src_instance_infos = [info for info in instance_infos if all(cond(info) for cond in src_filter_conditions)]
        filtered_dst_instance_infos = [info for info in instance_infos if all(cond(info) for cond in dst_filter_conditions)]
        return filtered_src_instance_infos, filtered_dst_instance_infos

    def filter_src_instances(self, instance_infos: List[InstanceInfo]) -> List[InstanceInfo]:
        src_filter_conditions = [filter.filter_src_condition(self.filter_config)
                                 for filter in self.registered_filters.values()]
        filtered_src_instance_infos = [info for info in instance_infos if all(cond(info) for cond in src_filter_conditions)]
        return filtered_src_instance_infos

    def filter_dst_instances(self, instance_infos: List[InstanceInfo]) -> List[InstanceInfo]:
        dst_filter_conditions = [filter.filter_dst_condition(self.filter_config)
                                 for filter in self.registered_filters.values()]
        filtered_dst_instance_infos = [info for info in instance_infos if all(cond(info) for cond in dst_filter_conditions)]
        return filtered_dst_instance_infos


class LoadFilter(MigrationFilter):
    def filter_src_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        def compare_load(instance_info: InstanceInfo) -> bool:
            metrics_cls = type(instance_info.migration_load_metric)
            migrate_out_load_threshold = metrics_cls(filter_config.migrate_out_load_threshold)
            return instance_info.num_killed_requests > 0 \
                or migrate_out_load_threshold < instance_info.migration_load_metric
        return compare_load

    def filter_dst_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        def compare_load(instance_info: InstanceInfo) -> bool:
            metrics_cls = type(instance_info.migration_load_metric)
            migrate_out_load_threshold = metrics_cls(filter_config.migrate_out_load_threshold)
            return instance_info.num_killed_requests == 0 \
                and instance_info.migration_load_metric < migrate_out_load_threshold

        return compare_load


class CustomFilter(MigrationFilter):
    def __init__(self):
        super().__init__()
        self.src_filter = lambda _: True
        self.dst_filter = lambda _: True

    def set_filter_condtition(self, src_filter: Optional[Callable[[InstanceInfo], bool]] = None,
                              dst_filter: Optional[Callable[[InstanceInfo], bool]] = None) -> None:
        if src_filter:
            self.src_filter = src_filter
        if dst_filter:
            self.dst_filter = dst_filter

    def filter_src_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        return self.src_filter

    def filter_dst_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        return self.dst_filter


class MigrationFilterFactory:
    _POLICY_REGISTRY = {
        'load': LoadFilter,
        'custom': CustomFilter,
    }

    @classmethod
    def get_filter(cls, filter_name: str, **kwargs) -> MigrationFilter:
        return cls._POLICY_REGISTRY[filter_name](**kwargs)
