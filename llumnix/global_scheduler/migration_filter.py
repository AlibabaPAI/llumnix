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

from typing import Callable, Dict, List, Optional
from abc import ABC, abstractmethod

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.global_scheduler.scaling_scheduler import InstanceType
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints

logger = init_logger(__name__)


class MigrationFilterConfig:
    def __init__(self, migrate_out_load_threshold):
        self.migrate_out_load_threshold: float = migrate_out_load_threshold


# TODO(KuilongCui): A filter might contain other filters; leave this for the future.
class MigrationFilterPolicy(ABC):
    @abstractmethod
    def filter_src_condition(self, filter_config, pair_migration_type) -> Callable[[InstanceInfo], bool]:
        raise NotImplementedError

    @abstractmethod
    def filter_dst_condition(self, filter_config, pair_migration_type) -> Callable[[InstanceInfo], bool]:
        raise NotImplementedError


class MigrationInstanceFilter(ABC):
    def __init__(self, filter_config: MigrationFilterConfig) -> None:
        self.filter_config = filter_config
        self.registered_filters: Dict[str, MigrationFilterPolicy] = {}

    def register_filter(self, filter_name: str, migration_filter: MigrationFilterPolicy) -> bool:
        if filter_name in self.registered_filters:
            logger.warning("Migration filter {} has been registered.".format(filter_name))
            return False

        self.registered_filters[filter_name] = migration_filter
        return True

    def unregister_filter(self, filter_name: str) -> None:
        self.registered_filters.pop(filter_name, None)

    def get_filter(self, filter_name: str) -> Optional[MigrationFilterPolicy]:
        return self.registered_filters.get(filter_name, None)

    def filter_instances(self, instance_infos: List[InstanceInfo],
                         pair_migration_type: PairMigrationConstraints) -> Dict[str, InstanceInfo]:
        src_filter_conditions = [filter.filter_src_condition(self.filter_config, pair_migration_type)
                                 for filter in self.registered_filters.values()]
        dst_filter_conditions = [filter.filter_dst_condition(self.filter_config, pair_migration_type)
                                 for filter in self.registered_filters.values()]

        if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
            policy_filter = MigrationFilterPolicyFactory.get_policy("load")
        elif pair_migration_type in [PairMigrationConstraints.PREFILL_2_DECODING, PairMigrationConstraints.DECODING_2_DECODING]:
            policy_filter = MigrationFilterPolicyFactory.get_policy('prefill_decode')
        else:
            raise ValueError(f"Unsupported pair migration type: {pair_migration_type}")
        src_filter_conditions.append(policy_filter.filter_src_condition(self.filter_config, pair_migration_type))
        dst_filter_conditions.append(policy_filter.filter_dst_condition(self.filter_config, pair_migration_type))

        filtered_src_instance_infos = [info for info in instance_infos if all(cond(info) for cond in src_filter_conditions)]
        filtered_dst_instance_infos = [info for info in instance_infos if all(cond(info) for cond in dst_filter_conditions)]

        return filtered_src_instance_infos, filtered_dst_instance_infos


class LoadConstrainedFilter(MigrationFilterPolicy):
    def filter_src_condition(self, filter_config: MigrationFilterConfig,
                             pair_migration_type: PairMigrationConstraints) -> Callable[[InstanceInfo], bool]:
        return lambda instance_info: instance_info.num_killed_requests > 0 \
            or instance_info.instance_load_migrate > filter_config.migrate_out_load_threshold

    def filter_dst_condition(self, filter_config: MigrationFilterConfig,
                             pair_migration_type: PairMigrationConstraints) -> Callable[[InstanceInfo], bool]:
        return lambda instance_info: instance_info.num_killed_requests == 0 \
            and instance_info.instance_load_migrate < filter_config.migrate_out_load_threshold


class PddFilter(MigrationFilterPolicy):
    INSTANCE_FILTER_RULES = {
        PairMigrationConstraints.DECODING_2_DECODING: (InstanceType.DECODE, InstanceType.DECODE),
        PairMigrationConstraints.PREFILL_2_DECODING: (InstanceType.PREFILL, InstanceType.DECODE),
    }

    def filter_src_condition(self, filter_config: MigrationFilterConfig,
                             pair_migration_type: PairMigrationConstraints) -> Callable[[InstanceInfo], bool]:
        src_type, _ = self.INSTANCE_FILTER_RULES[pair_migration_type]
        instance_type_filter = lambda instance_info: instance_info.instance_type == src_type

        if pair_migration_type == PairMigrationConstraints.DECODING_2_DECODING:
            inner_policy = MigrationFilterPolicyFactory.get_policy('load')
            policy_filter = inner_policy.filter_src_condition(filter_config, pair_migration_type)
        else:
            policy_filter = lambda instance_info: True

        return lambda instance_info: instance_type_filter(instance_info) and policy_filter(instance_info)

    def filter_dst_condition(self, filter_config: MigrationFilterConfig,
                             pair_migration_type: PairMigrationConstraints) -> Callable[[InstanceInfo], bool]:
        _, dst_type = self.INSTANCE_FILTER_RULES[pair_migration_type]
        instance_type_filter = lambda instance_info: instance_info.instance_type == dst_type

        if pair_migration_type == PairMigrationConstraints.DECODING_2_DECODING:
            inner_policy = MigrationFilterPolicyFactory.get_policy('load')
            policy_filter = inner_policy.filter_dst_condition(filter_config, pair_migration_type)
        else:
            policy_filter = lambda instance_info: instance_info.num_killed_requests == 0

        return lambda instance_info: instance_type_filter(instance_info) and policy_filter(instance_info)


class CustomFilter(MigrationFilterPolicy):
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

    def filter_src_condition(self, filter_config: MigrationFilterConfig,
                             pair_migration_type: PairMigrationConstraints) -> Callable[[InstanceInfo], bool]:
        return self.src_filter

    def filter_dst_condition(self, filter_config: MigrationFilterConfig,
                             pair_migration_type: PairMigrationConstraints) -> Callable[[InstanceInfo], bool]:
        return self.dst_filter


class MigrationFilterPolicyFactory:
    _POLICY_REGISTRY = {
        'load': LoadConstrainedFilter,
        'prefill_decode': PddFilter,
        'custom': CustomFilter,
    }

    @classmethod
    def get_policy(cls, policy_name: PairMigrationConstraints, **kwargs) -> MigrationFilterPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
