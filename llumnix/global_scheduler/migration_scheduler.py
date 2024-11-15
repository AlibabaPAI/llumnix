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

from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator
from llumnix.global_scheduler.migration_filter import MigrationInstanceFilter, MigrationFilterConfig
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints, PairMigrationPolicyFactory

logger = init_logger(__name__)

class MigrationScheduler:
    def __init__(self,
                 pair_migration_policy: str,
                 migrate_out_load_threshold: float,
                 instance_load_calculator: InstanceLoadCalculator) -> None:
        self.filter_config = MigrationFilterConfig(migrate_out_load_threshold=migrate_out_load_threshold)
        self.migration_filter = MigrationInstanceFilter(self.filter_config)

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
        src_instance_infos, dst_instance_infos = self.migration_filter.filter_instances(
            self.instance_info.values(), pair_migration_type)
        return self.pair_migration_policy.pair_migration(src_instance_infos, dst_instance_infos)

    def update_instance_infos(self,
                              instance_info: Dict[str, InstanceInfo]) -> None:
        self.instance_info = instance_info

    def add_instance(self, instance_id: str) -> None:
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)

    def remove_instance(self, instance_id: str) -> None:
        self.instance_id_set.remove(instance_id)
        self.num_instances = len(self.instance_id_set)
