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

from typing import Dict, List, Set, Tuple

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.global_scheduler.migration_filter import MigrationInstanceFilter, MigrationFilterConfig, CustomFilter
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints, PairMigrationPolicyFactory

logger = init_logger(__name__)


class MigrationScheduler:
    def __init__(self, pair_migration_policy: str,
                 migrate_out_load_threshold: float,
                 is_group_kind_migration_backend: bool) -> None:
        filter_config = MigrationFilterConfig(migrate_out_load_threshold=migrate_out_load_threshold)
        self.migration_filter = MigrationInstanceFilter(filter_config)
        self._register_migration_backend_init_filter(is_group_kind_migration_backend)

        self.pair_migration_policy = PairMigrationPolicyFactory.get_policy(
            pair_migration_policy, migrate_out_load_threshold=migrate_out_load_threshold)

        self.num_instances = 0
        self.instance_id_set: Set[str] = set()
        self.instance_info: Dict[str, InstanceInfo] = None

    def _register_migration_backend_init_filter(self, is_group_kind_migration_backend: bool) -> None:
        # some migration backends require init_process_group before passing the KV cache. Here, we add a filter
        # to prevent instances of migration backends that have not been initialized from participating in migration.
        migration_backend_init_filter = CustomFilter()
        migration_backend_init_filter.set_filter_condtition(
            src_filter=lambda _: not is_group_kind_migration_backend,
            dst_filter=lambda _: not is_group_kind_migration_backend)
        self.migration_filter.register_filter("migration_backend_init_filter", migration_backend_init_filter)

    # migration_filter must ensure that the specific instance_info does not appear in both src and dst simultaneously
    def pair_migration(self, pair_migration_type: PairMigrationConstraints) -> List[Tuple[str, str]]:
        src_instance_infos, dst_instance_infos = self.migration_filter.filter_instances(
            self.instance_info.values(), pair_migration_type)
        return self.pair_migration_policy.pair_migration(src_instance_infos, dst_instance_infos)

    def update_instance_infos(self, instance_info: Dict[str, InstanceInfo]) -> None:
        self.instance_info = instance_info

    # pylint: disable=unused-argument
    def add_instance(self, instance_id: str, instance_type: InstanceType) -> None:
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)

    def remove_instance(self, instance_id: str) -> None:
        if instance_id in self.instance_id_set:
            self.instance_id_set.remove(instance_id)
        self.num_instances = len(self.instance_id_set)
