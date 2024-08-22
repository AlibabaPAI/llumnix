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

from typing import Dict, List, Tuple, Union, Iterable, Set

from llumnix.logger import init_logger
from llumnix.config import GlobalSchedulerConfig
from llumnix.instance_info import InstanceLoadCalculator, InstanceInfo
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler
from llumnix.global_scheduler.scaling_scheduler import ScalingScheduler

logger = init_logger(__name__)


class GlobalScheduler:
    def __init__(self,
                 global_scheduler_config: GlobalSchedulerConfig) -> None:
        self.global_scheduler_config = global_scheduler_config
        # instance load and instance info args
        self.load_metric = global_scheduler_config.load_metric
        self.enable_defrag = global_scheduler_config.enable_defrag
        self.instance_load_calculator = InstanceLoadCalculator(load_metric=self.load_metric,
                                                               enable_defrag=self.enable_defrag)
        # dispatch args
        self.dispatch_policy = global_scheduler_config.dispatch_policy
        self.dispatch_scheduler = DispatchScheduler(global_scheduler_config.dispatch_policy,
                                                    self.instance_load_calculator)
        # migrate args
        self.migration_scheduler = MigrationScheduler(global_scheduler_config.pair_migration_policy,
                                                      global_scheduler_config.migrate_out_load_threshold,
                                                      self.instance_load_calculator)
        # auto-scaling args
        self.scaling_scheduler = ScalingScheduler(global_scheduler_config.scale_up_threshold,
                                                  global_scheduler_config.scale_down_threshold,
                                                  global_scheduler_config.scaling_policy,
                                                  self.instance_load_calculator)

        self.num_instances = 0
        self.instance_id_set: Set[str] = set()
        self.instance_info: Dict[str, InstanceInfo] = {}

    def update_instance_infos(self, instance_infos: List[InstanceInfo]) -> None:
        for instance_info in instance_infos:
            if instance_info.instance_id in self.instance_id_set:
                # Llumnix have different instance load compuatation methods for dispatch/migrate/scale.
                instance_info.instance_load_dispatch_scale = self.instance_load_calculator.compute_instance_load(instance_info, action='dispatch')
                instance_info.instance_load_migrate = self.instance_load_calculator.compute_instance_load(instance_info, action='migrate')
                self.instance_info[instance_info.instance_id] = instance_info

    def dispatch(self) -> str:
        self.dispatch_scheduler.update_instance_infos(self.instance_info)
        instance_id = self.dispatch_scheduler.dispatch()
        return instance_id

    def pair_migration(self) -> List[Tuple[str, str]]:
        self.migration_scheduler.update_instance_infos(self.instance_info)
        migrate_instance_pairs = self.migration_scheduler.pair_migration()
        return migrate_instance_pairs

    def check_scale(self) -> Tuple[str, str]:
        self.scaling_scheduler.update_instance_infos(self.instance_info)
        scale_up_num, scale_down_num = self.scaling_scheduler.check_scale()
        return scale_up_num, scale_down_num

    def scale_up(self, instance_id: Union[str, Iterable[str]]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            if ins_id not in self.instance_id_set:
                logger.info("scale up instance: {}".format(ins_id))
                new_intance_info = self._get_empty_instance_info()
                new_intance_info.instance_id = ins_id
                self.instance_info[ins_id] = new_intance_info
                self._add_instance(ins_id)
        logger.info("self.num_instances: {}, self.instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def scale_down(self, instance_id: Union[str, Iterable[str]]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            if ins_id in self.instance_id_set:
                logger.info("scale down instance: {}".format(ins_id))
                del self.instance_info[ins_id]
                self._remove_instance(ins_id)
        logger.info("self.num_instances: {}, self.instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def _add_instance(self, instance_id: str) -> None:
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)
        for scheduler in (self.dispatch_scheduler, self.migration_scheduler, self.scaling_scheduler):
            scheduler.update_instance_infos(self.instance_info)
            scheduler.add_instance(instance_id)

    def _remove_instance(self, instance_id: str) -> None:
        self.instance_id_set.remove(instance_id)
        self.num_instances = len(self.instance_id_set)
        for scheduler in (self.dispatch_scheduler, self.migration_scheduler, self.scaling_scheduler):
            scheduler.update_instance_infos(self.instance_info)
            scheduler.remove_instance(instance_id)

    def _get_empty_instance_info(self) -> InstanceInfo:
        return self.scaling_scheduler.get_empty_instance_info()
