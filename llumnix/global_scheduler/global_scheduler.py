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
import math

from llumnix.logging.logger import init_logger
from llumnix.internal_config import GlobalSchedulerConfig
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints
from llumnix.global_scheduler.scaling_scheduler import ScalingScheduler

logger = init_logger(__name__)


class GlobalScheduler:
    def __init__(self, global_scheduler_config: GlobalSchedulerConfig) -> None:
        self.global_scheduler_config = global_scheduler_config
        self.num_instances = 0
        self.instance_id_set: Set[str] = set()
        self.instance_info: Dict[str, InstanceInfo] = {}

        self.prefill_instance_info: Dict[str, InstanceInfo] = {}
        self.prefill_instance_num_requests: Dict[str, int] = {}

        self.decode_instance_info: Dict[str, InstanceInfo] = {}
        self.decode_instance_num_requests: Dict[str, int] = {}

        # dispatch args
        self.dispatch_scheduler = DispatchScheduler(global_scheduler_config.dispatch_policy,
                                                    global_scheduler_config.topk_random_dispatch)
        # migrate args
        self.migration_scheduler = MigrationScheduler(global_scheduler_config.pair_migration_policy,
                                                      global_scheduler_config.migrate_out_load_threshold,
                                                      global_scheduler_config.is_group_kind_migration_backend)
        # auto-scaling args
        self.scaling_scheduler = ScalingScheduler(global_scheduler_config.scale_up_threshold,
                                                  global_scheduler_config.scale_down_threshold,
                                                  global_scheduler_config.scaling_policy,
                                                  global_scheduler_config.scaling_load_metric,
                                                  global_scheduler_config.enable_pd_disagg)

    def update_instance_infos(self, instance_infos: List[InstanceInfo]) -> None:
        for instance_info in instance_infos:
            if instance_info.instance_id in self.instance_id_set:
                self.instance_info[instance_info.instance_id] = instance_info

    def dispatch(self, instance_type: InstanceType = InstanceType.PREFILL) -> str:
        if instance_type in (InstanceType.PREFILL, InstanceType.NO_CONSTRAINTS):
            instance_id = self.dispatch_scheduler.dispatch(
                instance_info=self.prefill_instance_info,
                instance_num_requests=self.prefill_instance_num_requests,
            )
            self.prefill_instance_num_requests[instance_id] += 1
        elif instance_type == InstanceType.DECODE:
            instance_id = self.dispatch_scheduler.dispatch(
                instance_info=self.decode_instance_info,
                instance_num_requests=self.decode_instance_num_requests,
            )
            self.decode_instance_num_requests[instance_id] += 1
        else:
            logger.error("instance_type {} is not supported".format(instance_type))
            raise TypeError("instance_type {} is not supported".format(instance_type))
        if self.global_scheduler_config.enable_pd_disagg and instance_type in (
            InstanceType.PREFILL,
            InstanceType.NO_CONSTRAINTS,
        ):
            request_expected_steps = 1
        else:
            request_expected_steps = math.inf
        return instance_id, request_expected_steps

    def pair_migration(
        self, pair_migration_type: PairMigrationConstraints
    ) -> List[Tuple[str, str]]:
        migrate_instance_pairs = self.migration_scheduler.pair_migration(
            instance_info=self.instance_info, pair_migration_type=pair_migration_type
        )
        return migrate_instance_pairs

    def check_scale(self) -> Tuple[str, str]:
        scale_up_num, scale_down_num = self.scaling_scheduler.check_scale(self.instance_info, self.instance_id_set)
        return scale_up_num, scale_down_num

    def scale_up(self, instance_id: Union[str, Iterable[str]], instance_type: List[InstanceType]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id, ins_type in zip(instance_ids, instance_type):
            if ins_id not in self.instance_id_set:
                logger.info("Scale up instance: {}.".format(ins_id))
                new_intance_info = self._get_empty_instance_info()
                new_intance_info.instance_id = ins_id
                self.instance_info[ins_id] = new_intance_info
                self._add_instance(ins_id, ins_type)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def scale_down(self, instance_id: Union[str, Iterable[str]]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            if ins_id in self.instance_id_set:
                self._remove_instance(ins_id)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def _add_instance(self, instance_id: str, instance_type: InstanceType) -> None:
        logger.info("Scale up instance: {}.".format(instance_id))
        new_intance_info = self._get_empty_instance_info()
        new_intance_info.instance_id = instance_id
        new_intance_info.instance_type = instance_type
        self.instance_info[instance_id] = new_intance_info
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)
        if instance_type in (InstanceType.PREFILL, InstanceType.NO_CONSTRAINTS):
            self.prefill_instance_info[instance_id] = new_intance_info
            self.prefill_instance_num_requests[instance_id] = 0
        elif instance_type == InstanceType.DECODE:
            self.decode_instance_info[instance_id] = new_intance_info
            self.decode_instance_num_requests[instance_id] = 0

    def _remove_instance(self, instance_id: str) -> None:
        logger.info("Scale down instance: {}.".format(instance_id))
        if instance_id not in self.instance_id_set:
            logger.warning("instance {} is not in instance_id_set".format(instance_id))
        if instance_id not in self.instance_info:
            logger.warning("instance {} is not in instance_info".format(instance_id))
        instance_info = self.instance_info.get(instance_id, None)
        if instance_info:
            if instance_info.instance_type in (InstanceType.PREFILL, InstanceType.NO_CONSTRAINTS):
                self.prefill_instance_info.pop(instance_id, 0)
                self.prefill_instance_num_requests.pop(instance_id, 0)
            elif instance_info.instance_type == InstanceType.DECODE:
                self.decode_instance_info.pop(instance_id, 0)
                self.decode_instance_num_requests.pop(instance_id, 0)

        self.instance_id_set.discard(instance_id)
        self.instance_info.pop(instance_id, 0)
        self.num_instances = len(self.instance_id_set)

    def _get_empty_instance_info(self) -> InstanceInfo:
        return self.scaling_scheduler.get_empty_instance_info()
