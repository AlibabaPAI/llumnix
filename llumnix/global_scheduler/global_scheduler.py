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
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler
from llumnix.global_scheduler.scaling_scheduler import ScalingScheduler
from llumnix.metrics.global_scheduler_metrics import GlobalSchedulerMetrics
from llumnix.utils import RequestIDType, InstanceType
from llumnix.instance_info import InstanceInfo

logger = init_logger(__name__)


class GlobalScheduler:
    def __init__(self, global_scheduler_config: GlobalSchedulerConfig) -> None:
        self.global_scheduler_config = global_scheduler_config
        self.num_instances = 0

        self.instance_id_set: Set[str] = set()
        self.instance_info: Dict[str, InstanceInfo] = {}
        self.instance_num_requests: Dict[str, int] = {}

        self.prefill_instance_info: Dict[str, InstanceInfo] = {}
        self.prefill_instance_num_requests: Dict[str, int] = {}

        self.decode_instance_info: Dict[str, InstanceInfo] = {}
        self.decode_instance_num_requests: Dict[str, int] = {}

        self.global_scheduler_metrics = GlobalSchedulerMetrics()

        self.dispatch_scheduler = DispatchScheduler(global_scheduler_config.dispatch_policy,
                                                    global_scheduler_config.topk_random_dispatch,
                                                    global_scheduler_config.enable_pd_disagg,
                                                    global_scheduler_config.enable_engine_pd_disagg,
                                                    global_scheduler_config.enable_engine_semi_pd_disagg,
                                                    global_scheduler_config.enable_adaptive_pd,
                                                    global_scheduler_config.dispatch_load_metric_config,
                                                    self.global_scheduler_metrics.dispatch_latency,
                                                    global_scheduler_config.cache_meta_client_config_path)

        self.migration_scheduler = MigrationScheduler(global_scheduler_config.pair_migration_policy,
                                                      global_scheduler_config.migrate_out_load_threshold,
                                                      global_scheduler_config.is_group_kind_migration_backend,
                                                      global_scheduler_config.enable_pd_disagg,
                                                      global_scheduler_config.enable_engine_pd_disagg,
                                                      global_scheduler_config.enable_engine_semi_pd_disagg,
                                                      global_scheduler_config.enable_adaptive_pd,
                                                      global_scheduler_config.dispatch_load_metric_config,
                                                      global_scheduler_config.enable_pre_step_migration)

        self.scaling_scheduler = ScalingScheduler(global_scheduler_config.scale_up_threshold,
                                                  global_scheduler_config.scale_down_threshold,
                                                  global_scheduler_config.scaling_policy,
                                                  global_scheduler_config.scaling_load_metric,
                                                  global_scheduler_config.enable_pd_disagg)

    # TODO(baizhuoyan): Since dispatch_load_metric is no longer available, it is temporarily replaced with remaining_steps.
    def update_instance_infos(self, instance_infos: List[InstanceInfo]) -> None:
        for instance_info in instance_infos:
            self.global_scheduler_metrics.dispatch_load.observe(
                value=instance_info.remaining_steps,
                labels={"instance_id": instance_info.instance_id},
            )
            if instance_info.instance_id in self.instance_id_set:
                print(f"[zzy][load] updating instance_info: {instance_info}")
                self.instance_info[instance_info.instance_id] = instance_info
                if instance_info.instance_type in (InstanceType.PREFILL, InstanceType.NEUTRAL):
                    self.prefill_instance_info[instance_info.instance_id] = instance_info
                if instance_info.instance_type in (InstanceType.DECODE, InstanceType.NEUTRAL):
                    self.decode_instance_info[instance_info.instance_id] = instance_info

    def _enable_pd(self):
        return self.global_scheduler_config.enable_pd_disagg \
            or self.global_scheduler_config.enable_engine_pd_disagg \
            or self.global_scheduler_config.enable_engine_semi_pd_disagg

    def _log_request_dispatch_info(self,
                                   request_id: str,
                                   prefill_instance_id: str,
                                   decode_instance_id: str,
                                   expected_steps: int):
        if not self._enable_pd():
            # when enable_pd_disagg is False, prefill_instance_id and decode_instance_id are the same
            logger.info("dispatch request {} to instance {}.".format(request_id, prefill_instance_id))
        else:
            if self.global_scheduler_config.enable_pd_disagg:
                logger.info("dispatch request {} to {} instance ({}), expected_steps: {}.".format(
                    request_id, self.instance_info[prefill_instance_id].instance_type, prefill_instance_id, expected_steps))
            else:
                logger.info("dispatch request {} to {} instance ({}) for prefill, {} instance ({}) for decode.".format(
                    request_id, self.instance_info[prefill_instance_id].instance_type, prefill_instance_id,
                    self.instance_info[decode_instance_id].instance_type, decode_instance_id))

    def dispatch(self, request_id: RequestIDType, dispatch_context: Dict) -> Tuple[str, str, int]:
        # instance_num_requests will be updated inplace in dispatch_scheduler.dispatch
        if not self._enable_pd():
            no_constrains_instance_id = self.dispatch_scheduler.dispatch_no_constrains(
                self.instance_info,
                self.instance_num_requests,
                dispatch_context,
            )
            prefill_instance_id = no_constrains_instance_id
            decode_instance_id = no_constrains_instance_id

            self.global_scheduler_metrics.dispatch_counter.increase(
                labels={"instance_id": no_constrains_instance_id}
            )
        else:
            no_constrains_instance_id = None
            prefill_instance_id, decode_instance_id = self.dispatch_scheduler.dispatch_pd(
                self.instance_info,
                self.instance_num_requests,
                self.prefill_instance_info,
                self.prefill_instance_num_requests,
                self.decode_instance_info,
                self.decode_instance_num_requests,
                dispatch_context,
            )

            self.global_scheduler_metrics.dispatch_counter.increase(
                labels={"instance_id": prefill_instance_id}
            )
            self.global_scheduler_metrics.dispatch_counter.increase(
                labels={"instance_id": decode_instance_id}
            )

        # request_expected_steps is only used in llumnix based prefill-decode disagg
        request_expected_steps = math.inf
        if self.global_scheduler_config.enable_pd_disagg and \
            self.instance_info[prefill_instance_id].instance_type != InstanceType.DECODE:
            request_expected_steps = 1

        self._log_request_dispatch_info(
            request_id=request_id,
            prefill_instance_id=prefill_instance_id,
            decode_instance_id=decode_instance_id,
            expected_steps=request_expected_steps)

        return prefill_instance_id, decode_instance_id, request_expected_steps

    def push_migrations(self, instance_info: Dict[str, InstanceInfo]) -> List[List[Tuple[str, str]]]:
        return self.migration_scheduler.push_migrations(instance_info)

    def check_scale(self) -> Tuple[str, str]:
        scale_up_num, scale_down_num = self.scaling_scheduler.check_scale(self.instance_info, self.instance_id_set)
        return scale_up_num, scale_down_num

    def scale_up(self,
                 instance_id: Union[str, Iterable[str]],
                 instance_type: Union[InstanceType, List[InstanceType]]) -> int:
        if isinstance(instance_id, str):
            instance_id, instance_type = [instance_id], [instance_type]
        instance_ids, instance_types = list(instance_id), list(instance_type)

        for ins_id, ins_type in zip(instance_ids, instance_types):
            if ins_id not in self.instance_id_set:
                logger.info("Scale up instance {}.".format(ins_id))
                self._add_instance(ins_id, ins_type)

        logger.info("num_instances: {}, instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def scale_down(self, instance_id: Union[str, Iterable[str]]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            if ins_id in self.instance_id_set:
                logger.info("Scale down instance {}".format(ins_id))
                self._remove_instance(ins_id)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def _add_instance(self,
                      instance_id: str,
                      instance_type: InstanceType) -> None:
        # pylint: disable=consider-iterating-dictionary
        if instance_id not in self.instance_info.keys():
            instance_info = self._get_empty_instance_info()
            instance_info.instance_id = instance_id
            instance_info.instance_type = instance_type
            self.instance_info[instance_id] = instance_info
        self.instance_num_requests[instance_id] = 0
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)

        if self._enable_pd():
            instance_info = self.instance_info[instance_id]
            if instance_type in (InstanceType.PREFILL, InstanceType.NEUTRAL):
                self.prefill_instance_info[instance_id] = instance_info
                self.prefill_instance_num_requests[instance_id] = 0
            if instance_type in (InstanceType.DECODE, InstanceType.NEUTRAL):
                self.decode_instance_info[instance_id] = instance_info
                self.decode_instance_num_requests[instance_id] = 0

    def _remove_instance(self, instance_id: str) -> None:
        if instance_id not in self.instance_id_set:
            logger.warning("instance {} is not in instance_id_set".format(instance_id))
        if instance_id not in self.instance_info:
            logger.warning("instance {} is not in instance_info".format(instance_id))
        instance_info = self.instance_info.get(instance_id, None)
        if instance_info and self._enable_pd():
            if instance_info.instance_type in (InstanceType.PREFILL, InstanceType.NEUTRAL):
                self.prefill_instance_info.pop(instance_id, 0)
                self.prefill_instance_num_requests.pop(instance_id, 0)
            if instance_info.instance_type in (InstanceType.DECODE, InstanceType.NEUTRAL):
                self.decode_instance_info.pop(instance_id, 0)
                self.decode_instance_num_requests.pop(instance_id, 0)

        self.instance_id_set.discard(instance_id)
        self.instance_info.pop(instance_id, 0)
        self.instance_num_requests.pop(instance_id, 0)
        self.num_instances = len(self.instance_id_set)

    def _get_empty_instance_info(self) -> InstanceInfo:
        return self.scaling_scheduler.get_empty_instance_info()

    def all_instances_not_migrating(self) -> bool:
        return all(not instance_info.is_migrating for instance_info in self.instance_info.values())
