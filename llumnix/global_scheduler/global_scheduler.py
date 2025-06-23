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

import asyncio
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union, Iterable, Set
import math

import ray
import ray.actor
from ray.util.placement_group import PlacementGroup

from llumnix.logging.logger import init_logger
from llumnix.internal_config import GlobalSchedulerConfig
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints
from llumnix.global_scheduler.scaling_scheduler import ScalingScheduler
from llumnix.llumlet.llumlet import Llumlet
from llumnix.utils import RequestIDType, asyncio_wait_for_with_timeout

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

        self.instance_id_2_engine_inner_inst_id: Dict[str, str] = {}

        self.dispatch_scheduler = DispatchScheduler(global_scheduler_config.dispatch_policy,
                                                    global_scheduler_config.topk_random_dispatch,
                                                    global_scheduler_config.enable_pd_disagg,
                                                    global_scheduler_config.enable_engine_pd_disagg,
                                                    global_scheduler_config.enable_adaptive_pd)

        self.migration_scheduler = MigrationScheduler(global_scheduler_config.pair_migration_policy,
                                                      global_scheduler_config.migrate_out_load_threshold,
                                                      global_scheduler_config.is_group_kind_migration_backend)

        self.scaling_scheduler = ScalingScheduler(global_scheduler_config.scale_up_threshold,
                                                  global_scheduler_config.scale_down_threshold,
                                                  global_scheduler_config.scaling_policy,
                                                  global_scheduler_config.scaling_load_metric,
                                                  global_scheduler_config.enable_pd_disagg)

    def update_instance_infos(self, instance_infos: List[InstanceInfo]) -> None:
        for instance_info in instance_infos:
            if instance_info.instance_id in self.instance_id_set:
                self.instance_info[instance_info.instance_id] = instance_info
                if instance_info.instance_type in (InstanceType.PREFILL, InstanceType.NO_CONSTRAINTS):
                    self.prefill_instance_info[instance_info.instance_id] = instance_info
                if instance_info.instance_type in (InstanceType.DECODE, InstanceType.NO_CONSTRAINTS):
                    self.decode_instance_info[instance_info.instance_id] = instance_info

    def _enable_pd(self):
        return self.global_scheduler_config.enable_pd_disagg \
            or self.global_scheduler_config.enable_engine_pd_disagg \
            or self.global_scheduler_config.enable_adaptive_pd

    def _log_request_dispatch_info(self,
                                   request_id: str,
                                   no_constrains_instance_id: str,
                                   prefill_instance_id: str,
                                   decode_instance_id: str,
                                   expected_steps: int,
                                   addition_dispatch_info: Dict[str, str]):
        if not self._enable_pd():
            logger.info("dispath request {} to instance {}.".format(request_id, no_constrains_instance_id))
        else:
            if self.global_scheduler_config.enable_pd_disagg:
                logger.info("dispath request {} to {} instance ({}), expected_steps: {}.".format(
                    request_id, self.instance_info[prefill_instance_id].instance_type, prefill_instance_id, expected_steps))
            else:
                logger.info("dispath request {} to {} instance ({}) for prefill, {} instance ({}) for decode, addition_dispatch_info: {}.".format(
                    request_id, self.instance_info[prefill_instance_id].instance_type, prefill_instance_id,
                    self.instance_info[decode_instance_id].instance_type, decode_instance_id, addition_dispatch_info))

    def dispatch(self, request_id: RequestIDType) -> Tuple[str, Dict[str, str]]:
        addition_dispatch_info = {}

        # instance_num_requests will be updated inplace in dispatch_scheduler.dispatch
        if not self._enable_pd():
            no_constrains_instance_id = self.dispatch_scheduler.dispatch_no_constrains(
                self.instance_info,
                self.instance_num_requests
            )
            target_instance_id = no_constrains_instance_id
            prefill_instance_id, decode_instance_id = None, None
        else:
            prefill_instance_id, decode_instance_id = self.dispatch_scheduler.dispatch_pd(
                self.instance_info,
                self.instance_num_requests,
                self.prefill_instance_info,
                self.prefill_instance_num_requests,
                self.decode_instance_info,
                self.decode_instance_num_requests
            )
            target_instance_id = prefill_instance_id
            no_constrains_instance_id = None

            if self.global_scheduler_config.enable_engine_pd_disagg:
                addition_dispatch_info["decode_instance_id"] = \
                    self.instance_id_2_engine_inner_inst_id[decode_instance_id]

        # request_expected_steps is only used in llumnix based prefill-decode disagg
        request_expected_steps = math.inf
        if self.global_scheduler_config.enable_pd_disagg and \
            self.instance_info[target_instance_id].instance_type != InstanceType.DECODE:
            request_expected_steps = 1

        self._log_request_dispatch_info(
            request_id=request_id,
            no_constrains_instance_id=no_constrains_instance_id,
            prefill_instance_id=prefill_instance_id,
            decode_instance_id=decode_instance_id,
            expected_steps=request_expected_steps,
            addition_dispatch_info=addition_dispatch_info)

        return target_instance_id, request_expected_steps, addition_dispatch_info

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

    async def _set_engine_self_assigned_id(self, ins_id: str, instance_actor: Llumlet):
        if self.global_scheduler_config.enable_engine_pd_disagg:
            try:
                self.instance_id_2_engine_inner_inst_id[ins_id] = \
                    await asyncio_wait_for_with_timeout(instance_actor.get_engine_disagg_inst_id.remote())
                logger.info("Bind instance id {} with engine instance id {}.".format(
                    ins_id, self.instance_id_2_engine_inner_inst_id[ins_id]))
            # pylint: disable=broad-except
            except Exception as e:
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up instance {}, instance is dead.".format(ins_id))
                elif isinstance(e, ray.exceptions.GetTimeoutError):
                    logger.error("Failed to scale up instance {}, instance is hang, "
                                "please check the cause.".format(ins_id))
                else:
                    logger.exception("Error during scale up instance {}, "
                                    "unexpected exception: {}".format(ins_id, e))
                raise e

    def _remove_engine_self_assigned_id(self, ins_id: str):
        self.instance_id_2_engine_inner_inst_id.pop(ins_id, None)

    async def scale_up(self,
                       instance_id: Union[str, Iterable[str]],
                       instance_actor: Union[Llumlet, List[Llumlet]],
                       instance_type: Union[InstanceType, List[InstanceType]],
                       placement_group: Union[PlacementGroup, Iterable[PlacementGroup]],
                       server: Union[ray.actor.ActorHandle, Iterable[ray.actor.ActorHandle]],
                       manage_scale_up_callback: Optional[Callable]) -> int:
        if isinstance(instance_id, str):
            instance_id, instance_actor, instance_type = [instance_id], [instance_actor], [instance_type]
            placement_group, server = [placement_group], [server]
        instance_ids, instance_actors, instance_types = list(instance_id), list(instance_actor), list(instance_type)
        placement_groups, servers = list(placement_group), list(server)

        available_instance_ids, available_instance_actors, available_instance_types = [], [], []
        available_placement_groups, available_servers = [], []

        def self_assign_id_success_callback(fut, instance_idx: int, scale_up_info: List[List], available_scale_up_info: List[List]):
            ret = fut.result()[0]
            if not isinstance(ret, Exception):
                for item_idx, scale_up_info_item in enumerate(scale_up_info):
                    available_scale_up_info[item_idx].append(scale_up_info_item[instance_idx])

        tasks = []
        for ins_idx, ins_info in enumerate(zip(instance_ids, instance_actors)):
            ins_id, ins_actor = ins_info
            if ins_id not in self.instance_id_set:
                task = asyncio.gather(self._set_engine_self_assigned_id(ins_id, ins_actor), return_exceptions=True)
                task.add_done_callback(
                    partial(
                        self_assign_id_success_callback,
                        instance_idx=ins_idx,
                        scale_up_info=[instance_ids, instance_actors, instance_types, placement_groups, servers],
                        available_scale_up_info=[available_instance_ids, available_instance_actors,
                                                 available_instance_types, available_placement_groups, available_servers])
                    )
                tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

        # Note: await is not allowed below !!!
        # Must keep the atomicity of global_scheduler._add_instance and mananger.scale_up_callback
        for ins_id, ins_actor, ins_type in zip(available_instance_ids, available_instance_actors,
                                               available_instance_types):
            if ins_id not in self.instance_id_set:
                logger.info("Scale up instance {}.".format(ins_id))
                new_intance_info = self._get_empty_instance_info()
                new_intance_info.instance_id = ins_id
                self._add_instance(ins_id, ins_type)
        if manage_scale_up_callback:
            manage_scale_up_callback(
                instance_ids=available_instance_ids,
                instance_actor_handles=available_instance_actors,
                placement_groups=available_placement_groups,
                servers=available_servers)

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
                self._remove_engine_self_assigned_id(ins_id)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def _add_instance(self, instance_id: str, instance_type: InstanceType) -> None:
        new_intance_info = self._get_empty_instance_info()
        new_intance_info.instance_id = instance_id
        new_intance_info.instance_type = instance_type
        self.instance_info[instance_id] = new_intance_info
        self.instance_num_requests[instance_id] = 0
        self.instance_id_set.add(instance_id)
        self.num_instances = len(self.instance_id_set)

        if self._enable_pd():
            if instance_type in (InstanceType.PREFILL, InstanceType.NO_CONSTRAINTS):
                self.prefill_instance_info[instance_id] = new_intance_info
                self.prefill_instance_num_requests[instance_id] = 0
            if instance_type in (InstanceType.DECODE, InstanceType.NO_CONSTRAINTS):
                self.decode_instance_info[instance_id] = new_intance_info
                self.decode_instance_num_requests[instance_id] = 0

    def _remove_instance(self, instance_id: str) -> None:
        if instance_id not in self.instance_id_set:
            logger.warning("instance {} is not in instance_id_set".format(instance_id))
        if instance_id not in self.instance_info:
            logger.warning("instance {} is not in instance_info".format(instance_id))
        instance_info = self.instance_info.get(instance_id, None)
        if instance_info and self._enable_pd():
            if instance_info.instance_type in (InstanceType.PREFILL, InstanceType.NO_CONSTRAINTS):
                self.prefill_instance_info.pop(instance_id, 0)
                self.prefill_instance_num_requests.pop(instance_id, 0)
            if instance_info.instance_type in (InstanceType.DECODE, InstanceType.NO_CONSTRAINTS):
                self.decode_instance_info.pop(instance_id, 0)
                self.decode_instance_num_requests.pop(instance_id, 0)

        self.instance_id_set.discard(instance_id)
        self.instance_info.pop(instance_id, 0)
        self.instance_num_requests.pop(instance_id, 0)
        self.num_instances = len(self.instance_id_set)

    def _get_empty_instance_info(self) -> InstanceInfo:
        return self.scaling_scheduler.get_empty_instance_info()
