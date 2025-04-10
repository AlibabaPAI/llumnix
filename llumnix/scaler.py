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
import copy
import traceback
import random
from typing import List, Tuple, Dict

import ray
from ray.util.placement_group import PlacementGroup
import ray.actor

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceType
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_type import QueueType
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import EntrypointsArgs, InstanceArgs, ManagerArgs, LaunchArgs
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.backends.utils import get_engine_world_size
from llumnix.utils import (load_engine_args, get_service_resouces, random_uuid,
                           get_service_instance_type)
from llumnix.ray_utils import (initialize_placement_group, get_manager_name, get_server_name,
                               get_data_from_ray_internal_kv, put_data_to_ray_internal_kv,
                               GPUBundlingStrategy, get_scaler_name, get_placement_group_name,
                               get_placement_group_infos_by_name, get_placement_group_infos_by_state,
                               kill_server, kill_instance, remove_placement_group,
                               get_actor_names_by_name_prefix, SERVER_NAME_PREFIX, INSTANCE_NAME_PREFIX,
                               actor_exists, get_instance_name)
from llumnix.internal_config import PDDConfig
from llumnix.constants import (INSTANCE_READY_TIMEOUT, SERVER_READY_TIMEOUT,
                               WAIT_PLACEMENT_GROUP_TIMEOUT, AUTO_SCALE_UP_INTERVAL,
                               CHECK_DEPLOYMENT_STATES_INTERVAL, WATCH_DEPLOYMENT_INTERVAL)
from llumnix.entrypoints.utils import LaunchMode

logger = init_logger(__name__)


class Scaler:
    def __init__(self,
                 entrypoints_args: EntrypointsArgs,
                 manager_args: ManagerArgs,
                 instance_args: InstanceArgs,
                 engine_args,
                 launch_args: LaunchArgs,
                 enable_port_increment: bool,
                 enable_port_offset_store: bool,
                 load_registered_service: bool,
                 load_registered_service_path: str,
                 pdd_config: PDDConfig):
        self.entrypoints_args = entrypoints_args
        self.manager_args = manager_args
        self.engine_args = engine_args
        self.instance_args = instance_args

        # manager_args
        self.max_instances = manager_args.max_instances

        # launch args
        if launch_args is not None:
            self.launch_mode: LaunchMode = launch_args.launch_mode
            self.backend_type: BackendType = launch_args.backend_type

        self.enable_port_increment = enable_port_increment
        self.enable_port_offset_store = enable_port_offset_store
        self.load_registered_service = load_registered_service
        self.load_registered_service_path = load_registered_service_path
        self.pdd_config = pdd_config

        if enable_port_increment:
            self.port_offset = 0
            if enable_port_offset_store:
                # TODO(s5u13b): Do not use ray interval kv.
                value = get_data_from_ray_internal_kv("manager.port_offset")
                self.port_offset = int(value)

        if self.load_registered_service:
            self.engine_args_dict = {}
            if not self.pdd_config.enable_pd_disagg and not self.pdd_config.enable_engine_pd_disagg:
                instance_type_list = ['no_constraints']
            else:
                instance_type_list = ['prefill', 'decode']
            for instance_type in instance_type_list:
                self.engine_args_dict[instance_type] = load_engine_args(instance_type, self.load_registered_service_path)

        self.inflight_num_prefill_instances = 0
        self.inflight_num_decode_instances = 0

        self.manager: ray.actor.ActorHandle = ray.get_actor(get_manager_name(), namespace="llumnix")

        if hasattr(self, "launch_mode") and self.launch_mode == LaunchMode.GLOBAL:
            assert self.entrypoints_args is not None and self.engine_args is not None
            self.last_timeout_instance_id = None
            if self.pdd_config.enable_pdd_node_affinity_scheduling:
                asyncio.create_task(self._auto_scale_up_loop(service_name="prefill",
                                                             max_instances=self.max_instances,
                                                             interval=AUTO_SCALE_UP_INTERVAL))
                asyncio.create_task(self._auto_scale_up_loop(service_name="decode",
                                                             max_instances=self.max_instances,
                                                             interval=AUTO_SCALE_UP_INTERVAL))
            else:
                asyncio.create_task(self._auto_scale_up_loop(service_name="no_constraints",
                                                             max_instances=self.max_instances,
                                                             interval=AUTO_SCALE_UP_INTERVAL))
            asyncio.create_task(self._check_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))
            if self.pdd_config.enable_pd_disagg:
                asyncio.create_task(self._check_pd_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))

    @classmethod
    def from_args(cls,
                  entrypoints_args: EntrypointsArgs,
                  manager_args: ManagerArgs,
                  instance_args: InstanceArgs,
                  engine_args,
                  launch_args: LaunchArgs,
                  enable_port_increment: bool,
                  enable_port_offset_store: bool,
                  load_registered_service: bool,
                  load_registered_service_path: str,
                  pdd_config: PDDConfig,
                  node_id: str):
        scaler_class = ray.remote(num_cpus=1,
                                  max_restarts=-1,
                                  name=get_scaler_name(),
                                  namespace="llumnix",
                                  lifetime="detached")(cls).options(
                                      scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                                        node_id=node_id,
                                        soft=False,
                                    )
                                  )
        scaler = scaler_class.remote(entrypoints_args,
                                     manager_args,
                                     instance_args,
                                     engine_args,
                                     launch_args,
                                     enable_port_increment,
                                     enable_port_offset_store,
                                     load_registered_service,
                                     load_registered_service_path,
                                     pdd_config)

        return scaler

    async def _auto_scale_up_loop(self, service_name: str, max_instances: int, interval: float) -> None:
        logger.info("Auto scale up loop starts, service name: {}".format(service_name))
        while True:
            try:
                new_pg = None
                if self.last_timeout_instance_id is not None:
                    last_timeout_pg_name = get_placement_group_name(self.last_timeout_instance_id)
                    last_timeout_pg_infos = get_placement_group_infos_by_name(name=last_timeout_pg_name)
                    if len(last_timeout_pg_infos) > 0 and last_timeout_pg_infos[0]["state"] != "REMOVED":
                        new_instance_id = self.last_timeout_instance_id
                        # pending, created(without server and instance) or rescheduling
                        new_pg = ray.util.get_placement_group(last_timeout_pg_name)
                    # reset
                    self.last_timeout_instance_id = None
                pending_pg_infos = get_placement_group_infos_by_state(state="PENDING")
                pending_pg_infos.extend(get_placement_group_infos_by_state(state="RESCHEDULING"))
                for pending_pg_info in pending_pg_infos:
                    instance_id = pending_pg_info["name"].split("_")[-1]
                    if new_pg is not None and instance_id == new_instance_id:
                        continue
                    self.clear_instance_ray_resources(instance_id)
                alive_pg_infos = get_placement_group_infos_by_state(state="CREATED")
                alive_pg_infos.extend(get_placement_group_infos_by_state(state="PENDING"))
                alive_pg_infos.extend(get_placement_group_infos_by_state(state="RESCHEDULING"))
                if max_instances != -1 and len(alive_pg_infos) >= max_instances:
                    logger.debug("The number of alive placement groups has reached the max_instances.")
                    await asyncio.sleep(interval)
                    continue
                if new_pg is None:
                    new_instance_id = random_uuid()
                    new_pg = self._init_placement_group(get_placement_group_name(new_instance_id), self.engine_args,
                                                        self.backend_type, init_server=True, block=False,
                                                        service_name=service_name)
                try:
                    await asyncio.wait_for(new_pg.ready(), WAIT_PLACEMENT_GROUP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug("Waiting for new placement group {} ready timeout.".format(new_instance_id))
                    # After timeout, the new placement group might be pending,
                    # created(without server and instance), rescheduling.
                    self.last_timeout_instance_id = new_instance_id
                    await asyncio.sleep(interval)
                    continue
                if service_name in ["prefill", "decode"]:
                    await self._init_server_and_instance(new_instance_id, self.entrypoints_args, self.instance_args,
                                                         self.engine_args, self.backend_type, new_pg,
                                                         instance_type=get_service_instance_type(service_name))
                else:
                    await self._init_server_and_instance(new_instance_id, self.entrypoints_args, self.instance_args,
                                                         self.engine_args, self.backend_type, new_pg)
                logger.info("Deploy server and instance to new placement group done, "
                            "instance_id: {}.".format(new_instance_id))
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))
                await asyncio.sleep(interval)

    async def _check_deployment_states_loop(self, interval: float) -> None:
        async def watch_instance_deployment_states(instance_id: str):
            # There might be some delays of calling _init_server_and_instance, so sleep first.
            await asyncio.sleep(WATCH_DEPLOYMENT_INTERVAL)
            pg_created, server_exists, instance_exists = self._get_instance_deployment_states(instance_id)
            if pg_created and (not server_exists or not instance_exists):
                logger.warning("Instance {} deployment states incorrect, states: (pg {}, server {}, instance {})"
                               .format(instance_id, pg_created, server_exists, instance_exists))
                await self.manager.scale_down.remote(instance_id)

        while True:
            try:
                curr_pgs, curr_servers, curr_instances = self._get_cluster_deployment_states()
                assert len(curr_pgs) >= max(len(curr_servers), len(curr_instances))
                tasks = []
                for instance_id in curr_pgs:
                    if instance_id not in curr_servers or instance_id not in curr_instances:
                        tasks.append(asyncio.create_task(watch_instance_deployment_states(instance_id)))
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(interval)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    # TODO(KuilongCui): Deploy prefill and decode instances strictly according to the pd_ratio.
    # Currently, only one naive prefill-decode disaggregation deployment states check policy is implemented,
    # which prevents all instances in the cluster are prefill instances or decode instances.
    async def _check_pd_deployment_states_loop(self, interval: float) -> None:
        previous_penging_pg_names = None
        while True:
            try:
                pending_pg_infos = get_placement_group_infos_by_state(state="PENDING")
                rescheduling_pg_infos = get_placement_group_infos_by_state(state="RESCHEDULING")
                all_penging_pg_names = [pg["name"] for pg in pending_pg_infos]
                if previous_penging_pg_names and len(rescheduling_pg_infos) == 0 :
                    new_pending_pg_infos = get_placement_group_infos_by_state(state="PENDING")
                    all_new_penging_pg_names = [pg["name"] for pg in new_pending_pg_infos]
                    if len(set(previous_penging_pg_names).difference(set(all_new_penging_pg_names))) == 0:
                        await self._check_pd_deployment_states()
                    previous_penging_pg_names = all_new_penging_pg_names
                else:
                    previous_penging_pg_names = all_penging_pg_names

                await asyncio.sleep(interval)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    async def _check_pd_deployment_states(self) -> str:
        prefill_instance_id_set, decode_instance_id_set = await self.manager.get_prefill_decode_instance_id_set.remote()
        cur_num_prefill_instances = len(prefill_instance_id_set)
        cur_num_decode_instances = len(decode_instance_id_set)

        scale_down_instance_id = None
        if cur_num_prefill_instances == 0 and cur_num_decode_instances > 0:
            scale_down_instance_id = random.choice(list(decode_instance_id_set))
            logger.info("Check pd deployment, pd_ratio: {}, cur_num_prefill_instances: {}, cur_num_decode_instances: {}, "
                        "all decode instances is decode instance, scale down decode instance {}".format(
                        self.pdd_config.pd_ratio, cur_num_prefill_instances, cur_num_decode_instances, scale_down_instance_id))

        if cur_num_decode_instances == 0 and cur_num_prefill_instances > 0:
            scale_down_instance_id = random.choice(list(prefill_instance_id_set))
            logger.info("Check pd deployment, pd_ratio: {}, cur_num_prefill_instances: {}, cur_num_decode_instances: {}, "
                        "all instances is prefill instance, scale down prefill instance {}".format(
                        self.pdd_config.pd_ratio, cur_num_prefill_instances, cur_num_decode_instances, scale_down_instance_id))

        if scale_down_instance_id:
            await self.manager.scale_down.remote(scale_down_instance_id)

        return scale_down_instance_id

    def _get_cluster_deployment_states(self) -> Tuple[Dict[str, PlacementGroup], Dict[str, APIServerActor], Dict[str, Llumlet]]:
        curr_pgs: Dict[str, PlacementGroup] = {}
        curr_servers: Dict[str, PlacementGroup] = {}
        curr_instances: Dict[str, Llumlet] = {}

        created_pg_infos = get_placement_group_infos_by_state(state="CREATED")
        for created_pg_info in created_pg_infos:
            instance_id = created_pg_info["name"].split("_")[-1]
            try:
                curr_pgs[instance_id] = ray.util.get_placement_group(created_pg_info["name"])
            except ValueError:
                continue

        curr_server_names = get_actor_names_by_name_prefix(name_prefix=SERVER_NAME_PREFIX)
        for curr_server_name in curr_server_names:
            instance_id = curr_server_name.split("_")[-1]
            try:
                curr_servers[instance_id] = ray.get_actor(curr_server_name, namespace="llumnix")
            except ValueError:
                continue

        curr_instance_names = get_actor_names_by_name_prefix(name_prefix=INSTANCE_NAME_PREFIX)
        for curr_instance_name in curr_instance_names:
            instance_id = curr_instance_name.split("_")[-1]
            try:
                curr_instances[instance_id] = ray.get_actor(curr_instance_name, namespace="llumnix")
            except ValueError:
                continue

        return curr_pgs, curr_servers, curr_instances

    def _get_instance_deployment_states(self, instance_id: str):
        pg_infos = get_placement_group_infos_by_name(name=get_placement_group_name(instance_id))
        pg_created = len(pg_infos) == 1 and pg_infos[0]["state"] == "CREATED"
        server_exists = actor_exists(get_server_name(instance_id))
        instance_exists = actor_exists(get_instance_name(instance_id))

        return pg_created, server_exists, instance_exists

    def _init_placement_group(self,
                              placement_group_name: str,
                              engine_args,
                              backend_type: BackendType,
                              init_server: bool = False,
                              block: bool = True,
                              service_name: str = None
                              ) -> PlacementGroup:
        # num_cpus=2+(0/1), for Llumlet + AsyncPutQueueActor + (ApiServerActor)
        if not BackendType.is_sim_backend(backend_type):
            # num_gpus=world_size, for world_size Workers
            world_size = get_engine_world_size(engine_args, backend_type)
            gpu_bundling_strategy = GPUBundlingStrategy.SPREAD if backend_type == BackendType.VLLM else GPUBundlingStrategy.PACK
            resources = get_service_resouces(service_name, world_size)
            placement_group = initialize_placement_group(placement_group_name, num_cpus=3+int(init_server),
                                                         num_gpus=world_size, detached=True, block=block,
                                                         gpu_bundling_strategy=gpu_bundling_strategy, resources=resources)
        else:
            placement_group = initialize_placement_group(placement_group_name, num_cpus=2+int(init_server),
                                                         num_gpus=0, detached=True, block=block)

        return placement_group

    async def _init_server_and_instance(self,
                                        instance_id: str,
                                        entrypoints_args: EntrypointsArgs,
                                        instance_args: InstanceArgs,
                                        engine_args,
                                        backend_type: BackendType,
                                        placement_group: PlacementGroup,
                                        instance_type: InstanceType = None):
        async def done_scale_up(instance_id: str, instance: ray.actor.ActorHandle,
                                instance_type: InstanceType, placement_group: PlacementGroup,
                                server: ray.actor.ActorHandle):
            try:
                instance_ready = False
                await asyncio.wait_for(instance.is_ready.remote(), timeout=INSTANCE_READY_TIMEOUT)
                instance_ready = True
                # Run server until instance is ready.
                await asyncio.wait_for(server.run.remote(self.manager, instance_id, instance), timeout=SERVER_READY_TIMEOUT)
                await self.manager.scale_up.remote(instance_id, instance, instance_type, placement_group, server)
                logger.info("Init server and instance done, instance_id: {}, instance_type: {}.".format(instance_id, instance_type))
            except asyncio.TimeoutError:
                if not instance_ready:
                    logger.error("Instance {} is not ready in {} seconds.".format(instance_id, INSTANCE_READY_TIMEOUT))
                else:
                    logger.error("Server {} is not ready in {} seconds.".format(instance_id, SERVER_READY_TIMEOUT))
                self.clear_instance_ray_resources(instance_id)
            except Exception as e: # pylint: disable=broad-except
                logger.error("Unexpected exception occurs: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))
                self.clear_instance_ray_resources(instance_id)
            finally:
                self.inflight_num_prefill_instances -= 1 if instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode_instances -= 1 if instance_type == InstanceType.DECODE else 0

        request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)
        next_instance_args = await self._get_next_instance_args(instance_args, instance_type)
        next_entrypoints_args = self._get_next_entrypoints_args(entrypoints_args)
        next_engine_args = self._get_next_engine_args(engine_args, next_instance_args.instance_type)
        instance = self._init_instance(instance_id, next_instance_args, placement_group,
                                       request_output_queue_type, backend_type, next_engine_args)
        server = self._init_server(get_server_name(instance_id), placement_group, backend_type, next_entrypoints_args, next_engine_args)

        self.inflight_num_prefill_instances += 1 if next_instance_args.instance_type == InstanceType.PREFILL else 0
        self.inflight_num_decode_instances += 1 if next_instance_args.instance_type == InstanceType.DECODE else 0
        asyncio.create_task(done_scale_up(instance_id, instance, next_instance_args.instance_type, placement_group, server))

    def _init_server(self,
                     server_name: str,
                     placement_group: PlacementGroup,
                     backend_type: BackendType,
                     entrypoints_args: EntrypointsArgs,
                     engine_args) -> APIServerActor:
        if backend_type == BackendType.BLADELLM:
            from llumnix.entrypoints.bladellm.api_server_actor import APIServerActorBladeLLM # pylint: disable=import-outside-toplevel
            # Reserve 0.5 gpu for ApiServerActor, because APIServerActor imports blade module and blade module needs cuda environments.
            api_server = APIServerActorBladeLLM.from_args(0.5, server_name, placement_group, entrypoints_args, engine_args)
        else: # BackendType.VLLM, BackendType.SIM_VLLM
            from llumnix.entrypoints.vllm.api_server_actor import APIServerActorVLLM # pylint: disable=import-outside-toplevel
            api_server = APIServerActorVLLM.from_args(0, server_name, placement_group, entrypoints_args, engine_args)

        return api_server

    def _init_instance(self,
                       instance_id: str,
                       instance_args: InstanceArgs,
                       placement_group: PlacementGroup,
                       request_output_queue_type: QueueType,
                       backend_type: BackendType,
                       engine_args
                       ) -> Tuple[str, Llumlet]:
        instance = Llumlet.from_args(
                        instance_id,
                        instance_args,
                        placement_group,
                        request_output_queue_type,
                        backend_type,
                        engine_args)

        return instance

    async def init_instances(self,
                             request_output_queue_type: QueueType,
                             backend_type: BackendType,
                             instance_args: InstanceArgs,
                             engine_args
                             ) -> Tuple[List[str], List[Llumlet]]:
        async def instance_ready_scale_up(instance_id: str, instance: ray.actor.ActorHandle,
                                          instance_type: InstanceType, placement_group: PlacementGroup):
            try:
                await asyncio.wait_for(instance.is_ready.remote(), timeout=INSTANCE_READY_TIMEOUT)
                await self.manager.scale_up.remote(instance_id, instance, instance_type, placement_group)
            except asyncio.TimeoutError:
                logger.error("Instance {} is not ready in {} seconds.".format(instance_id, INSTANCE_READY_TIMEOUT))
                self.clear_instance_ray_resources(instance_id)
            except Exception as e: # pylint: disable=broad-except
                logger.error("Unexpected exception occurs: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))
                self.clear_instance_ray_resources(instance_id)

        instance_ids: List[str] = []
        instances: List[Llumlet] = []
        for _ in range(self.manager_args.initial_instances):
            if (
                backend_type == BackendType.BLADELLM
                and self.manager_args.enable_engine_pd_disagg
                and engine_args.instance_id
            ):
                # use blade instance id as llumlet instance id
                instance_id = engine_args.instance_id
            else:
                instance_id = random_uuid()
            placement_group = self._init_placement_group(get_placement_group_name(instance_id), engine_args, backend_type,
                                                         init_server=False, block=True)
            instance = self._init_instance(instance_id, instance_args, placement_group, request_output_queue_type,
                                           backend_type, engine_args)
            instance_ids.append(instance_id)
            instances.append(instance)
            asyncio.create_task(instance_ready_scale_up(instance_id, instance, instance_args.instance_type, placement_group))

        return instance_ids, instances

    def clear_instance_ray_resources(self, instance_id: str):
        if hasattr(self, "launch_mode") and self.launch_mode == LaunchMode.GLOBAL and not kill_server(instance_id):
            logger.warning("Failed to kill server {}.".format(instance_id))
        if not kill_instance(instance_id):
            logger.warning("Failed to kill instance {}.".format(instance_id))
        if not remove_placement_group(instance_id):
            logger.warning("Failed to remove placement group {}.".format(instance_id))

    async def _get_next_instance_args(self, instance_args: InstanceArgs, instance_type: InstanceType) -> InstanceType:
        if not self.enable_port_increment:
            return instance_args

        next_instance_args: InstanceArgs = copy.deepcopy(instance_args)
        # self.port_offset will be incremented by 1 in the next _get_next_entrypoints_args call.
        next_instance_args.grpc_migration_backend_server_port += self.port_offset

        if not self.pdd_config.enable_pd_disagg and not self.pdd_config.enable_engine_pd_disagg:
            return instance_args

        # Await can still ensure make sure _init_server_and_instance is atomic due to _auto_scale_up_loop.
        cur_num_prefill_instances, cur_num_decode_instances = await self.manager.get_num_prefill_decode_instances.remote()
        next_instance_args.instance_type = self._get_next_instance_type(cur_num_prefill_instances, cur_num_decode_instances,
                                                                        self.pdd_config.pd_ratio, instance_type)

        return next_instance_args

    def _get_next_entrypoints_args(self, entrypoints_args: EntrypointsArgs) -> EntrypointsArgs:
        if not self.enable_port_increment:
            return entrypoints_args

        next_entrypoints_args = copy.deepcopy(entrypoints_args)
        next_entrypoints_args.port += self.port_offset
        next_entrypoints_args.request_output_queue_port += self.port_offset
        self.port_offset += 1
        if self.enable_port_offset_store:
            put_data_to_ray_internal_kv("manager.port_offset", self.port_offset)

        return next_entrypoints_args

    def _get_next_engine_args(self, engine_args, instance_type: str):
        if not self.load_registered_service:
            return engine_args

        new_engine_args = self.engine_args_dict[instance_type]

        return new_engine_args

    def _get_next_instance_type(self,
                                cur_num_prefill_instances: int,
                                cur_num_decode_instances: int,
                                pd_ratio: List[int],
                                instance_type: InstanceType = None) -> str:
        if instance_type:
            return instance_type

        if not self.pdd_config.enable_pd_disagg and not self.pdd_config.enable_engine_pd_disagg:
            return InstanceType.NO_CONSTRAINTS

        # There are no instances simultaneously in inflight_num_prefill_instances and cur_num_prefill_instances
        # as inflight_num will decrease before scaling up the instances. The same applies to num_decode.
        total_num_prefill_instances = self.inflight_num_prefill_instances + cur_num_prefill_instances
        total_num_decode_instances = self.inflight_num_decode_instances + cur_num_decode_instances

        if total_num_prefill_instances == 0:
            instance_type = InstanceType.PREFILL
        elif total_num_decode_instances == 0:
            instance_type = InstanceType.DECODE
        else:
            # compute distance if launch prefill or decode
            normal_distance = pd_ratio[0] - pd_ratio[1]

            base_num_ratio = min(total_num_prefill_instances//pd_ratio[0], total_num_decode_instances//pd_ratio[1])
            total_num_prefill_instances = total_num_prefill_instances - base_num_ratio * pd_ratio[0]
            total_num_decode_instances = total_num_decode_instances - base_num_ratio * pd_ratio[1]

            if total_num_prefill_instances + total_num_decode_instances == 0:
                instance_type = InstanceType.PREFILL
            else:
                distance_if_prefill = total_num_prefill_instances + 1 - total_num_decode_instances
                distance_if_decode = total_num_prefill_instances - (total_num_decode_instances + 1)
                gap_to_normal_if_prefill = abs(distance_if_prefill - normal_distance)
                gap_to_normal_if_decode = abs(distance_if_decode - normal_distance)
                instance_type = InstanceType.PREFILL if gap_to_normal_if_prefill <= gap_to_normal_if_decode \
                    else InstanceType.DECODE

        return instance_type
