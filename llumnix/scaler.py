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
import random
from typing import List, Tuple, Union, Iterable

import ray
from ray.util.placement_group import PlacementGroup
import ray.actor
import ray.exceptions

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceType
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_type import QueueType
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import (
    EntrypointsArgs,
    InstanceArgs,
    ManagerArgs,
    LaunchArgs,
    LlumnixEngineArgs,
    LlumnixEngineArgsFactory,
)
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.utils import (
    get_service_resouces,
    random_uuid,
    get_service_instance_type,
    asyncio_wait_for_with_timeout,
)
from llumnix.ray_utils import (
    initialize_placement_group,
    get_server_name,
    get_data_from_ray_internal_kv,
    put_data_to_ray_internal_kv,
    get_scaler_name,
    get_placement_group_name,
    get_placement_group_infos_by_name,
    get_placement_group_infos_by_state,
    kill_server,
    kill_instance,
    remove_placement_group,
    get_actor_names_by_name_prefix,
    SERVER_NAME_PREFIX,
    INSTANCE_NAME_PREFIX,
    actor_exists,
    get_instance_name,
    PLACEMENT_GROUP_NAME_PREFIX,
)
from llumnix.internal_config import PDDConfig
from llumnix.constants import (
    WAIT_PLACEMENT_GROUP_TIMEOUT,
    AUTO_SCALE_UP_INTERVAL,
    CHECK_DEPLOYMENT_STATES_INTERVAL,
    WATCH_DEPLOYMENT_INTERVAL,
    MAX_ACTOR_METHOD_RETRIES,
)
from llumnix import envs as llumnix_envs
from llumnix.entrypoints.utils import LaunchMode
from llumnix.constants import NUM_GPUS_BLADELLM_GPU_ACTOR
from llumnix.ray_utils import clear_gloo_backend_ray_resources
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.manager import Manager

logger = init_logger(__name__)


class Scaler:
    def __init__(self,
                 entrypoints_args: EntrypointsArgs,
                 manager_args: ManagerArgs,
                 instance_args: InstanceArgs,
                 engine_args: LlumnixEngineArgs,
                 launch_args: LaunchArgs):
        self.entrypoints_args = entrypoints_args
        self.manager_args = manager_args
        self.instance_args = instance_args
        self.engine_args = engine_args

        # TODO(s5u13b): Merge manager args and instance args.
        # manager_args
        self.max_instances = manager_args.max_instances
        self.enable_port_increment = manager_args.enable_port_increment
        self.enable_port_offset_store = manager_args.enable_port_offset_store
        self.load_registered_service = manager_args.load_registered_service
        self.load_registered_service_path = manager_args.load_registered_service_path
        self.pdd_config: PDDConfig = manager_args.create_pdd_config()

        self.scaler: Scaler = ray.get_actor(get_scaler_name(), namespace="llumnix")
        self.manager = Manager.from_args(
            entrypoints_args=entrypoints_args,
            manager_args=manager_args,
            instance_args=instance_args,
            engine_args=engine_args,
            launch_args=launch_args,
        )
        # Start scaling after manager is ready.
        ray.get(self.manager.is_ready.remote())

        # launch args
        if launch_args is not None:
            self.launch_mode: LaunchMode = launch_args.launch_mode
            self.backend_type: BackendType = launch_args.backend_type

        if self.enable_port_increment:
            self.port_offset = 0
            if self.enable_port_offset_store:
                # TODO(s5u13b): Do not use ray interval kv.
                value = get_data_from_ray_internal_kv("scaler.port_offset")
                self.port_offset = int(value)

        self.llumnix_engine_args_factory = LlumnixEngineArgsFactory(
            enable_port_increment=self.enable_port_increment,
            load_registered_service=self.load_registered_service,
            load_registered_service_path=self.load_registered_service_path,
            pdd_config=self.pdd_config,
        )

        self.inflight_num_prefill_instances = 0
        self.inflight_num_decode_instances = 0

        if hasattr(self, "launch_mode") and self.launch_mode == LaunchMode.GLOBAL:
            assert self.entrypoints_args is not None and self.engine_args is not None
            self.last_timeout_instance_id = None
            if self.pdd_config.enable_pdd_node_affinity_scheduling:
                asyncio.create_task(
                    self._auto_scale_up_loop(
                        service_name="prefill",
                        max_instances=self.max_instances,
                        interval=AUTO_SCALE_UP_INTERVAL,
                    )
                )
                asyncio.create_task(
                    self._auto_scale_up_loop(
                        service_name="decode",
                        max_instances=self.max_instances,
                        interval=AUTO_SCALE_UP_INTERVAL,
                    )
                )
            else:
                asyncio.create_task(
                    self._auto_scale_up_loop(
                        service_name="no_constraints",
                        max_instances=self.max_instances,
                        interval=AUTO_SCALE_UP_INTERVAL,
                    )
                )
            asyncio.create_task(self._check_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))
            if self.pdd_config.enable_pd_disagg:
                asyncio.create_task(self._check_pd_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))

    @classmethod
    def from_args(cls,
                  entrypoints_args: EntrypointsArgs,
                  manager_args: ManagerArgs,
                  instance_args: InstanceArgs,
                  engine_args: LlumnixEngineArgs,
                  launch_args: LaunchArgs,
                  ) -> "Scaler":
        scaler_class = ray.remote(
            num_cpus=1,
            max_restarts=-1,
            name=get_scaler_name(),
            namespace="llumnix",
            lifetime="detached"
        )(cls)
        scaler = scaler_class.remote(
            entrypoints_args,
            manager_args,
            instance_args,
            engine_args,
            launch_args
        )
        return scaler

    async def _auto_scale_up_loop(self, service_name: str, max_instances: int, interval: float) -> None:
        logger.info("Auto scale up loop starts, service name: {}".format(service_name))
        while True:
            new_pg = None
            if self.last_timeout_instance_id is not None:
                last_timeout_pg_name = get_placement_group_name(self.last_timeout_instance_id)
                last_timeout_pg_infos = get_placement_group_infos_by_name(name=last_timeout_pg_name)
                if len(last_timeout_pg_infos) > 0 and last_timeout_pg_infos[0]["state"] != "REMOVED":
                    new_instance_id = self.last_timeout_instance_id
                    # pending, created(without server and instance) or rescheduling
                    try:
                        new_pg = ray.util.get_placement_group(last_timeout_pg_name)
                    except ValueError:
                        logger.warning("Placement group {} not found.".format(
                            last_timeout_pg_name[:len(PLACEMENT_GROUP_NAME_PREFIX)]))
                # reset
                self.last_timeout_instance_id = None
            pending_pg_infos = get_placement_group_infos_by_state(state="PENDING")
            pending_pg_infos.extend(get_placement_group_infos_by_state(state="RESCHEDULING"))
            for pending_pg_info in pending_pg_infos:
                instance_id = pending_pg_info["name"].split("_")[-1]
                if new_pg is not None and instance_id == new_instance_id:
                    continue
                await self.clear_instance_ray_resources(instance_id)
            alive_pg_infos = get_placement_group_infos_by_state(state="CREATED")
            alive_pg_infos.extend(get_placement_group_infos_by_state(state="PENDING"))
            alive_pg_infos.extend(get_placement_group_infos_by_state(state="RESCHEDULING"))
            if max_instances != -1 and len(alive_pg_infos) >= max_instances:
                logger.debug("The number of alive placement groups has reached the max_instances.")
                await asyncio.sleep(interval)
                continue
            if new_pg is None:
                new_instance_id = random_uuid()
                new_pg = self._init_placement_group(
                    get_placement_group_name(new_instance_id),
                    self.engine_args,
                    init_server=True,
                    block=False,
                    service_name=service_name,
                )
            try:
                await asyncio.wait_for(new_pg.ready(), timeout=WAIT_PLACEMENT_GROUP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.debug("Waiting for new placement group {} ready timeout.".format(new_instance_id))
                # After timeout, the new placement group might be pending,
                # created(without server and instance), rescheduling.
                self.last_timeout_instance_id = new_instance_id
                await asyncio.sleep(interval)
                continue
            if service_name in ["prefill", "decode"]:
                await self._init_server_and_instance(
                    new_instance_id,
                    self.entrypoints_args,
                    self.instance_args,
                    self.engine_args,
                    new_pg,
                    instance_type=get_service_instance_type(service_name),
                )
            else:
                # If not prefill/decode service, we do not specify the instance type,
                # and the instance type is decided by _get_next_instance_type.
                await self._init_server_and_instance(new_instance_id, self.entrypoints_args, self.instance_args,
                                                     self.engine_args, new_pg)
            logger.info("Deploy server and instance to new placement group done, "
                        "instance_id: {}.".format(new_instance_id))

    async def _check_deployment_states_loop(self, interval: float) -> None:
        async def watch_instance_deployment_states(instance_id: str, server_exists: bool, instance_exists: bool):
            # Waiting for _init_server_and_instance scheduled.
            if not server_exists and not instance_exists:
                await asyncio.sleep(1.0)
            # Server is initialized after instance is ready, so waiting for instance ready first.
            if not server_exists and instance_exists:
                instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
                try:
                    await asyncio_wait_for_with_timeout(instance.is_ready.remote())
                # Instance exception could be handled by manager, so check states loop omits exception case here.
                # pylint: disable=bare-except
                except:
                    return
            logger.info("watch instance {} deployment states, server_exists: {}, instance_exists: {}".format(
                instance_id, server_exists, instance_exists))
            await asyncio.sleep(WATCH_DEPLOYMENT_INTERVAL)
            pg_created, server_exists, instance_exists = self._get_instance_deployment_states(instance_id)
            if pg_created and (not server_exists or not instance_exists):
                logger.warning("Instance {} deployment states incorrect, states: (pg {}, server {}, instance {})"
                               .format(instance_id, pg_created, server_exists, instance_exists))
                await asyncio_wait_for_with_timeout(self.manager.scale_down.remote(instance_id))

        while True:
            try:
                # Not check right after scaler initialized, so sleep at the beginning.
                await asyncio.sleep(interval)
                curr_pgs, curr_servers, curr_instances = self._get_cluster_deployment_states()
                assert len(curr_pgs) >= max(len(curr_servers), len(curr_instances))
                tasks = []
                for instance_id in curr_pgs:
                    server_exists = instance_id in curr_servers
                    instance_exists = instance_id in curr_instances
                    if not server_exists or not instance_exists:
                        tasks.append(
                            asyncio.create_task(
                                watch_instance_deployment_states(instance_id, server_exists, instance_exists)
                            )
                        )
                await asyncio.gather(*tasks, return_exceptions=True)
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Scaler get error in _check_deployment_states_loop, scaler keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

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
            except Exception:
                logger.critical(
                    "Scaler get error in _check_pd_deployment_states_loop, scaler keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    async def _check_pd_deployment_states(self) -> str:
        prefill_instance_id_set, decode_instance_id_set = await asyncio_wait_for_with_timeout(
            self.manager.get_prefill_decode_instance_id_set.remote()
        )
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
            await asyncio_wait_for_with_timeout(self.manager.scale_down.remote(scale_down_instance_id))

        return scale_down_instance_id

    def _get_cluster_deployment_states(self) -> Tuple[List[str], List[str], List[str]]:
        curr_pgs: List[str] = []
        curr_servers: List[str] = []
        curr_instances: List[str] = []

        created_pg_infos = get_placement_group_infos_by_state(state="CREATED")
        for created_pg_info in created_pg_infos:
            instance_id = created_pg_info["name"].split("_")[-1]
            curr_pgs.append(instance_id)

        curr_server_names = get_actor_names_by_name_prefix(name_prefix=SERVER_NAME_PREFIX)
        for curr_server_name in curr_server_names:
            instance_id = curr_server_name.split("_")[-1]
            curr_servers.append(instance_id)

        curr_instance_names = get_actor_names_by_name_prefix(name_prefix=INSTANCE_NAME_PREFIX)
        for curr_instance_name in curr_instance_names:
            instance_id = curr_instance_name.split("_")[-1]
            curr_instances.append(instance_id)

        return curr_pgs, curr_servers, curr_instances

    def _get_instance_deployment_states(self, instance_id: str):
        pg_infos = get_placement_group_infos_by_name(name=get_placement_group_name(instance_id))
        pg_created = len(pg_infos) == 1 and pg_infos[0]["state"] == "CREATED"
        server_exists = actor_exists(get_server_name(instance_id))
        instance_exists = actor_exists(get_instance_name(instance_id))

        return pg_created, server_exists, instance_exists

    def _init_placement_group(self,
                              placement_group_name: str,
                              engine_args: LlumnixEngineArgs,
                              init_server: bool = False,
                              block: bool = True,
                              node_id: str = None,
                              service_name: str = None
                              ) -> PlacementGroup:
        backend_type = engine_args.backend_type
        # num_cpus=2+(0/1), for Llumlet + ActorOutputMediator + (ApiServerActor)
        if not BackendType.is_sim_backend(backend_type):
            # num_gpus=world_size, for world_size Workers
            world_size = engine_args.get_world_size()
            resources = get_service_resouces(service_name, world_size)
            placement_group = initialize_placement_group(
                placement_group_name,
                num_cpus=2+int(init_server),
                num_gpus=world_size,
                detached=True,
                block=block,
                node_id=node_id,
                resources=resources,
            )
        else:
            placement_group = initialize_placement_group(
                placement_group_name,
                num_cpus=2+int(init_server),
                num_gpus=0,
                detached=True,
                block=block,
                node_id=node_id,
            )

        return placement_group

    async def _init_server_and_instance(self,
                                        instance_id: str,
                                        entrypoints_args: EntrypointsArgs,
                                        instance_args: InstanceArgs,
                                        engine_args: LlumnixEngineArgs,
                                        placement_group: PlacementGroup,
                                        instance_type: InstanceType = None):
        async def done_scale_up(instance_id: str, instance: Llumlet,
                                instance_type: InstanceType, next_entrypoints_args: EntrypointsArgs,
                                next_engine_args: LlumnixEngineArgs):
            try:
                instance_ready = False
                await asyncio.wait_for(instance.is_ready.remote(), timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT))
                instance_ready = True
                # Initialize server after instance is ready.
                server = self._init_server(
                    instance_id,
                    placement_group,
                    backend_type,
                    next_entrypoints_args,
                    next_engine_args,
                    self.scaler,
                    self.manager,
                    instance,
                )
                await asyncio.wait_for(server.is_ready.remote(), timeout=float(llumnix_envs.SERVER_READY_TIMEOUT))
                await asyncio_wait_for_with_timeout(
                    self.manager.scale_up.remote(instance_id, instance, instance_type, placement_group, server)
                )
                logger.info("Init server and instance done, instance_id: {}, instance_type: {}.".format(instance_id, instance_type))
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up instance {}, instance is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    if not instance_ready:
                        logger.error("Instance {} is not ready in {} seconds.".format(instance_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)))
                    else:
                        logger.error("Server {} is not ready in {} seconds.".format(instance_id, float(llumnix_envs.SERVER_READY_TIMEOUT)))
                else:
                    logger.exception("Error in scaler done_scale_up (instance_id: {})".format(instance_id))
                await self.clear_instance_ray_resources(instance_id)
            finally:
                self.inflight_num_prefill_instances -= 1 if instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode_instances -= 1 if instance_type == InstanceType.DECODE else 0

        backend_type = engine_args.backend_type
        request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)
        next_instance_args = await self._get_next_instance_args(instance_args, instance_type)
        next_entrypoints_args = self._get_next_entrypoints_args(entrypoints_args)
        next_engine_args = self.llumnix_engine_args_factory.gen_next_engine_args(
            backend_type=backend_type,
            current_engine_args=engine_args,
            instance_type=next_instance_args.instance_type,
        )
        instance = self._init_instance(
            instance_id,
            next_instance_args,
            placement_group,
            request_output_queue_type,
            next_engine_args,
        )

        self.inflight_num_prefill_instances += 1 if next_instance_args.instance_type == InstanceType.PREFILL else 0
        self.inflight_num_decode_instances += 1 if next_instance_args.instance_type == InstanceType.DECODE else 0
        asyncio.create_task(
            done_scale_up(
                instance_id,
                instance,
                next_instance_args.instance_type,
                next_entrypoints_args,
                next_engine_args,
            )
        )

    def _init_server(self,
                     instance_id: str,
                     placement_group: PlacementGroup,
                     backend_type: BackendType,
                     entrypoints_args: EntrypointsArgs,
                     engine_args,
                     scaler: ray.actor.ActorHandle,
                     manager: ray.actor.ActorHandle,
                     instance: ray.actor.ActorHandle) -> APIServerActor:
        if backend_type == BackendType.BLADELLM:
            from llumnix.entrypoints.bladellm.api_server_actor import APIServerActorBladeLLM # pylint: disable=import-outside-toplevel
            api_server = APIServerActorBladeLLM.from_args(
                NUM_GPUS_BLADELLM_GPU_ACTOR,
                instance_id,
                placement_group,
                entrypoints_args,
                engine_args,
                scaler,
                manager,
                instance,
            )
        elif backend_type == BackendType.VLLM_V1:
            from llumnix.entrypoints.vllm_v1.api_server_actor import APIServerActorVLLMV1
            api_server = APIServerActorVLLMV1.from_args(
                0,
                instance_id,
                placement_group,
                entrypoints_args,
                engine_args,
                scaler,
                manager,
                instance,
            )
        else: # BackendType.VLLM, BackendType.SIM_VLLM
            from llumnix.entrypoints.vllm.api_server_actor import APIServerActorVLLM # pylint: disable=import-outside-toplevel
            api_server = APIServerActorVLLM.from_args(
                0,
                instance_id,
                placement_group,
                entrypoints_args,
                engine_args,
                scaler,
                manager,
                instance,
            )

        return api_server

    def _init_instance(self,
                       instance_id: str,
                       instance_args: InstanceArgs,
                       placement_group: PlacementGroup,
                       request_output_queue_type: QueueType,
                       engine_args: LlumnixEngineArgs
                       ) -> Tuple[str, Llumlet]:
        instance = Llumlet.from_args(
            instance_id,
            instance_args,
            placement_group,
            request_output_queue_type,
            engine_args,
        )

        return instance

    async def init_instances(self,
                             request_output_queue_type: QueueType,
                             instance_args: InstanceArgs,
                             engine_args: LlumnixEngineArgs,
                             node_id: str
                             ) -> Tuple[List[str], List[Llumlet]]:
        async def instance_ready_scale_up(instance_id: str, instance: Llumlet,
                                          instance_type: InstanceType, placement_group: PlacementGroup):
            try:
                await asyncio.wait_for(instance.is_ready.remote(), timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT))
                await asyncio_wait_for_with_timeout(
                    self.manager.scale_up.remote(instance_id, instance, instance_type, placement_group)
                )
            except asyncio.TimeoutError:
                logger.error("Instance {} is not ready in {} seconds.".format(instance_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)))
                await self.clear_instance_ray_resources(instance_id)
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up instance {}, instance is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error(
                        "Failed to scale up instance {}, "
                        "instance is not ready in {} seconds.".format(
                            instance_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                        )
                    )
                else:
                    logger.exception("Error in scaler instance_ready_scale_up (instance_id: {})".format(instance_id))
                await self.clear_instance_ray_resources(instance_id)

        instance_ids: List[str] = []
        instances: List[Llumlet] = []
        backend_type = engine_args.backend_type
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
            placement_group = self._init_placement_group(
                get_placement_group_name(instance_id),
                engine_args,
                init_server=False,
                block=True,
                node_id=node_id,
            )
            if placement_group is None:
                logger.warning("Failed to initialize placement group for instance {}, "
                               "the remaining resources of node {} might be not enough, "
                               "stop initializing instances.".format(instance_id, node_id))
                return instance_ids, instances
            instance = self._init_instance(
                instance_id,
                instance_args,
                placement_group,
                request_output_queue_type,
                engine_args,
            )
            instance_ids.append(instance_id)
            instances.append(instance)
            asyncio.create_task(
                instance_ready_scale_up(
                    instance_id,
                    instance,
                    instance_args.instance_type,
                    placement_group
                )
            )

        return instance_ids, instances

    def init_request_output_queue_server(self, ip: str, queue_type: QueueType) -> QueueServerBase:
        return init_request_output_queue_server(ip, queue_type)

    def is_ready(self) -> bool:
        return True

    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    async def clear_instance_ray_resources(self, instance_id: Union[str, Iterable[str]]):
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            # There could be multiple clear_instance_ray_resources calls for one error instance,
            # so the kill operations could be failed if it is not the first attempt to kill.
            if hasattr(self, "launch_mode") and self.launch_mode == LaunchMode.GLOBAL:
                await kill_server(ins_id)
            await kill_instance(ins_id)
            remove_placement_group(ins_id)

    async def _get_next_instance_args(self, instance_args: InstanceArgs, instance_type: InstanceType) -> InstanceArgs:
        if (
            not self.enable_port_increment
            and not self.pdd_config.enable_pd_disagg
            and not self.pdd_config.enable_engine_pd_disagg
        ):
            return instance_args

        next_instance_args: InstanceArgs = copy.deepcopy(instance_args)

        if self.pdd_config.enable_pd_disagg or self.pdd_config.enable_engine_pd_disagg:
            # Await can still ensure make sure _init_server_and_instance is atomic due to _auto_scale_up_loop.
            cur_num_prefill_instances, cur_num_decode_instances = await asyncio_wait_for_with_timeout(
                self.manager.get_num_prefill_decode_instances.remote()
            )
            next_instance_args.instance_type = self._get_next_instance_type(
                cur_num_prefill_instances, cur_num_decode_instances, self.pdd_config.pd_ratio, instance_type,)

        return next_instance_args

    def _get_next_entrypoints_args(self, entrypoints_args: EntrypointsArgs) -> EntrypointsArgs:
        if not self.enable_port_increment:
            return entrypoints_args

        next_entrypoints_args = copy.deepcopy(entrypoints_args)
        next_entrypoints_args.port += self.port_offset
        self.port_offset += 1
        if self.enable_port_offset_store:
            put_data_to_ray_internal_kv("scaler.port_offset", self.port_offset)

        return next_entrypoints_args

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
                instance_type = InstanceType.PREFILL if gap_to_normal_if_prefill <= gap_to_normal_if_decode else InstanceType.DECODE

        return instance_type

    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    def clear_gloo_backend_ray_resources(self):
        clear_gloo_backend_ray_resources()
