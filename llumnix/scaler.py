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

import os
import asyncio
import random
from typing import List, Tuple, Union, Iterable, Dict, Any

import ray
from ray.util.placement_group import PlacementGroup
import ray.actor
import ray.exceptions

from llumnix.logging.logger import init_logger
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_type import QueueType
from llumnix.arg_utils import (
    EntrypointsArgs,
    InstanceArgs,
    ManagerArgs,
    LaunchArgs,
    LlumnixEngineArgs,
    LlumnixEngineArgsFactory,
    load_engine_args,
    load_llumlet_env_vars,
)
from llumnix.utils import (
    get_service_resouces,
    random_uuid,
    get_service_instance_type,
    asyncio_wait_for_ray_remote_call_with_timeout,
    BackendType,
    LaunchMode,
    InstanceType,
    run_coroutine_in_new_thread,
)
from llumnix.ray_utils import (
    initialize_placement_group,
    get_data_from_ray_internal_kv,
    put_data_to_ray_internal_kv,
    get_placement_group_name,
    list_placement_group_infos_by_name,
    list_placement_group_infos_by_state,
    remove_placement_group,
    actor_exists,
    get_llumnix_actor_name,
    get_llumnix_actor_handle,
    kill_dp_manager,
    connect_to_actors_with_instance_type,
    kill_instance,
    check_actors_health,
    get_placement_group_unit_id,
    LlumnixActor,
    list_actor_names_by_actor_type,
    get_llumnix_actor_id,
    BundlingStrategy,
)
from llumnix.internal_config import PDDConfig
from llumnix.constants import (
    WAIT_PLACEMENT_GROUP_TIMEOUT,
    AUTO_SCALE_UP_INTERVAL,
    CHECK_DEPLOYMENT_STATES_INTERVAL,
    WATCH_DEPLOYMENT_INTERVAL,
    HEARTBEAT_INTERVAL,
)
from llumnix import envs as llumnix_envs
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.manager import Manager
from llumnix.dp_manager import DPManager

logger = init_logger(__name__)


class Scaler:
    def __init__(
        self,
        entrypoints_args: EntrypointsArgs,
        manager_args: ManagerArgs,
        instance_args: InstanceArgs,
        engine_args: LlumnixEngineArgs,
        launch_args: LaunchArgs,
    ):
        self.entrypoints_args = entrypoints_args
        self.manager_args = manager_args
        self.instance_args = instance_args
        self.engine_args = engine_args

        # TODO(s5u13b): Merge manager args and instance args.
        # manager_args
        self.max_units = manager_args.max_units
        self.enable_port_increment = manager_args.enable_port_increment
        self.enable_port_offset_store = manager_args.enable_port_offset_store
        self.load_registered_service = manager_args.load_registered_service
        self.load_registered_service_path = manager_args.load_registered_service_path
        self.pdd_config: PDDConfig = manager_args.create_pdd_config()

        # launch args
        self.launch_mode: LaunchMode = launch_args.launch_mode
        self.backend_type: BackendType = launch_args.backend_type

        self.scaler: Scaler = get_llumnix_actor_handle(LlumnixActor.SCALER)

        try:
            self.manager = Manager.from_args(
                entrypoints_args=entrypoints_args,
                manager_args=manager_args,
                instance_args=instance_args,
                engine_args=engine_args,
                launch_args=launch_args,
            )
            logger.info("Init Manager actor.")
        except ValueError:
            logger.info("Manager actor already exists, get existed manager actor.")
            self.manager = get_llumnix_actor_handle(LlumnixActor.MANAGER)
            ray.get(self.manager.is_ready.remote())

        # scale up utility
        engine_args_factory_cls = LlumnixEngineArgsFactory._get_engine_args_factory_cls(self.backend_type)
        self.llumnix_engine_args_factory: LlumnixEngineArgsFactory = engine_args_factory_cls(
            enable_port_increment=self.enable_port_increment,
            load_registered_service=self.load_registered_service,
            load_registered_service_path=self.load_registered_service_path,
            pdd_config=self.pdd_config,
        )
        if self.load_registered_service:
            self.engine_args_dict: Dict[str, LlumnixEngineArgs] = {}
            self.env_vars_dict: Dict[str, Dict[str, Any]] = {}
            if not self.enable_pd:
                instance_type_list = ["neutral"]
            else:
                instance_type_list = ["prefill", "decode"]
            for instance_type in instance_type_list:
                self.engine_args_dict[instance_type] = load_engine_args(
                    instance_type, self.load_registered_service_path
                )

                self.env_vars_dict[instance_type] = load_llumlet_env_vars(
                    instance_type, self.load_registered_service_path
                )

        # scale up states
        self.port_offset = 0
        if self.enable_port_increment:
            if self.enable_port_offset_store:
                # TODO(s5u13b): Do not use ray interval kv.
                value = get_data_from_ray_internal_kv("scaler.port_offset")
                self.port_offset = int(value)
        if self.backend_type == BackendType.VLLM_V1:
            # Mantain a monotonically increasing `client_index` for vLLM V1 APIServer.
            # It will be passed to APIServerActor through EntrypointsArgs.
            current_client_index = -1
            try:
                value = int(get_data_from_ray_internal_kv("scaler.client_index")) + 1
                current_client_index = value
                put_data_to_ray_internal_kv("scaler.client_index", value)
            except AssertionError:
                current_client_index = 0
                put_data_to_ray_internal_kv("scaler.client_index", 0)
            except Exception: # pylint: disable=broad-except
                logger.exception("Scaler init client_index error.")
            finally:
                self.client_index = current_client_index
                if self.client_index == -1:
                    logger.error("Failed to get client_index from ray internal kv: {}".format(self.client_index))

        # cached dynamic states
        self.dp_managers: Dict[str, ray.actor.ActorHandle] = {}
        self.instance_types: Dict[str, InstanceType] = {}
        self.prefill_unit_id_set = set()
        self.decode_unit_id_set = set()
        self.inflight_num_prefill_units = 0
        self.inflight_num_decode_units = 0

        run_coroutine_in_new_thread(self._connect_to_dp_managers(), blocking=True)

        if self.launch_mode == LaunchMode.GLOBAL:
            self.last_timeout_unit_id = None
            if self.pdd_config.enable_pdd_node_affinity_scheduling:
                asyncio.create_task(
                    self._auto_scale_up_loop(
                        service_name="prefill",
                        max_units=self.max_units,
                        interval=AUTO_SCALE_UP_INTERVAL,
                    )
                )
                asyncio.create_task(
                    self._auto_scale_up_loop(
                        service_name="decode",
                        max_units=self.max_units,
                        interval=AUTO_SCALE_UP_INTERVAL,
                    )
                )
            else:
                asyncio.create_task(
                    self._auto_scale_up_loop(
                        service_name="neutral",
                        max_units=self.max_units,
                        interval=AUTO_SCALE_UP_INTERVAL,
                    )
                )
            asyncio.create_task(self._check_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))
            if self.pdd_config.enable_pd_disagg:
                asyncio.create_task(self._check_pd_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))
            asyncio.create_task(self._heartbeat_loop(HEARTBEAT_INTERVAL))

    def is_ready(self) -> bool:
        return True

    @classmethod
    def from_args(
        cls,
        entrypoints_args: EntrypointsArgs,
        manager_args: ManagerArgs,
        instance_args: InstanceArgs,
        engine_args: LlumnixEngineArgs,
        launch_args: LaunchArgs,
    ) -> "Scaler":
        scaler_class = ray.remote(
            num_cpus=1,
            max_restarts=-1,
            name=get_llumnix_actor_name(LlumnixActor.SCALER),
            namespace="llumnix",
            lifetime="detached"
        )(cls)
        scaler = scaler_class.remote(
            entrypoints_args,
            manager_args,
            instance_args,
            engine_args,
            launch_args,
        )

        return scaler

    async def _auto_scale_up_loop(self, service_name: str, max_units: int, interval: float) -> None:
        """Auto scale up loop, used in global launch mode."""

        logger.info("Auto scale up loop starts, service name: {}".format(service_name))
        while True:
            try:
                new_pg = None
                instance_type = get_service_instance_type(service_name) if service_name in ["prefill", "decode"] else None
                next_instance_type = self._get_next_instance_type() if instance_type is None else instance_type
                engine_args = self.engine_args
                env_vars = dict(os.environ)
                if self.load_registered_service:
                    engine_args = self.engine_args_dict[next_instance_type]
                    env_vars = self.env_vars_dict[next_instance_type]

                if self.last_timeout_unit_id is not None:
                    last_timeout_pg_name = get_placement_group_name(self.last_timeout_unit_id)
                    last_timeout_pg_infos = list_placement_group_infos_by_name(name=last_timeout_pg_name)
                    if len(last_timeout_pg_infos) > 0 and last_timeout_pg_infos[0]["state"] != "REMOVED":
                        new_unit_id = self.last_timeout_unit_id
                        # pending, created(without server and instance) or rescheduling
                        try:
                            new_pg = ray.util.get_placement_group(last_timeout_pg_name)
                        except ValueError:
                            logger.warning(
                                "Placement group {} not found.".format(
                                    get_placement_group_unit_id(last_timeout_pg_name)
                                )
                            )
                    # reset
                    self.last_timeout_unit_id = None
                pending_pg_infos = list_placement_group_infos_by_state(state="PENDING")
                pending_pg_infos.extend(list_placement_group_infos_by_state(state="RESCHEDULING"))
                for pending_pg_info in pending_pg_infos:
                    unit_id = pending_pg_info["name"].split("_")[-1]
                    if new_pg is not None and unit_id == new_unit_id:
                        continue
                    await self._clear_unit_ray_resources(unit_id)
                alive_pg_infos = list_placement_group_infos_by_state(state="CREATED")
                alive_pg_infos.extend(list_placement_group_infos_by_state(state="PENDING"))
                alive_pg_infos.extend(list_placement_group_infos_by_state(state="RESCHEDULING"))
                if max_units != -1 and len(alive_pg_infos) >= max_units:
                    logger.debug("The number of alive placement groups has reached the max_units.")
                    await asyncio.sleep(interval)
                    continue
                if new_pg is None:
                    new_unit_id = random_uuid()
                    new_pg = self._init_placement_group(
                        get_placement_group_name(new_unit_id),
                        engine_args,
                        init_server=True,
                        block=False,
                        service_name=service_name,
                    )
                try:
                    await asyncio.wait_for(new_pg.ready(), timeout=WAIT_PLACEMENT_GROUP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug("Waiting for new placement group {} ready timeout.".format(new_unit_id))
                    # After timeout, the new placement group might be pending,
                    # created(without server and instance), rescheduling.
                    self.last_timeout_unit_id = new_unit_id
                    await asyncio.sleep(interval)
                    continue
                self._init_dp_manager(
                    new_unit_id,
                    self.entrypoints_args,
                    self.instance_args,
                    engine_args,
                    env_vars,
                    new_pg,
                    next_instance_type,
                    self.backend_type,
                )
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Scaler get error in _auto_scale_up_loop, scaler keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    def _init_placement_group(
        self,
        placement_group_name: str,
        engine_args: LlumnixEngineArgs,
        init_server: bool = False,
        block: bool = True,
        node_id: str = None,
        service_name: str = None,
    ) -> PlacementGroup:
        backend_type = engine_args.backend_type
        if backend_type == BackendType.VLLM_V1:
            dp_size = engine_args.get_dp_size()
            world_size = engine_args.get_world_size()
            # num_cpus: [lumlet + ActorOutputMediator + (ApiServerActor)] * dp_size + dp_manager
            # num_gpus: world_size * dp_size
            placement_group = initialize_placement_group(
                placement_group_name,
                num_cpus=(2+int(init_server)) * dp_size + 1,
                num_gpus=world_size * dp_size,
                dp_size=dp_size,
                detached=True,
                block=block,
                node_id=node_id,
                bundling_strategy=BundlingStrategy.INSTANCE,
            )
        elif not BackendType.is_sim_backend(backend_type):
            # num_cpus=2+(0/1), for Llumlet + ActorOutputForwarder + (ApiServerActor) + dp_manager
            # num_gpus=world_size, for world_size Workers
            world_size = engine_args.get_world_size()
            resources = get_service_resouces(service_name, world_size)
            placement_group = initialize_placement_group(
                placement_group_name,
                num_cpus=2+int(init_server) + 1,
                num_gpus=world_size,
                detached=True,
                block=block,
                node_id=node_id,
                resources=resources,
                bundling_strategy=BundlingStrategy.WORKER,
            )
        else:
            placement_group = initialize_placement_group(
                placement_group_name,
                num_cpus=2+int(init_server) + 1,
                num_gpus=0,
                detached=True,
                block=block,
                node_id=node_id,
                bundling_strategy=BundlingStrategy.WORKER,
            )

        return placement_group

    def _init_dp_manager(
        self,
        unit_id: str,
        entrypoints_args: EntrypointsArgs,
        instance_args: InstanceArgs,
        engine_args: LlumnixEngineArgs,
        env_vars: Dict[str, Any],
        placement_group: PlacementGroup,
        next_instance_type: InstanceType,
        backend_type: BackendType,
    ) -> None:
        async def dp_manager_ready_scale_up(unit_id: str, dp_manager: DPManager, instance_type: InstanceType):
            try:
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    dp_manager.is_ready, timeout=float(llumnix_envs.UNIT_READY_TIMEOUT)
                )
                self._scale_up(unit_id, dp_manager, instance_type)
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up dp manager {}, dp manager is dead.".format(unit_id))
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error(
                        "Failed to scale up dp manager {}, dp manager is not ready in {} seconds.".format(
                            unit_id, float(llumnix_envs.UNIT_READY_TIMEOUT)
                        )
                    )
                else:
                    logger.exception("Error in scaler dp_manager_ready_scale_up (unit_id: {})".format(unit_id))
                await self._clear_unit_ray_resources(unit_id)
            finally:
                self.inflight_num_prefill_units -= 1 if instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode_units -= 1 if instance_type == InstanceType.DECODE else 0

        dp_size = engine_args.get_dp_size()
        dp_size_local = engine_args.get_dp_size_local()

        entrypoints_args_list: List[EntrypointsArgs] = []
        instance_id_list: List[str] = []
        instance_args_list: List[InstanceArgs] = []
        engine_args_list: List[LlumnixEngineArgs] = []
        for rank in range(dp_size):
            port_offset = self.port_offset + rank
            client_index = self.client_index + rank if backend_type == BackendType.VLLM_V1 else None
            entrypoints_args_list.append(
                EntrypointsArgs.get_next_entrypoints_args(
                    entrypoints_args, self.enable_port_increment, port_offset, client_index
                )
            )
            instance_id = random_uuid()
            instance_id_list.append("{}_{}".format(unit_id, instance_id))
            instance_args_list.append(InstanceArgs.get_next_instance_args(instance_args, next_instance_type))
            engine_args_list.append(
                self.llumnix_engine_args_factory.gen_next_engine_args(
                    engine_args, instance_args_list[-1], port_offset, unit_id
                )
            )

        if self.enable_port_increment:
            self.port_offset += dp_size
            if self.enable_port_offset_store:
                put_data_to_ray_internal_kv("scaler.port_offset", self.port_offset)
        if backend_type == BackendType.VLLM_V1:
            self.client_index += dp_size
            put_data_to_ray_internal_kv("scaler.client_index", self.client_index)

        dp_manager = DPManager.from_args(
            unit_id,
            next_instance_type,
            dp_size,
            dp_size_local,
            instance_id_list,
            entrypoints_args_list,
            instance_args_list,
            engine_args_list,
            env_vars,
            placement_group,
            backend_type,
            self.scaler,
            self.manager,
        )

        self.inflight_num_prefill_units += 1 if next_instance_type == InstanceType.PREFILL else 0
        self.inflight_num_decode_units += 1 if next_instance_type == InstanceType.DECODE else 0

        asyncio.create_task(dp_manager_ready_scale_up(unit_id, dp_manager, next_instance_type))

        logger.info("Deploy dp manager to new placement group done, unit_id: {}.".format(unit_id))

        return dp_manager

    async def _check_deployment_states_loop(self, interval: float) -> None:
        "Check if placement groups and dp managers in the cluster keep 1:1 deployment."

        async def watch_unit_deployment_states(unit_id: str, dp_manager_exists: bool):
            try:
                dp_manager: DPManager = get_llumnix_actor_handle(LlumnixActor.DP_MANAGER, unit_id)
            except ValueError:
                logger.warning("Dp manager {} not found.".format(unit_id))
                dp_manager_exists = False
            if dp_manager_exists:
                try:
                    await asyncio_wait_for_ray_remote_call_with_timeout(
                        dp_manager.is_ready, timeout=llumnix_envs.UNIT_READY_TIMEOUT
                    )
                # Check states loop does not handle dp manager exception.
                # pylint: disable=bare-except
                except:
                    pass
            logger.info(
                "Watch unit {} deployment states, dp_manager_exists: {}".format(unit_id, dp_manager_exists)
            )
            await asyncio.sleep(WATCH_DEPLOYMENT_INTERVAL)
            pg_created, dp_manager_exists = self._get_unit_deployment_states(unit_id)
            if pg_created and not dp_manager_exists:
                logger.warning(
                    "Unit {} deployment states incorrect, states: (pg: {}, dp manager: {})".format(
                        unit_id, pg_created, dp_manager_exists
                    )
                )
                await self._scale_down(unit_id)

        while True:
            try:
                # Not check right after scaler initialized, so sleep at the beginning.
                await asyncio.sleep(interval)
                curr_pg_uids, curr_dp_manager_uids = self._get_cluster_deployment_states()
                assert len(curr_pg_uids) >= len(curr_dp_manager_uids)
                tasks = []
                for uid in curr_pg_uids:
                    dp_manager_exists = uid in curr_dp_manager_uids
                    if not dp_manager_exists:
                        tasks.append(
                            asyncio.create_task(
                                watch_unit_deployment_states(uid, dp_manager_exists)
                            )
                        )
                await asyncio.gather(*tasks, return_exceptions=True)
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Scaler get error in _check_deployment_states_loop, scaler keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    def _get_cluster_deployment_states(self) -> Tuple[List[str], List[str]]:
        curr_pg_uids: List[str] = []
        curr_dp_manager_uids: List[str] = []

        created_pg_infos = list_placement_group_infos_by_state(state="CREATED")
        for created_pg_info in created_pg_infos:
            unit_id = get_placement_group_unit_id(created_pg_info["name"])
            curr_pg_uids.append(unit_id)

        curr_dp_manager_names = list_actor_names_by_actor_type(LlumnixActor.DP_MANAGER)
        for curr_dp_manager_name in curr_dp_manager_names:
            unit_id = get_llumnix_actor_id(LlumnixActor.DP_MANAGER, curr_dp_manager_name)
            curr_dp_manager_uids.append(unit_id)

        return curr_pg_uids, curr_dp_manager_uids

    def _get_unit_deployment_states(self, unit_id: str):
        pg_infos = list_placement_group_infos_by_name(get_placement_group_name(unit_id))
        pg_created = len(pg_infos) == 1 and pg_infos[0]["state"] == "CREATED"
        dp_manager_exists = actor_exists(get_llumnix_actor_name(LlumnixActor.DP_MANAGER, unit_id))

        return pg_created, dp_manager_exists

    async def _check_pd_deployment_states_loop(self, interval: float) -> None:
        """
        Currently, only one naive prefill-decode disaggregation deployment states check policy is implemented,
        which prevents all units in the cluster are prefill units or decode units.
        """
        # TODO(KuilongCui): Deploy prefill and decode units strictly according to the pd_ratio.
        previous_penging_pg_names = None
        while True:
            try:
                pending_pg_infos = list_placement_group_infos_by_state(state="PENDING")
                rescheduling_pg_infos = list_placement_group_infos_by_state(state="RESCHEDULING")
                all_penging_pg_names = [pg["name"] for pg in pending_pg_infos]
                if previous_penging_pg_names and len(rescheduling_pg_infos) == 0 :
                    new_pending_pg_infos = list_placement_group_infos_by_state(state="PENDING")
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
        cur_num_prefill_units = len(self.prefill_unit_id_set)
        cur_num_decode_units = len(self.decode_unit_id_set)
        scale_down_unit_id = None
        if cur_num_prefill_units == 0 and cur_num_decode_units > 0:
            scale_down_unit_id = random.choice(list(self.decode_unit_id_set))
            logger.info(
                "Check pd deployment, pd_ratio: {}, cur_num_prefill_units: {}, cur_num_decode_units: {}, "
                "there are only decode units in cluster, scale down decode unit {}".format(
                    self.pdd_config.pd_ratio, cur_num_prefill_units, cur_num_decode_units, scale_down_unit_id
                )
            )

        if cur_num_decode_units == 0 and cur_num_prefill_units > 0:
            scale_down_unit_id = random.choice(list(self.prefill_unit_id_set))
            logger.info(
                "Check pd deployment, pd_ratio: {}, cur_num_prefill_units: {}, cur_num_decode_units: {}, "
                "there are only prefill units in cluster, scale down prefill unit {}".format(
                    self.pdd_config.pd_ratio, cur_num_prefill_units, cur_num_decode_units, scale_down_unit_id
                )
            )

        return scale_down_unit_id

    async def _heartbeat_loop(self, interval: float) -> None:
        """Watch all cached dp manager actors health."""

        while True:
            try:
                await asyncio.sleep(interval)
                dead_unit_ids = await check_actors_health(self.dp_managers)
                if len(dead_unit_ids) > 0:
                    await self._scale_down(dead_unit_ids)
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Scaler get error in _heartbeat_loop, scaler keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    async def init_instances(
        self,
        request_output_queue_type: QueueType,
        instance_args: InstanceArgs,
        engine_args: LlumnixEngineArgs,
        node_id: str,
    ) -> Tuple[List[str], List[Llumlet]]:
        async def ready_scale_up(instance_id: str, instance: Llumlet, instance_type: InstanceType):
            try:
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    instance.is_ready, timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                )
                self._scale_up(instance_id, instance, instance_type)
                await asyncio_wait_for_ray_remote_call_with_timeout(self.manager.scale_up, instance_id, instance, instance_type)
                logger.info("Init server and instance done, instance_id: {}, instance_type: {}.".format(instance_id, instance_type))
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up instance {}, instance is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error(
                        "Failed to scale up instance {}, instance is not ready in {} seconds.".format(
                            instance_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                        )
                    )
                else:
                    logger.exception("Error in scaler ready_scale_up (instance_id: {})".format(instance_id))
                await self._clear_instance_ray_resources(instance_id)

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
                logger.warning(
                    "Failed to initialize placement group for instance {}, "
                    "the remaining resources of node {} might be not enough, "
                    "stop initializing instances.".format(
                        instance_id, node_id
                    )
                )
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
                ready_scale_up(
                    instance_id,
                    instance,
                    instance_args.instance_type,
                )
            )

        return instance_ids, instances

    def _init_instance(
        self,
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

    def init_request_output_queue_server(self, ip: str, queue_type: QueueType) -> QueueServerBase:
        return init_request_output_queue_server(ip, queue_type)

    def _scale_up(
        self,
        unit_id: Union[str, Iterable[str]],
        dp_manager_actor_handle: Union[DPManager, List[DPManager]],
        instance_type: Union[InstanceType, List[InstanceType]]
    ) -> int:
        if isinstance(unit_id, str):
            unit_id = [unit_id,]
            dp_manager_actor_handle = [dp_manager_actor_handle,]
            instance_type = [instance_type,]

        unit_ids: List[str] = list(unit_id)
        dp_manager_handles: List[DPManager] = list(dp_manager_actor_handle)
        instance_types: List[InstanceType] = list(instance_type)

        for uid, dp_manager, ins_type in zip(unit_ids, dp_manager_handles, instance_types):
            self.dp_managers[uid] = dp_manager
            self.instance_types[uid] = ins_type
            if ins_type == InstanceType.PREFILL:
                self.prefill_unit_id_set.add(uid)
            elif ins_type == InstanceType.DECODE:
                self.decode_unit_id_set.add(uid)
        self.num_units = len(self.dp_managers)
        logger.info(
            "num_units: {}, units: {}, instance_types: {}".format(
                self.num_units, list(self.dp_managers.keys()), list(self.instance_types.values())
            )
        )

        return self.num_units

    async def _scale_down(self, unit_id: Union[str, Iterable[str]]) -> int:
        if isinstance(unit_id, str):
            unit_id = [unit_id,]
        unit_ids = list(unit_id)
        for uid in unit_ids:
            self.dp_managers.pop(uid, None)
            self.instance_types.pop(uid, None)
            self.prefill_unit_id_set.discard(uid)
            self.decode_unit_id_set.discard(uid)
        self.num_units = len(self.dp_managers)
        logger.info(
            "num_units: {}, units: {}, instance_types: {}".format(
                self.num_units, list(self.dp_managers.keys()), list(self.instance_types.values())
            )
        )

        await self._clear_unit_ray_resources(unit_ids)

        return self.num_units

    async def _clear_unit_ray_resources(self, unit_id: Union[str, Iterable[str]]) -> None:
        if isinstance(unit_id, str):
            unit_id = [unit_id,]
        unit_ids = list(unit_id)
        for uid in unit_ids:
            await kill_dp_manager(uid)
            remove_placement_group(uid)

    async def _clear_instance_ray_resources(self, instance_id: Union[str, Iterable[str]]):
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            await kill_instance(ins_id)

    async def _connect_to_dp_managers(self) -> int:
        """Connect to alive dp managers in the cluster when scaler restarts."""

        unit_ids, dp_managers, unit_types = await connect_to_actors_with_instance_type(actor_type=LlumnixActor.DP_MANAGER)
        if len(unit_ids) > 0:
            self._scale_up(unit_ids, dp_managers, unit_types)

    @property
    def enable_pd(self):
        return self.pdd_config.enable_pd_disagg or self.pdd_config.enable_engine_pd_disagg or self.pdd_config.enable_engine_semi_pd_disagg

    def _get_next_instance_type(self) -> str:
        if not self.enable_pd:
            return InstanceType.NEUTRAL

        pd_ratio = self.pdd_config.pd_ratio

        # There are no instances simultaneously in inflight_num_prefill_units and cur_num_prefill_units
        # as inflight_num will decrease before scaling up the instances. The same applies to num_decode.
        cur_num_prefill_units = len(self.prefill_unit_id_set)
        cur_num_decode_units = len(self.decode_unit_id_set)
        total_num_prefill_units = self.inflight_num_prefill_units + cur_num_prefill_units
        total_num_decode_units = self.inflight_num_decode_units + cur_num_decode_units

        if total_num_prefill_units == 0:
            instance_type = InstanceType.PREFILL
        elif total_num_decode_units == 0:
            instance_type = InstanceType.DECODE
        else:
            # compute distance if launch prefill or decode
            normal_distance = pd_ratio[0] - pd_ratio[1]

            base_num_ratio = min(total_num_prefill_units//pd_ratio[0], total_num_decode_units//pd_ratio[1])
            total_num_prefill_units = total_num_prefill_units - base_num_ratio * pd_ratio[0]
            total_num_decode_units = total_num_decode_units - base_num_ratio * pd_ratio[1]

            if total_num_prefill_units + total_num_decode_units == 0:
                instance_type = InstanceType.PREFILL
            else:
                distance_if_prefill = total_num_prefill_units + 1 - total_num_decode_units
                distance_if_decode = total_num_prefill_units - (total_num_decode_units + 1)
                gap_to_normal_if_prefill = abs(distance_if_prefill - normal_distance)
                gap_to_normal_if_decode = abs(distance_if_decode - normal_distance)
                instance_type = InstanceType.PREFILL if gap_to_normal_if_prefill <= gap_to_normal_if_decode else InstanceType.DECODE

        return instance_type
