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
import random
from typing import List, Tuple, Union, Iterable, Dict
from functools import partial

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
)
from llumnix.entrypoints.api_server_actor import APIServerActor
from llumnix.utils import (
    get_service_resouces,
    random_uuid,
    get_service_instance_type,
    asyncio_wait_for_ray_remote_call_with_timeout,
    run_coroutine_in_new_thread,
    log_instance_exception,
    BackendType,
    LaunchMode,
    InstanceType,
)
from llumnix.ray_utils import (
    get_manager_name,
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
    kill_dp_manager,
)
from llumnix.internal_config import PDDConfig
from llumnix.constants import (
    WAIT_PLACEMENT_GROUP_TIMEOUT,
    AUTO_SCALE_UP_INTERVAL,
    CHECK_DEPLOYMENT_STATES_INTERVAL,
    WATCH_DEPLOYMENT_INTERVAL,
    MAX_ACTOR_METHOD_RETRIES,
    HEARTBEAT_LOOP_INTERVAL,
)
from llumnix import envs as llumnix_envs
from llumnix.constants import NUM_GPUS_BLADELLM_GPU_ACTOR, NUM_GPUS_VLLM_V1_GPU_ACTOR
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
            self.manager = ray.get_actor(get_manager_name(), namespace="llumnix")
        # Start scaling after manager is ready.
        ray.get(self.manager.is_ready.remote())

        # launch args
        self.launch_mode: LaunchMode = launch_args.launch_mode
        self.backend_type: BackendType = launch_args.backend_type

        engine_args_factory_cls = self._get_engine_args_factory_cls(self.backend_type)
        self.llumnix_engine_args_factory: LlumnixEngineArgsFactory = engine_args_factory_cls(
            enable_port_increment=self.enable_port_increment,
            load_registered_service=self.load_registered_service,
            load_registered_service_path=self.load_registered_service_path,
            pdd_config=self.pdd_config,
        )

        self.port_offset = 0
        if self.enable_port_increment:
            if self.enable_port_offset_store:
                # TODO(s5u13b): Do not use ray interval kv.
                value = get_data_from_ray_internal_kv("scaler.port_offset")
                self.port_offset = int(value)

        self.instances: Dict[str, ray.actor.ActorHandle] = {}
        self.instance_types: Dict[str, InstanceType] = {}
        self.prefill_instance_id_set = set()
        self.decode_instance_id_set = set()

        self.instance_id_api_server_dict: Dict[str, ray.actor.ActorHandle] = {}

        self.inflight_num_prefill_instances = 0
        self.inflight_num_decode_instances = 0

        if self.backend_type == BackendType.VLLM_V1:
            # Mantain a monotonically increasing `client_index` for vLLM V1 APIServer.
            # It will be passed to APIServerActor through EntrypointsArgs.
            # TODO(shejiarui): Do not use ray interval kv.
            current_client_index = -1
            try:
                value = int(get_data_from_ray_internal_kv("scaler.client_index")) + 1
                current_client_index = value
                put_data_to_ray_internal_kv("scaler.client_index", value)
            except AssertionError:
                logger.debug("First time set scaler.client_index to 0.")
                current_client_index = 0
                put_data_to_ray_internal_kv("scaler.client_index", 0)
            except Exception as e: # pylint: disable=broad-except
                logger.exception(e)
            finally:
                self.client_index = current_client_index
                if self.client_index == -1:
                    logger.error("Failed to get client_index from ray internal kv: {}".format(self.client_index))

        run_coroutine_in_new_thread(self._connect_to_instances(), blocking=True)

        if self.launch_mode == LaunchMode.GLOBAL:
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
                        service_name="neutral",
                        max_instances=self.max_instances,
                        interval=AUTO_SCALE_UP_INTERVAL,
                    )
                )
            asyncio.create_task(self._heartbeat_loop(HEARTBEAT_LOOP_INTERVAL))
            # asyncio.create_task(self._check_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))
            if self.pdd_config.enable_pd_disagg:
                asyncio.create_task(self._check_pd_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))

    def _get_engine_args_factory_cls(self, instance_type: BackendType) -> LlumnixEngineArgsFactory:
        engine_args_factory_cls = None

        # pylint: disable=import-outside-toplevel
        if instance_type == BackendType.VLLM:
            from llumnix.entrypoints.vllm.arg_utils import VLLMEngineArgsFactory
            engine_args_factory_cls = VLLMEngineArgsFactory
        elif instance_type == BackendType.VLLM_V1:
            from llumnix.entrypoints.vllm_v1.arg_utils import VLLMV1EngineArgsFactory
            engine_args_factory_cls = VLLMV1EngineArgsFactory
        elif instance_type == BackendType.BLADELLM:
            from llumnix.entrypoints.bladellm.arg_utils import BladeLLMEngineArgsFactory
            engine_args_factory_cls = BladeLLMEngineArgsFactory
        else:
            raise ValueError("Unsupported instance type: {}.".format(instance_type))

        return engine_args_factory_cls

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
            try:
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
                    await self._clear_instance_ray_resources(instance_id)
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
                instance_type = get_service_instance_type(service_name) if service_name in ["prefill", "decode"] else None
                if self.engine_args.backend_type == BackendType.VLLM_V1:
                    self._init_dp_manager_v1(
                        new_instance_id,
                        self.entrypoints_args,
                        self.instance_args,
                        self.engine_args,
                        new_pg,
                        instance_type,
                    )
                else:
                    self._init_server_and_instance(
                        new_instance_id,
                        self.entrypoints_args,
                        self.instance_args,
                        self.engine_args,
                        new_pg,
                        instance_type,
                    )
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Scaler get error in _auto_scale_up_loop, scaler keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    async def _heartbeat_loop(self, interval: float) -> None:
        def check_instance_health_done_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if isinstance(ret, Exception):
                log_instance_exception(ret, instance_id, "check_instance_health")
                dead_instance_ids.append(instance_id)

        while True:
            try:
                await asyncio.sleep(interval)
                tasks = []
                dead_instance_ids = []
                for instance_id, instance in self.instances.items():
                    task = asyncio.gather(
                        asyncio_wait_for_ray_remote_call_with_timeout(instance.is_ready),
                        return_exceptions=True
                    )
                    task.add_done_callback(
                        partial(check_instance_health_done_callback, instance_id)
                    )
                    tasks.append(task)
                await asyncio.gather(*tasks)
                if len(dead_instance_ids) > 0:
                    await self.scale_down(dead_instance_ids)
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "Scaler get error in _heartbeat_loop, scaler keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    async def _check_deployment_states_loop(self, interval: float) -> None:
        # pylint: disable=unused-variable
        async def watch_instance_deployment_states(instance_id: str, server_exists: bool, instance_exists: bool):
            # Waiting for _init_server_and_instance scheduled.
            if not server_exists and not instance_exists:
                await asyncio.sleep(1.0)
            # Server is initialized after instance is ready, so waiting for instance ready first.
            if not server_exists and instance_exists:
                try:
                    instance = ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
                except ValueError:
                    logger.warning("Instance {} not found.".format(instance_id))
                    instance_exists = False
                if instance_exists:
                    try:
                        await asyncio_wait_for_ray_remote_call_with_timeout(
                            instance.is_ready, timeout=llumnix_envs.INSTANCE_READY_TIMEOUT
                        )
                    # Instance exception could be handled by manager/scaler, so check states loop omits exception case here.
                    # pylint: disable=bare-except
                    except:
                        pass
            logger.info(
                "Watch instance {} deployment states, server_exists: {}, instance_exists: {}".format(
                    instance_id, server_exists, instance_exists
                )
            )
            await asyncio.sleep(WATCH_DEPLOYMENT_INTERVAL)
            pg_created, server_exists, instance_exists = self._get_instance_deployment_states(instance_id)
            if pg_created and (not server_exists or not instance_exists):
                logger.warning(
                    "Instance {} deployment states incorrect, states: (pg {}, server {}, instance {})".format(
                        instance_id, pg_created, server_exists, instance_exists
                    )
                )
                await asyncio_wait_for_ray_remote_call_with_timeout(self.manager.scale_down, instance_id)

        def update_api_servers(instance_ids: List[str]):
            # delete dead api servers
            current_ids = list(self.instance_id_api_server_dict.keys())
            for instance_id in current_ids:
                if instance_id not in instance_ids:
                    self.instance_id_api_server_dict.pop(instance_id, None)
            # add new api servers
            for instance_id in instance_ids:
                if instance_id not in self.instance_id_api_server_dict:
                    self.instance_id_api_server_dict[instance_id] = ray.get_actor(get_server_name(instance_id), namespace="llumnix")


        while True:
            try:
                # Not check right after scaler initialized, so sleep at the beginning.
                await asyncio.sleep(interval)
                # pylint: disable=unused-variable
                curr_pgs, curr_servers, curr_instances = self._get_cluster_deployment_states()
                update_api_servers(curr_servers)
                # TODO(shejiarui): fix it in DP
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
        cur_num_prefill_instances = len(self.prefill_instance_id_set)
        cur_num_decode_instances = len(self.decode_instance_id_set)
        scale_down_instance_id = None
        if cur_num_prefill_instances == 0 and cur_num_decode_instances > 0:
            scale_down_instance_id = random.choice(list(self.decode_instance_id_set))
            logger.info("Check pd deployment, pd_ratio: {}, cur_num_prefill_instances: {}, cur_num_decode_instances: {}, "
                        "all decode instances is decode instance, scale down decode instance {}".format(
                        self.pdd_config.pd_ratio, cur_num_prefill_instances, cur_num_decode_instances, scale_down_instance_id))

        if cur_num_decode_instances == 0 and cur_num_prefill_instances > 0:
            scale_down_instance_id = random.choice(list(self.prefill_instance_id_set))
            logger.info("Check pd deployment, pd_ratio: {}, cur_num_prefill_instances: {}, cur_num_decode_instances: {}, "
                        "all instances is prefill instance, scale down prefill instance {}".format(
                        self.pdd_config.pd_ratio, cur_num_prefill_instances, cur_num_decode_instances, scale_down_instance_id))

        if scale_down_instance_id:
            await asyncio_wait_for_ray_remote_call_with_timeout(self.manager.scale_down, scale_down_instance_id)

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
            )
        elif not BackendType.is_sim_backend(backend_type):
            # num_cpus=2+(0/1), for Llumlet + ActorOutputForwarder + (ApiServerActor)
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

    def _init_server_and_instance(self,
                                 instance_id: str,
                                 entrypoints_args: EntrypointsArgs,
                                 instance_args: InstanceArgs,
                                 engine_args: LlumnixEngineArgs,
                                 placement_group: PlacementGroup,
                                 instance_type: InstanceType):
        async def instance_ready_scale_up(
            instance_id: str,
            instance: Llumlet,
            next_entrypoints_args: EntrypointsArgs,
            next_instance_args: InstanceArgs,
            next_engine_args: LlumnixEngineArgs,
        ):
            try:
                instance_ready = False
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    instance.is_ready, timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                )
                instance_ready = True
                # Initialize server after instance is ready.
                server = self._init_server(
                    instance_id,
                    placement_group,
                    backend_type,
                    next_entrypoints_args,
                    next_instance_args,
                    next_engine_args,
                    self.scaler,
                    self.manager,
                    instance,
                )
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    server.is_ready, timeout=float(llumnix_envs.SERVER_READY_TIMEOUT)
                )
                instance_type = next_instance_args.instance_type
                self.instance_id_api_server_dict[instance_id] = server
                await self.scale_up(instance_id, instance, instance_type)
                logger.info("Init server and instance done, instance_id: {}, instance_type: {}.".format(instance_id, instance_type))
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up instance {}, instance is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    if not instance_ready:
                        logger.error(
                            "Failed to scale up instance {}, instance is not ready in {} seconds.".format(
                                instance_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                            )
                        )
                    else:
                        logger.error(
                            "Failed to scale up instance {}, server is not ready in {} seconds.".format(
                                instance_id, float(llumnix_envs.SERVER_READY_TIMEOUT)
                            )
                        )
                else:
                    logger.exception("Error in scaler instance_ready_scale_up (instance_id: {})".format(instance_id))
                await self._clear_instance_ray_resources(instance_id)
            finally:
                self.inflight_num_prefill_instances -= 1 if instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode_instances -= 1 if instance_type == InstanceType.DECODE else 0

        backend_type = engine_args.backend_type
        request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)
        next_instance_type = self._get_next_instance_type() if instance_type is None else instance_type
        next_instance_args = InstanceArgs.get_next_instance_args(instance_args, next_instance_type)
        next_entrypoints_args = EntrypointsArgs.get_next_entrypoints_args(entrypoints_args, self.enable_port_increment, self.port_offset)
        next_engine_args = self.llumnix_engine_args_factory.gen_next_engine_args(
            current_engine_args=engine_args,
            next_instance_args=next_instance_args,
            port_offset=self.port_offset,
            instance_id=instance_id,
        )

        if self.enable_port_increment:
            self.port_offset += 1
            if self.enable_port_offset_store:
                put_data_to_ray_internal_kv("scaler.port_offset", self.port_offset)

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
            instance_ready_scale_up(
                instance_id,
                instance,
                next_entrypoints_args,
                next_instance_args,
                next_engine_args,
            )
        )

        logger.info(
            "Deploy server and instance to new placement group done, instance_id: {}.".format(instance_id)
        )

    def _init_server(self,
                     instance_id: str,
                     placement_group: PlacementGroup,
                     backend_type: BackendType,
                     entrypoints_args: EntrypointsArgs,
                     instance_args: InstanceArgs,
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
                instance_args,
                engine_args,
                scaler,
                manager,
                instance,
            )
        elif backend_type == BackendType.VLLM_V1:
            # pylint: disable=import-outside-toplevel
            from llumnix.entrypoints.vllm_v1.api_server_actor import APIServerActorVLLMV1
            # To avoid triton runtime error, assign GPU to api server.
            api_server = APIServerActorVLLMV1.from_args(
                NUM_GPUS_VLLM_V1_GPU_ACTOR,
                instance_id,
                placement_group,
                entrypoints_args,
                instance_args,
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
                instance_args,
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

    def _init_dp_manager_v1(self,
                            instance_id: str,
                            entrypoints_args: EntrypointsArgs,
                            instance_args: InstanceArgs,
                            engine_args: LlumnixEngineArgs,
                            placement_group: PlacementGroup,
                            instance_type: InstanceType) -> None:
        # pylint: disable=import-outside-toplevel
        from llumnix.entrypoints.vllm_v1.dp_manager import DPManager
        async def dp_manager_ready_scale_up(instance_id: str, dp_manager: DPManager, dp_size: int, instance_type: InstanceType):
            try:
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    dp_manager.is_ready, timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                )
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up dp manager {}, dp manager is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error(
                        "Failed to scale up dp manager {}, dp manager is not ready in {} seconds.".format(
                            instance_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                        )
                    )
                else:
                    logger.exception("Error in scaler dp_manager_ready_scale_up (instance_id: {})".format(instance_id))
                await self._clear_dp_manager_ray_resources(instance_id)
            finally:
                self.inflight_num_prefill_instances -= dp_size if instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode_instances -= dp_size if instance_type == InstanceType.DECODE else 0

        dp_size = engine_args.get_dp_size()

        next_instance_type = self._get_next_instance_type(dp_size) if instance_type is None else instance_type
        instance_args.instance_type = next_instance_type
        instance_args_list: List[InstanceArgs] = [instance_args] * dp_size
        entrypoints_args_list: List[EntrypointsArgs] = []
        instance_id_list: List[str] = []
        engine_args_list: List[LlumnixEngineArgs] = []
        for rank in range(dp_size):
            port_offset = self.port_offset + rank
            client_index = self.client_index + rank
            entrypoints_args_list.append(
                EntrypointsArgs.get_next_entrypoints_args(
                    entrypoints_args, self.enable_port_increment, port_offset, client_index
                )
            )
            instance_id_list.append("{}_{}".format(instance_id, random_uuid()))
            engine_args_list.append(
                self.llumnix_engine_args_factory.gen_next_engine_args(
                    engine_args, instance_args_list[rank], port_offset, instance_id
                )
            )

        dp_manager = DPManager.from_args(
            instance_id,
            next_instance_type,
            dp_size,
            instance_id_list,
            entrypoints_args_list,
            instance_args_list,
            engine_args_list,
            placement_group,
            self.scaler,
            self.manager,
        )

        self.inflight_num_prefill_instances += dp_size if next_instance_type == InstanceType.PREFILL else 0
        self.inflight_num_decode_instances += dp_size if next_instance_type == InstanceType.DECODE else 0

        if self.enable_port_increment:
            self.port_offset += dp_size
            if self.enable_port_offset_store:
                put_data_to_ray_internal_kv("scaler.port_offset", self.port_offset)
        self.client_index += dp_size
        put_data_to_ray_internal_kv("scaler.client_index", self.client_index)

        asyncio.create_task(dp_manager_ready_scale_up(instance_id, dp_manager, dp_size, next_instance_type))

        logger.info(
            "Deploy dp manager to new placement group done, instance_id: {}.".format(instance_id)
        )

    async def init_instances(self,
                             request_output_queue_type: QueueType,
                             instance_args: InstanceArgs,
                             engine_args: LlumnixEngineArgs,
                             node_id: str
                             ) -> Tuple[List[str], List[Llumlet]]:
        async def ready_scale_up(instance_id: str, instance: Llumlet, instance_type: InstanceType):
            try:
                await asyncio_wait_for_ray_remote_call_with_timeout(
                    instance.is_ready, timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                )
                await self.scale_up(instance_id, instance, instance_type)
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
                ready_scale_up(
                    instance_id,
                    instance,
                    instance_args.instance_type,
                )
            )

        return instance_ids, instances

    def init_request_output_queue_server(self, ip: str, queue_type: QueueType) -> QueueServerBase:
        return init_request_output_queue_server(ip, queue_type)

    # scale up: from scaler to manager
    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    async def scale_up(self,
                       instance_id: Union[str, Iterable[str]],
                       instance_actor_handle: Union[Llumlet, List[Llumlet]],
                       instance_type: Union[InstanceType, List[InstanceType]]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
            instance_actor_handle = [instance_actor_handle,]
            instance_type = [instance_type,]

        instance_ids: List[str] = list(instance_id)
        instance_actor_handles: List[Llumlet] = list(instance_actor_handle)
        instance_types: List[InstanceType] = list(instance_type)

        for ins_id, ins_actor_handle, ins_type in zip(instance_ids, instance_actor_handles, instance_types):
            self.instances[ins_id] = ins_actor_handle
            self.instance_types[ins_id] = ins_type
            if ins_type == InstanceType.PREFILL:
                self.prefill_instance_id_set.add(ins_id)
            elif ins_type == InstanceType.DECODE:
                self.decode_instance_id_set.add(ins_id)
        self.num_instances = len(self.instances)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, list(self.instances.keys())))

        await asyncio_wait_for_ray_remote_call_with_timeout(
            self.manager.scale_up, instance_ids, instance_actor_handles, instance_types
        )

        return self.num_instances

    # scale down: from manager to scaler
    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    async def scale_down(self, instance_id: Union[str, Iterable[str]]) -> None:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            self.instances.pop(ins_id, None)
            self.instance_types.pop(ins_id, None)
            self.prefill_instance_id_set.discard(ins_id)
            self.decode_instance_id_set.discard(ins_id)
        self.num_instances = len(self.instances)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, list(self.instances.keys())))
        await self.cancel_api_server_request(instance_ids)
        await self._clear_instance_ray_resources(instance_ids)

    async def cancel_api_server_request(self, dead_instance_ids: List[str]) -> None:
        tasks = []
        for _, api_server_actor_handle in self.instance_id_api_server_dict.items():
            api_server_actor_handle: APIServerActor = api_server_actor_handle
            task = asyncio.gather(
                asyncio_wait_for_ray_remote_call_with_timeout(api_server_actor_handle.clear_dead_instances, dead_instance_ids),
                return_exceptions=True
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    def is_ready(self) -> bool:
        return True

    async def _clear_instance_ray_resources(self, instance_id: Union[str, Iterable[str]]):
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

    async def _clear_dp_manager_ray_resources(self, instance_id: Union[str, Iterable[str]]):
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            await kill_dp_manager(ins_id)
            remove_placement_group(ins_id)

    async def _connect_to_instances(self):
        def connect_to_instance_done_callback(instance_id: str, instance_actor_handle: ray.actor.ActorHandle, fut):
            ret = fut.result()[0]
            if not isinstance(ret, Exception):
                try:
                    instance_ids.append(instance_id)
                    instance_actor_handles.append(instance_actor_handle)
                    instance_types.append(ret)
                    logger.info("Connect to instance {}".format(instance_id))
                except Exception as e: # pylint: disable=broad-except
                    if isinstance(e, ValueError):
                        logger.warning("Failed to connect to instance {}, placement group not found.".format(instance_id))
                    else:
                        logger.exception("Error in scaler _connect_to_instances get_placement_group (instance_id: {})".format(instance_id))
            else:
                log_instance_exception(ret, instance_id, "_connect_to_instances")

        instance_actor_names = get_actor_names_by_name_prefix(INSTANCE_NAME_PREFIX)
        available_instance_actor_names = []
        available_instance_actor_handles: List[ray.actor.ActorHandle] = []
        for actor_name in instance_actor_names:
            try:
                instance_actor_handle = ray.get_actor(actor_name, namespace='llumnix')
                available_instance_actor_names.append(actor_name)
                available_instance_actor_handles.append(instance_actor_handle)
            except Exception as e: # pylint: disable=broad-except
                instance_id = actor_name[len(INSTANCE_NAME_PREFIX):]
                if isinstance(e, ValueError):
                    logger.warning("Failed to connect to instance {}, actor not found.".format(instance_id))
                else:
                    logger.exception("Error in scaler _connect_to_instances get_actor (instance_id: {})".format(instance_id))

        instance_ids = []
        instance_actor_handles = []
        instance_types = []
        tasks = []
        for instance_actor_name, instance_actor_handle in \
            zip(available_instance_actor_names, available_instance_actor_handles):
            instance_id = instance_actor_name[len(INSTANCE_NAME_PREFIX):]
            task = asyncio.gather(
                asyncio_wait_for_ray_remote_call_with_timeout(instance_actor_handle.get_instance_type),
                return_exceptions=True
            )
            task.add_done_callback(
                partial(connect_to_instance_done_callback, instance_id, instance_actor_handle)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

        await self.scale_up(instance_ids, instance_actor_handles, instance_types)

    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    def get_instances(self):
        instance_ids = []
        instance_actor_handles = []
        instance_types = []
        for instance_id, instance_actor_handle in self.instances.items():
            instance_ids.append(instance_id)
            instance_actor_handles.append(instance_actor_handle)
            instance_types.append(self.instance_types[instance_id])

        return instance_ids, instance_actor_handles, instance_types

    @property
    def enable_pd(self):
        return self.pdd_config.enable_pd_disagg or self.pdd_config.enable_engine_pd_disagg or self.pdd_config.enable_engine_semi_pd_disagg

    def _get_next_instance_type(self, instance_num: int = 1) -> str:
        if not self.enable_pd:
            return InstanceType.NEUTRAL

        pd_ratio = self.pdd_config.pd_ratio

        # There are no instances simultaneously in inflight_num_prefill_instances and cur_num_prefill_instances
        # as inflight_num will decrease before scaling up the instances. The same applies to num_decode.
        cur_num_prefill_instances = len(self.prefill_instance_id_set)
        cur_num_decode_instances = len(self.decode_instance_id_set)
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
                distance_if_prefill = total_num_prefill_instances + instance_num - total_num_decode_instances
                distance_if_decode = total_num_prefill_instances - (total_num_decode_instances + instance_num)
                gap_to_normal_if_prefill = abs(distance_if_prefill - normal_distance)
                gap_to_normal_if_decode = abs(distance_if_decode - normal_distance)
                instance_type = InstanceType.PREFILL if gap_to_normal_if_prefill <= gap_to_normal_if_decode else InstanceType.DECODE

        return instance_type

    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    def clear_gloo_backend_ray_resources(self):
        clear_gloo_backend_ray_resources()
