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

from functools import partial
from typing import List, Dict, Optional, Union, Iterable
from enum import Enum
import asyncio
import time

import ray
import ray.actor
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray.exceptions

from llumnix.logging.logger import init_logger
from llumnix.arg_utils import (
    EntrypointsArgs,
    InstanceArgs,
    LlumnixEngineArgs,
)
from llumnix.llumlet.llumlet import Llumlet
from llumnix.ray_utils import (
    kill_server,
    kill_instance,
    remove_placement_group,
    INSTANCE_NAME_PREFIX,
    SERVER_NAME_PREFIX,
    list_actor_names_by_name_prefix,
    LlumnixActor,
    update_cluster_actor_handles,
    check_actors_health,
    get_llumnix_actor_name,
    get_llumnix_actor_id,
    get_llumnix_actor_handle,
    log_instance_exception
)
from llumnix.queue.queue_type import QueueType
from llumnix.constants import NUM_GPUS_BLADELLM_GPU_ACTOR, HEARTBEAT_INTERVAL
from llumnix.utils import (
    run_coroutine_in_new_thread,
    InstanceType,
    BackendType,
    UnitStatus,
    asyncio_wait_for_ray_remote_call_with_timeout,
)
import llumnix.envs as llumnix_envs

logger = init_logger(__name__)


class DPGroupStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "partial"
    EMPTY = "empty"


# DPManager is a general launcher used to launch instance and server,
# which is used in both non-data-parallel(dp_size=1) and data-parallel(dp_size>1) cases.
class DPManager:
    def __init__(
        self,
        unit_id: str,
        instance_type: InstanceType,
        dp_size: int,
        dp_size_local: int,
        instance_id_list: List[str],
        entrypoints_args_list: List[EntrypointsArgs],
        instance_args_list: List[InstanceArgs],
        engine_args_list: List[LlumnixEngineArgs],
        placement_group: PlacementGroup,
        backend_type: BackendType,
        scaler: ray.actor.ActorHandle,
        manager: ray.actor.ActorHandle,
    ):
        self.unit_id = unit_id
        self.instance_type = instance_type
        self.dp_size = dp_size
        self.dp_size_local = dp_size_local
        self.placement_group = placement_group
        self.backend_type = backend_type
        self.scaler = scaler
        self.manager = manager

        self.actor_handle = get_llumnix_actor_handle(LlumnixActor.DP_MANAGER, unit_id)

        self.instances: Dict[str, ray.actor.ActorHandle] = {}
        self.servers: Dict[str, ray.actor.ActorHandle] = {}
        self.cached_cluster_servers: Dict[str, ray.actor.ActorHandle] = {}

        self.unit_status = UnitStatus.HEALTHY
        dp_group_status = self._connect_to_instances_and_servers()
        logger.info("DPManager starts, dp_group_status: {}".format(dp_group_status))
        # Trigger heartbeat loop here so that 'self.stop' can be called when detected PARTIAL status.
        asyncio.create_task(self._heartbeat_loop(HEARTBEAT_INTERVAL))

        if dp_group_status == DPGroupStatus.PARTIAL:
            run_coroutine_in_new_thread(self.stop(), blocking=True)
        elif dp_group_status == DPGroupStatus.EMPTY:
            instances, servers = self._init_instances_and_servers(
                dp_size,
                dp_size_local,
                instance_id_list,
                entrypoints_args_list,
                instance_args_list,
                engine_args_list,
                placement_group,
                backend_type,
                scaler,
                manager,
            )
            dp_group_ready = self._wait_for_instances_and_servers_ready(instance_id_list, instances, servers)
            if dp_group_ready:
                self._scale_up(instance_id_list, instances, servers)
            else:
                run_coroutine_in_new_thread(self.stop(), blocking=True)
        else: # DPGroupStatus.COMPLETE
            logger.info("Restart dp manager successfully, dp group is complete.")

    def is_ready(self) -> bool:
        return True

    def get_instance_type(self) -> bool:
        return self.instance_type

    @classmethod
    def from_args(
        cls,
        unit_id: str,
        instance_type: InstanceType,
        dp_size: int,
        dp_size_local: int,
        instance_id_list: List[str],
        entrypoints_args_list: List[EntrypointsArgs],
        instance_args_list: List[InstanceArgs],
        engine_args_list: List[LlumnixEngineArgs],
        placement_group: PlacementGroup,
        backend_type: BackendType,
        scaler: ray.actor.ActorHandle,
        manager: ray.actor.ActorHandle
    ) -> "DPManager":
        dp_manager_class = ray.remote(
            num_cpus=1,
            name=get_llumnix_actor_name(LlumnixActor.DP_MANAGER, unit_id),
            namespace="llumnix",
            lifetime="detached",
            max_restarts=-1,
        )(cls).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True
            )
        )
        dp_manager = dp_manager_class.remote(
            unit_id,
            instance_type,
            dp_size,
            dp_size_local,
            instance_id_list,
            entrypoints_args_list,
            instance_args_list,
            engine_args_list,
            placement_group,
            backend_type,
            scaler,
            manager,
        )
        return dp_manager

    def _init_instances_and_servers(
        self,
        dp_size: int,
        dp_size_local: int,
        instance_id_list: List[str],
        entrypoints_args_list: List[EntrypointsArgs],
        instance_args_list: List[InstanceArgs],
        engine_args_list: List[LlumnixEngineArgs],
        placement_group: PlacementGroup,
        backend_type: BackendType,
        scaler: ray.actor.ActorHandle,
        manager: ray.actor.ActorHandle,
    ) -> None:
        # NOTE(shejiarui): noqa for kvt
        dp_rank_local = 0
        instances = []
        servers = []
        for dp_rank in range(dp_size):
            instance_id = instance_id_list[dp_rank]
            entrypoints_args = entrypoints_args_list[dp_rank]
            instance_args = instance_args_list[dp_rank]
            engine_args = engine_args_list[dp_rank]
            request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)

            instance = self._init_instance(
                instance_id,
                placement_group,
                instance_args,
                engine_args,
                dp_rank,
                dp_rank_local,
                request_output_queue_type,
            )
            dp_rank_local += 1
            dp_rank_local %= dp_size_local

            server = self._init_server(
                instance_id,
                placement_group,
                entrypoints_args,
                instance_args,
                engine_args,
                scaler,
                manager,
                instance,
                dp_rank,
                backend_type,
            )

            instances.append(instance)
            servers.append(server)

        logger.info(
            "Deploy {} servers and {} instances to new placement group done, instance_id: {}.".format(
                self.dp_size, self.dp_size, self.unit_id
            )
        )

        return instances, servers

    def _init_instance(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        instance_args: InstanceArgs,
        engine_args: LlumnixEngineArgs,
        dp_rank: int,
        dp_rank_local: int,
        request_output_queue_type: QueueType,
    ) -> ray.actor.ActorHandle:
        instance = Llumlet.from_args(
            instance_id,
            instance_args,
            placement_group,
            request_output_queue_type,
            engine_args,
            dp_rank=dp_rank,
            dp_rank_local=dp_rank_local,
        )

        return instance

    def _init_server(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        entrypoints_args: EntrypointsArgs,
        instance_args: InstanceArgs,
        engine_args: LlumnixEngineArgs,
        scaler: ray.actor.ActorHandle,
        manager: ray.actor.ActorHandle,
        instance: ray.actor.ActorHandle,
        dp_rank: int,
        backend_type: BackendType,
    ) -> ray.actor.ActorHandle:
        if backend_type == BackendType.VLLM_V1:
            # pylint: disable=import-outside-toplevel
            from llumnix.entrypoints.vllm_v1.api_server_actor import APIServerActorVLLMV1
            # To avoid triton runtime error, assign GPU to api server.
            server = APIServerActorVLLMV1.from_args(
                0,
                instance_id,
                placement_group,
                entrypoints_args,
                instance_args,
                engine_args,
                scaler,
                manager,
                instance,
                bundle_index=dp_rank,
            )
        elif backend_type == BackendType.BLADELLM:
            # pylint: disable=import-outside-toplevel
            from llumnix.entrypoints.bladellm.api_server_actor import APIServerActorBladeLLM
            server = APIServerActorBladeLLM.from_args(
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
        else: # [BackendType.VLLM, BackendType.SIM_VLLM]
            from llumnix.entrypoints.vllm.api_server_actor import APIServerActorVLLM # pylint: disable=import-outside-toplevel
            server = APIServerActorVLLM.from_args(
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

        return server

    def _wait_for_instances_and_servers_ready(
        self,
        instance_id_list: List[str],
        instances: List[ray.actor.ActorHandle],
        servers: List[ray.actor.ActorHandle]
    ) -> bool:
        """Wait for instances and servers to be ready in dp manager consturctor."""

        try:
            instance_ready = False
            ray.get([instance.is_ready.remote() for instance in instances], timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT))
            instance_ready = True
            ray.get([server.is_ready.remote() for server in servers], timeout=float(llumnix_envs.SERVER_READY_TIMEOUT))
            logger.info(
                "Init dp manager {} instances and {} servers done, instance_id: {}, instance_type: {}.".format(
                    self.dp_size, self.dp_size, instance_id_list, self.instance_type
                )
            )
            return True
        except Exception as e: # pylint: disable=broad-except
            if isinstance(e, ray.exceptions.RayActorError):
                logger.warning("Failed to scale up dp manager {}, some actors(instances/servers) are dead.".format(self.unit_id))
            elif isinstance(e, ray.exceptions.GetTimeoutError):
                if not instance_ready:
                    logger.error(
                        "Failed to scale up dp manager {}, instances is not ready in {} seconds.".format(
                            self.unit_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                        )
                    )
                else:
                    logger.error(
                        "Failed to scale up dp manager {}, servers is not ready in {} seconds.".format(
                            self.unit_id, float(llumnix_envs.SERVER_READY_TIMEOUT)
                        )
                    )
            return False

    async def _stop(self):
        logger.info("DPManager {} stops.".format(self.unit_id))
        await self._broadcast_dead_instances_to_cluster_servers(list(self.instances.keys()))
        await self._scale_down()
        # Detached actor will not be killed when the placement group is removed.
        remove_placement_group(self.unit_id)
        ray.kill(self.actor_handle)

    async def stop(self):
        self.terminating_event = asyncio.Event()
        await self._set_unit_status(UnitStatus.TERMINATING)
        self.stop_time = time.perf_counter()
        await self.terminating_event.wait()
        # Call 'self._stop' here instead of calling it implicitly in heartbear loop.
        await self._stop()

    def _scale_up(
        self,
        instance_id: Union[str, Iterable[str]],
        instance_actor_handle: Union[ray.actor.ActorHandle, Iterable[ray.actor.ActorHandle]],
        server_actor_handle: Union[ray.actor.ActorHandle, Iterable[ray.actor.ActorHandle]]
    ) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
            instance_actor_handle = [instance_actor_handle,]
            server_actor_handle = [server_actor_handle,]
        instance_ids: List[str] = list(instance_id)
        instance_actor_handles: List[ray.actor.ActorHandle] = list(instance_actor_handle)
        server_actor_handles: List[ray.actor.ActorHandle] = list(server_actor_handle)
        for ins_id, instance, server in zip(instance_ids, instance_actor_handles, server_actor_handles):
            self.instances[ins_id] = instance
            self.servers[ins_id] = server
        self.num_instances = len(self.instances)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, list(self.instances.keys())))

        # manager lack of instance discovery ability, so dp manager call manager here to notify.
        asyncio.create_task(
            asyncio_wait_for_ray_remote_call_with_timeout(
                self.manager.scale_up, instance_ids, instance_actor_handles, [self.instance_type] * self.dp_size
            )
        )

        return self.num_instances

    async def _scale_down(self, instance_ids: Optional[List[str]] = None) -> None:
        if instance_ids is None:
            instance_ids = list(self.instances.keys())
        await self._clear_instance_ray_resources(instance_ids)
        for ins_id in instance_ids:
            self.servers.pop(ins_id, None)
            self.instances.pop(ins_id, None)

    async def _clear_instance_ray_resources(self, instance_id: Union[str, Iterable[str]] = None) -> None:
        if instance_id is not None:
            if isinstance(instance_id, str):
                instance_id = [instance_id,]
            instance_ids = list(instance_id)
        else:
            instance_names = list_actor_names_by_name_prefix(f"{INSTANCE_NAME_PREFIX}{self.unit_id}")
            server_names = list_actor_names_by_name_prefix(f"{SERVER_NAME_PREFIX}{self.unit_id}")
            instance_ids = [get_llumnix_actor_id(LlumnixActor.INSTANCE, instance_name) for instance_name in instance_names]
            instance_ids.extend([get_llumnix_actor_id(LlumnixActor.SERVER, server_name) for server_name in server_names])
        for ins_id in instance_ids:
            await kill_server(ins_id, self.servers.get(ins_id, None))
            await kill_instance(ins_id, self.instances.get(ins_id, None))

    async def _heartbeat_loop(self, interval: float) -> None:
        """Watch cached instances and servers health."""
        unit_failover_timeout = float(llumnix_envs.UNIT_FAILOVER_TIMEOUT)

        while True:
            try:
                await asyncio.sleep(interval)
                # If the unit is health up to this point, continue checking the status of instances and servers.
                if self.unit_status == UnitStatus.HEALTHY:
                    dead_instance_ids = await check_actors_health(self.instances)
                    dead_instance_ids.extend(await check_actors_health(self.servers))
                    if len(dead_instance_ids) > 0:
                        await self._set_unit_status(UnitStatus.BROKEN)
                        self.stop_time = time.perf_counter()
                # If the unit has been broken already, wait for instances to migrate requests.
                else:
                    all_stopped = await self.check_instance_failover_status()
                    failover_timeout = time.perf_counter() - self.stop_time > unit_failover_timeout
                    if all_stopped or failover_timeout:
                        if self.unit_status == UnitStatus.BROKEN:
                            await self._stop()
                        elif self.unit_status == UnitStatus.TERMINATING:
                            # 'self.stop' will wait on this event.
                            self.terminating_event.set()
            # pylint: disable=broad-except
            except Exception:
                logger.critical(
                    "DPManager get error in _heartbeat_loop, dp manager keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    async def _broadcast_dead_instances_to_cluster_servers(self, dead_instance_ids: List[str]) -> None:
        """Broadcast dead instances to servers in the cluster to cancel dead instance requests."""
        self._update_cached_cluster_servers()

        logger.info(
            "Broadcast dead instances to cluster servers, cached cluster servers: {}, instance_ids: {}".format(
                self.cached_cluster_servers, dead_instance_ids
            )
        )
        tasks = []
        for _, server in self.cached_cluster_servers.items():
            task = asyncio.gather(
                asyncio_wait_for_ray_remote_call_with_timeout(server.cancel_dead_instance_requests, dead_instance_ids),
                return_exceptions=True
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    def _update_cached_cluster_servers(self) -> None:
        """Update cached cluster servers to broadcast to all servers when instances are dead."""

        self.cached_cluster_servers = update_cluster_actor_handles(
            actor_type=LlumnixActor.SERVER,
            cached_cluster_actors=self.cached_cluster_servers,
        )

    def _connect_to_instances_and_servers(self) -> DPGroupStatus:
        "Connect to alive instances and servers in the cluster when dp manager restarts."

        instance_names = list_actor_names_by_name_prefix(f"{INSTANCE_NAME_PREFIX}{self.unit_id}")
        server_names = list_actor_names_by_name_prefix(f"{SERVER_NAME_PREFIX}{self.unit_id}")

        # non-restart case, return True to initialize instances and servers
        if len(instance_names) == 0 and len(server_names) == 0:
            return DPGroupStatus.EMPTY

        if len(instance_names) < self.dp_size or len(server_names) < self.dp_size:
            return DPGroupStatus.PARTIAL

        instance_ids = [get_llumnix_actor_id(LlumnixActor.INSTANCE, instance_name) for instance_name in instance_names]
        instances = []
        servers = []
        for instance_id, instance_name, server_name in zip(instance_ids, instance_names, server_names):
            try:
                instance_found, server_found = False, False
                instance = ray.get_actor(instance_name, namespace="llumnix")
                instance_found = True
                server = ray.get_actor(server_name, namespace="llumnix")
                server_found = True
                instances.append(instance)
                servers.append(server)
            # pylint: disable=broad-except
            except Exception as e:
                if isinstance(e, ValueError):
                    logger.warning(
                        "Failed to connect to instance {}, actor is dead (instance found: {}, server found: {}).".format(
                            instance_id, instance_found, server_found
                        )
                    )
                else:
                    logger.exception("Error in dp manager _connect_to_instances_and_servers get_actor (instance_id: {})".format(instance_id))
                return False

        self._scale_up(instance_ids, instances, servers)

        return DPGroupStatus.COMPLETE

    async def _partial_scale_down(self, ret: Exception, instance_id: str, method_name: str):
        log_instance_exception(ret, instance_id, method_name)
        await self._broadcast_dead_instances_to_cluster_servers([instance_id])
        await self._scale_down([instance_id])

    async def _set_unit_status(self, status: UnitStatus) -> None:
        def instance_set_unit_status_callback(fut, instance_id: str) -> None:
            try:
                ret = fut.result()[0]
                if isinstance(ret, Exception):
                    asyncio.create_task(self._partial_scale_down(ret, instance_id, "set_unit_status"))
            except Exception as e:
                logger.exception(e)

        tasks = []
        for ins_id, ins_handle in self.instances.items():
            task = asyncio.gather(
                asyncio_wait_for_ray_remote_call_with_timeout(ins_handle.set_unit_status, status),
                return_exceptions=True
            )
            task.add_done_callback(
                partial(
                    instance_set_unit_status_callback, instance_id=ins_id
                )
            )
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("DPManager(unit_id={}) change unit_status {} -> {}".format(
            self.unit_id, self.unit_status, status))
        self.unit_status = status

    async def check_instance_failover_status(self) -> bool:
        def check_instance_failover_status_callback(fut, instance_id: str):
            try:
                ret = fut.result()[0]
                if isinstance(ret, Exception):
                    asyncio.create_task(self._partial_scale_down(ret, instance_id, "get_unit_status"))
                else:
                    if ret == UnitStatus.STOPPED:
                        stopped_instances.append(ret)
            except Exception as e:
                # TODO(shejiarui): may need more robust error handling
                logger.exception(e)

        tasks = []
        stopped_instances = []
        for instance_id, instance in self.instances.items():
            task = asyncio.gather(
                asyncio_wait_for_ray_remote_call_with_timeout(instance.get_unit_status),
                return_exceptions=True
            )
            task.add_done_callback(
                partial(check_instance_failover_status_callback, instance_id=instance_id)
            )
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

        all_stopped = len(self.instances) == 0 or len(self.instances) == len(stopped_instances)
        return all_stopped
