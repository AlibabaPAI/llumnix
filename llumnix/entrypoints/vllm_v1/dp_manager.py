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

from typing import List, Dict, Union, Iterable
from enum import Enum

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
    get_dp_manager_name,
    kill_server,
    kill_instance,
    remove_placement_group,
    INSTANCE_NAME_PREFIX,
    SERVER_NAME_PREFIX,
    get_actor_names_by_name_prefix,
)
from llumnix.queue.queue_type import QueueType
from llumnix.entrypoints.vllm_v1.api_server_actor import APIServerActorVLLMV1
from llumnix.constants import NUM_GPUS_VLLM_V1_GPU_ACTOR
from llumnix.utils import run_coroutine_in_new_thread, InstanceType
import llumnix.envs as llumnix_envs

logger = init_logger(__name__)


class DPGroupStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "partial"
    EMPTY = "empty"


class DPManager:
    def __init__(
        self,
        instance_id: str,
        instance_type: InstanceType,
        dp_size: int,
        instance_id_list: List[str],
        entrypoints_args_list: List[EntrypointsArgs],
        instance_args_list: List[InstanceArgs],
        engine_args_list: List[LlumnixEngineArgs],
        placement_group: PlacementGroup,
        scaler: ray.actor.ActorHandle,
        manager: ray.actor.ActorHandle,
    ):
        self.instance_id = instance_id
        self.instance_type = instance_type
        self.dp_size = dp_size
        self.placement_group = placement_group
        self.scaler = scaler
        self.manager = manager

        self.actor_handle = ray.get_actor(name=get_dp_manager_name(instance_id), namespace="llumnix")

        self.instances: Dict[str, ray.actor.ActorHandle] = {}
        self.servers: Dict[str, ray.actor.ActorHandle] = {}

        dp_group_status = self._connect_to_instances_and_servers()
        if dp_group_status == DPGroupStatus.PARTIAL:
            run_coroutine_in_new_thread(self.stop(), blocking=True)
        elif dp_group_status == DPGroupStatus.EMPTY:
            instances, servers = self._init_instances_and_servers(
                dp_size,
                instance_id_list,
                entrypoints_args_list,
                instance_args_list,
                engine_args_list,
                placement_group,
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

    @classmethod
    def from_args(
        cls,
        instance_id: str,
        instance_type: InstanceType,
        dp_size: int,
        instance_id_list: List[str],
        entrypoints_args_list: List[EntrypointsArgs],
        instance_args_list: List[InstanceArgs],
        engine_args_list: List[LlumnixEngineArgs],
        placement_group: PlacementGroup,
        scaler: ray.actor.ActorHandle,
        manager: ray.actor.ActorHandle
    ) -> "DPManager":
        dp_manager_class = ray.remote(
            num_cpus=1,
            name=get_dp_manager_name(instance_id),
            namespace="llumnix",
            lifetime="detached",
        )(cls).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True
            )
        )
        dp_manager = dp_manager_class.remote(
            instance_id,
            instance_type,
            dp_size,
            instance_id_list,
            entrypoints_args_list,
            instance_args_list,
            engine_args_list,
            placement_group,
            scaler,
            manager,
        )
        return dp_manager

    def _init_instances_and_servers(
        self,
        dp_size: int,
        instance_id_list: List[str],
        entrypoints_args_list: List[EntrypointsArgs],
        instance_args_list: List[InstanceArgs],
        engine_args_list: List[LlumnixEngineArgs],
        placement_group: PlacementGroup,
        scaler: ray.actor.ActorHandle,
        manager: ray.actor.ActorHandle,
    ) -> None:
        # NOTE(shejiarui): noqa for kvt
        dp_rank_local = 0
        instances = []
        servers = []
        for rank in range(dp_size):
            entrypoints_args = entrypoints_args_list[rank]
            instance_args = instance_args_list[rank]
            engine_args = engine_args_list[rank]
            request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)

            instance = Llumlet.from_args(
                instance_id_list[rank],
                instance_args,
                placement_group,
                request_output_queue_type,
                engine_args,
                dp_rank=rank,
                dp_rank_local=dp_rank_local,
            )

            server = APIServerActorVLLMV1.from_args(
                NUM_GPUS_VLLM_V1_GPU_ACTOR,
                instance_id_list[rank],
                placement_group,
                entrypoints_args,
                instance_args,
                engine_args,
                scaler,
                manager,
                instance,
                bundle_index=rank
            )

            instances.append(instance)
            servers.append(server)

        logger.info(
            "Deploy {} servers and {} instances to new placement group done, instance_id: {}.".format(
                self.dp_size, self.dp_size, self.instance_id
            )
        )

        return instances, servers

    def _wait_for_instances_and_servers_ready(
        self,
        instance_id_list: List[str],
        instances: List[ray.actor.ActorHandle],
        servers: List[ray.actor.ActorHandle]
    ) -> bool:
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
                logger.warning("Failed to scale up dp manager {}, some actors(instances/servers) are dead.".format(self.instance_id))
            elif isinstance(e, ray.exceptions.GetTimeoutError):
                if not instance_ready:
                    logger.error(
                        "Failed to scale up dp manager {}, instances is not ready in {} seconds.".format(
                            self.instance_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)
                        )
                    )
                else:
                    logger.error(
                        "Failed to scale up dp manager {}, servers is not ready in {} seconds.".format(
                            self.instance_id, float(llumnix_envs.SERVER_READY_TIMEOUT)
                        )
                    )
            return False

    def _connect_to_instances_and_servers(self) -> DPGroupStatus:
        instance_names = get_actor_names_by_name_prefix(f"{INSTANCE_NAME_PREFIX}{self.instance_id}")
        server_names = get_actor_names_by_name_prefix(f"{SERVER_NAME_PREFIX}{self.instance_id}")

        # non-restart case, return True to initialize instances and servers
        if len(instance_names) == 0 and len(server_names) == 0:
            return DPGroupStatus.EMPTY

        if len(instance_names) < self.dp_size or len(server_names) < self.dp_size:
            return DPGroupStatus.PARTIAL

        instance_ids = [instance_name[len(INSTANCE_NAME_PREFIX):] for instance_name in instance_names]
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

    def is_ready(self):
        return True

    async def stop(self):
        await self._scale_down()
        ray.kill(self.actor_handle)
        remove_placement_group(self.instance_id)

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

        ray.get(self.scaler.scale_up.remote(instance_ids, instance_actor_handles, [self.instance_type] * self.dp_size))

        return self.num_instances

    async def _scale_down(self) -> None:
        instance_ids = list(self.instances.keys())
        self.instances.clear()
        self.servers.clear()
        await self._clear_instance_ray_resources(instance_ids)

    async def _clear_instance_ray_resources(self, instance_id: Union[str, Iterable[str]]):
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            await kill_server(ins_id)
            await kill_instance(ins_id)
