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
from typing import List

import ray
import ray.actor
from ray.util.placement_group import PlacementGroup

from llumnix import envs as llumnix_envs
from llumnix.logging.logger import init_logger
from llumnix.arg_utils import (
    EntrypointsArgs,
    InstanceArgs,
    LlumnixEngineArgs,
)
from llumnix.llumlet.llumlet import Llumlet
from llumnix.entrypoints.vllm_v1.arg_utils import VLLMV1DPArgs
from llumnix.ray_utils import (
    get_scaler_name, get_manager_name, get_dpmanager_name,
    kill_server, kill_instance, remove_placement_group)
from llumnix.instance_info import InstanceType
from llumnix.queue.queue_type import QueueType
from llumnix.entrypoints.vllm_v1.api_server_actor import APIServerActorVLLMV1
from llumnix.constants import NUM_GPUS_VLLM_V1_GPU_ACTOR, MAX_ACTOR_METHOD_RETRIES
from llumnix.utils import random_uuid, asyncio_wait_for_ray_remote_call_with_timeout

logger = init_logger(__name__)


class DPManager:
    def __init__(self,
                 instance_id: str,
                 new_instance_ids: List[str],
                 entrypoints_args: EntrypointsArgs,
                 instance_args: InstanceArgs,
                 engine_args: LlumnixEngineArgs,
                 placement_group: PlacementGroup):
        self.instance_id = instance_id
        self.entrypoints_args = entrypoints_args
        self.instance_args = instance_args
        self.placement_group = placement_group
        self.engine_args = engine_args

        dp_args: VLLMV1DPArgs = engine_args.get_dp_args()
        self.dp_size = dp_args.dp_size
        self.dp_size_local = dp_args.dp_size_local
        self.dp_address = dp_args.dp_address
        self.dp_rpc_port = dp_args.dp_rpc_port

        self.scaler = ray.get_actor(get_scaler_name(), namespace="llumnix")
        self.manager = ray.get_actor(get_manager_name(), namespace="llumnix")

        self.port_offset = 0
        self.client_index_offset = 0

        self.instance_ids = new_instance_ids
        self.instances: List[ray.actor.ActorHandle] = []
        self.servers: List[ray.actor.ActorHandle] = []
        self._init_instances_and_servers()
        
        try:
            ray.get([server.is_ready.remote() for server in self.servers])
            ray.get([instance.is_ready.remote() for instance in self.instances])
            self.manager.scale_up.remote(self.instance_ids, self.instances, 
                                         [None] * self.dp_size)
        except Exception as e:
            # TODO(shejiarui): exception need to be handled
            logger.exception(e)

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  new_instance_ids: List[str],
                  entrypoints_args: EntrypointsArgs,
                  instance_args: InstanceArgs,
                  engine_args: LlumnixEngineArgs,
                  placement_group: PlacementGroup,
                  ) -> "DPManager":
        dp_manager_class = ray.remote(
            num_cpus=1,
            name=get_dpmanager_name(instance_id),
            namespace="llumnix",
        )(cls)
        dp_manager = dp_manager_class.remote(
            instance_id,
            new_instance_ids,
            entrypoints_args,
            instance_args,
            engine_args,
            placement_group,
        )
        return dp_manager

    def _init_instances_and_servers(self) -> None:
        request_output_queue_type = QueueType(
            self.entrypoints_args.request_output_queue_type)
        # NOTE(shejiarui): noqa for kvt
        dp_rank_local = 0
        
        for rank in range(self.dp_size):
            new_instance_id = self.instance_ids[rank]
                
            instance = Llumlet.from_args(
                new_instance_id,
                self.instance_args,
                self.placement_group,
                request_output_queue_type,
                self.engine_args,
                dp_rank=rank,
                dp_rank_local=dp_rank_local,
            )
            self.instances.append(instance)

            self.entrypoints_args.port += self.port_offset
            self.entrypoints_args.client_index += self.client_index_offset
            self.port_offset += 1
            self.client_index_offset += 1
            server = APIServerActorVLLMV1.from_args(
                NUM_GPUS_VLLM_V1_GPU_ACTOR,
                new_instance_id,
                self.placement_group,
                self.entrypoints_args,
                self.engine_args,
                self.scaler,
                self.manager,
                instance,
                bundle_index=rank
            )
            self.servers.append(server)

    async def _done_scale_up(self,
                             placement_group: PlacementGroup,
                             instance_type: InstanceType = None) -> None:
        for i in range(self.dp_size):
            instance_id = self.instance_ids[i]
            try:
                server_ready = False
                await asyncio.wait_for(self.servers[i].is_ready.remote(), 
                                       timeout=float(llumnix_envs.SERVER_READY_TIMEOUT))
                server_ready = True
                await asyncio.wait_for(self.instances[i].is_ready.remote(), 
                                       timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT))
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up DP instance {}, instance is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    if server_ready:
                        logger.error("DP Instance {} is not ready in {} seconds."
                                     .format(instance_id, float(llumnix_envs.INSTANCE_READY_TIMEOUT)))
                    else:
                        logger.error("DP Server {} is not ready in {} seconds."
                                     .format(instance_id, float(llumnix_envs.SERVER_READY_TIMEOUT)))
                else:
                    logger.exception("Error in dpmanager(instance_id: {}) done_scale_up.".format(self.instance_id))
                await self.clear_dp_ray_resources()

        await asyncio_wait_for_ray_remote_call_with_timeout(
            self.manager.scale_up.remote, self.instance_ids, self.instances, instance_type,
            placement_group, self.servers
        )

    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    async def clear_dp_ray_resources(self):
        for ins_id in self.instance_ids:
            # There could be multiple clear_instance_ray_resources calls for one error instance,
            # so the kill operations could be failed if it is not the first attempt to kill.
            await kill_server(ins_id)
            await kill_instance(ins_id)
        remove_placement_group(self.instance_id)

    async def is_ready(self):
        """Called by Scaler, return true when all the DP instances and api servers
        have been successfully created.
        """
        tasks = [
            asyncio_wait_for_ray_remote_call_with_timeout(instance.is_ready.remote)
            for instance in self.instances
        ]
        # Note that llumnix run server and scale up instance in manager after instance is ready,
        # so the waiting time here will not include the initialization time of instance.
        is_ready_list = await asyncio.gather(*tasks, return_exceptions=True)
        return all(is_ready_list)
