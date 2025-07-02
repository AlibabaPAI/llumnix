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
from llumnix.utils import asyncio_wait_for_with_timeout, random_uuid

logger = init_logger(__name__)


class DPManager:
    def __init__(self,
                 instance_id: str,
                 entrypoints_args: EntrypointsArgs,
                 instance_args: InstanceArgs,
                 engine_args: LlumnixEngineArgs,
                 placement_group: PlacementGroup,):
        # TODO(shejiarui): check some args as vLLM here.
        self.instance_id = instance_id
        self.engine_args = engine_args.load_engine_args()
        dp_args: VLLMV1DPArgs = engine_args.get_dp_args()
        self.dp_size = dp_args.dp_size
        self.dp_size_local = dp_args.dp_size_local
        self.dp_address = dp_args.dp_address
        self.dp_rpc_port = dp_args.dp_rpc_port

        self.scaler = ray.get_actor(get_scaler_name(), namespace="llumnix")
        self.manager = ray.get_actor(get_manager_name(), namespace="llumnix")

        self._init_instances_and_servers(
            entrypoints_args, instance_args, engine_args, placement_group)

    @classmethod
    def from_args(cls,
                  instance_id: str,
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
            entrypoints_args,
            instance_args,
            engine_args,
            placement_group,
        )
        return dp_manager

    def _init_instances_and_servers(
            self,
            entrypoints_args: EntrypointsArgs,
            instance_args: InstanceArgs,
            engine_args: LlumnixEngineArgs,
            placement_group: PlacementGroup,):
        self.instance_ids: List[str] = []
        self.instances: List[ray.actor.ActorHandle] = []
        self.servers: List[ray.actor.ActorHandle] = []

        request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)

        for _ in range(self.dp_size):
            new_instance_id = random_uuid()
            self.instance_ids.append(new_instance_id)
                
            # TODO(shejiarui): how to get local_dp_ranks?
            # Vars below will be used in DPEngineCoreActor, 
            # need to implement the class.
            # on_head_node = i < self.dp_size_local
            # local_index = local_dp_ranks[i]

            # Pull up Llumlets
            instance = Llumlet.from_args(
                new_instance_id,
                instance_args,
                placement_group,
                request_output_queue_type,
                engine_args,
            )
            self.instances.append(instance)
            # Pull up APIServerActor
            server = APIServerActorVLLMV1.from_args(
                NUM_GPUS_VLLM_V1_GPU_ACTOR,
                new_instance_id,
                placement_group,
                entrypoints_args,
                engine_args,
                self.scaler,
                self.manager,
                instance,
            )
            self.servers.append(server)

        asyncio.create_task(
            self._done_scale_up(placement_group)
        )

    async def _done_scale_up(self,
                             placement_group: PlacementGroup,
                             instance_type: InstanceType = None,):
        for i in range(self.dp_size):
            instance_id = self.instance_ids[i]
            try:
                instance_ready = False
                await asyncio.wait_for(self.instances[i].is_ready.remote(), 
                                       timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT))
                instance_ready = True
                await asyncio.wait_for(self.servers[i].is_ready.remote(), 
                                       timeout=float(llumnix_envs.SERVER_READY_TIMEOUT))
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, ray.exceptions.RayActorError):
                    logger.warning("Failed to scale up DP instance {}, instance is dead.".format(instance_id))
                elif isinstance(e, asyncio.TimeoutError):
                    if not instance_ready:
                        logger.error("DP Instance {} is not ready in {} seconds.".format(instance_id, 
                                                                                         float(llumnix_envs.INSTANCE_READY_TIMEOUT)))
                    else:
                        logger.error("DP Server {} is not ready in {} seconds.".format(instance_id, 
                                                                                       float(llumnix_envs.SERVER_READY_TIMEOUT)))
                else:
                    logger.exception("Error in dpmanager(instance_id: {}) done_scale_up.".format(self.instance_id))
                await self.clear_dp_ray_resources()
        await asyncio_wait_for_with_timeout(
            self.manager.scale_up.remote(
                self.instance_ids, 
                self.instances, 
                instance_type, 
                placement_group, 
                self.servers)
        )
        logger.info()

    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    async def clear_dp_ray_resources(self):
        for ins_id in self.instance_ids:
            # There could be multiple clear_instance_ray_resources calls for one error instance,
            # so the kill operations could be failed if it is not the first attempt to kill.
            await kill_server(ins_id)
            await kill_instance(ins_id)
        remove_placement_group(self.instance_id)

    def is_ready(self):
        return False
