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

from typing import List

import ray
import ray.actor
from ray.util.placement_group import PlacementGroup
from ray.util.state import list_actors

from llumnix.logging.logger import init_logger
from llumnix.arg_utils import (
    EntrypointsArgs,
    InstanceArgs,
    LlumnixEngineArgs,
)
from llumnix.llumlet.llumlet import Llumlet
from llumnix.ray_utils import get_scaler_name, get_manager_name, get_dpmanager_name
from llumnix.queue.queue_type import QueueType
from llumnix.entrypoints.vllm_v1.api_server_actor import APIServerActorVLLMV1
from llumnix.constants import NUM_GPUS_VLLM_V1_GPU_ACTOR

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
        self.dp_size = engine_args.get_dp_size()

        self.scaler = ray.get_actor(get_scaler_name(), namespace="llumnix")
        self.manager = ray.get_actor(get_manager_name(), namespace="llumnix")

        self.port_offset = 0
        self.client_index_offset = 0

        self.instance_ids = new_instance_ids
        self.instances: List[ray.actor.ActorHandle] = []
        self.servers: List[ray.actor.ActorHandle] = []

        instance_filter = [("class_name", "=", "Llumlet"), ("placement_group_id", "=", str(placement_group.id))]
        server_filter = [("class_name", "=", "APIServerActor"), ("placement_group_id", "=", str(placement_group.id))]
        try:
            # Try to find instances and servers with class name and placement group
            # in case that the dp_manager is restarted.
            instance_actors = list_actors(filters=instance_filter)
            server_actors = list_actors(filters=server_filter)
            found_instances = len(instance_actors)
            found_servers = len(server_actors)

            if found_instances == 0 and found_servers == 0:
                # No instances or servers found, create them
                self._init_instances_and_servers()
            elif found_instances == self.dp_size and found_servers == self.dp_size:
                for actor in instance_actors:
                    self.instances.append(ray.get_actor(actor.actor_id))
                for actor in server_actors:
                    self.servers.append(ray.get_actor(actor.actor_id))
            else:
                raise(RuntimeError("DPManager restarted but found not exactly dp_size instances and servers. "
                                   "Restart the whole DP group."))

            ray.get([server.is_ready.remote() for server in self.servers])
            ray.get([instance.is_ready.remote() for instance in self.instances])
            self.manager.scale_up.remote(self.instance_ids, self.instances, [None] * self.dp_size)
        # pylint: disable=broad-except
        except Exception as e:
            # TODO(shejiarui): scale down if dp_manager restart failed.
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
        request_output_queue_type = QueueType(self.entrypoints_args.request_output_queue_type)
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
                self.instance_args,
                self.engine_args,
                self.scaler,
                self.manager,
                instance,
                bundle_index=rank
            )
            self.servers.append(server)
