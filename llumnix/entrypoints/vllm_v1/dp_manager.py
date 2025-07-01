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

import ray
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
from llumnix.ray_utils import (get_scaler_name, get_manager_name, get_dpmanager_name)
from llumnix.instance_info import InstanceType
from llumnix.entrypoints.vllm_v1.api_server_actor import APIServerActorVLLMV1
from llumnix.constants import NUM_GPUS_VLLM_V1_GPU_ACTOR

logger = init_logger(__name__)


class DPManager:
    def __init__(self,
                 entrypoints_args: EntrypointsArgs,
                 engine_args: LlumnixEngineArgs):
        # TODO(shejiarui): check some args as vLLM here.
        self.engine_args = engine_args.load_engine_args()
        dp_args: VLLMV1DPArgs = engine_args.get_dp_args()
        self.dp_size = dp_args.dp_size
        self.dp_size_local = dp_args.dp_size_local
        self.dp_address = dp_args.dp_address
        self.dp_rpc_port = dp_args.dp_rpc_port

        await self._init_instances_and_servers()

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  entrypoints_args: EntrypointsArgs,
                  engine_args: LlumnixEngineArgs,
                  ) -> "DPManager":
        dp_manager_class = ray.remote(
            num_cpus=1,
            name=get_dpmanager_name(instance_id),
            namespace="llumnix",
        )(cls)
        dp_manager = dp_manager_class.remote(
            entrypoints_args,
            engine_args,
        )
        return dp_manager

    async def _init_instances_and_servers(
            self,
            instance_id: str,
            entrypoints_args: EntrypointsArgs,
            instance_args: InstanceArgs,
            engine_args: LlumnixEngineArgs,
            placement_group: PlacementGroup,
            instance_type: InstanceType = None):
        # Pull up Llumlets
        # self.local_llumlets: List[ray.actor.ActorHandle] = []
        # self.remote_llumlets: List[ray.actor.ActorHandle] = []
        for i in range(self.dp_size):
            on_head_node = i < self.dp_size_local
            # FIXME(shejiarui)
            local_index = local_dp_ranks[i]

            instance = Llumlet.from_args(
                instance_id,
                instance_args,
                placement_group,
                request_output_queue_type,
                engine_args,
            )
            await asyncio.wait_for(instance.is_ready.remote(), 
                                   timeout=float(llumnix_envs.INSTANCE_READY_TIMEOUT))

        # Pull up APIServerActor
        scaler = ray.get_actor(get_scaler_name(), namespace="llumnix")
        manager = ray.get_actor(get_manager_name(), namespace="llumnix")
        for i in range(self.dp_size):
            api_server = APIServerActorVLLMV1.from_args(
                NUM_GPUS_VLLM_V1_GPU_ACTOR,
                instance_id,
                placement_group,
                entrypoints_args,
                engine_args,
                scaler,
                manager,
                instance,
            )

    def is_ready(self):
        return False
