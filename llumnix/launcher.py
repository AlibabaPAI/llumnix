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
from typing import Callable, Dict, List, Tuple

import ray
from ray.util.state import list_placement_groups, list_actors
from ray.util.placement_group import PlacementGroup

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceType
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_type import QueueType
from llumnix.backends.backend_interface import BackendType
from llumnix.arg_utils import EntrypointsArgs, InstanceArgs
from llumnix.entrypoints.vllm.api_server_actor import APIServerActor
from llumnix.backends.utils import get_engine_world_size
from llumnix.utils import (remove_placement_group, get_manager_name, INSTANCE_NAME_PREFIX, get_instance_name,
                           SERVER_NAME_PREFIX, kill_server, kill_instance, get_actor_data_from_ray_internal_kv,
                           initialize_placement_group, get_server_name, put_actor_data_to_ray_internal_kv,
                           get_placement_group_name)

logger = init_logger(__name__)


class Launcher:
    def __init__(self, global_scheduler: GlobalScheduler, enable_port_increment: bool,
                 enable_port_offset_store: bool, enable_pd_disagg: bool,
                 enablde_engine_pd_disagg: bool, pd_ratio: List[int]):
        self.global_scheduler = global_scheduler
        self.enable_port_increment = enable_port_increment
        self.enable_port_offset_store = enable_port_offset_store
        self.enable_pd_disagg = enable_pd_disagg
        self.enablde_engine_pd_disagg = enablde_engine_pd_disagg
        self.pd_ratio = pd_ratio

        if enable_port_increment:
            self.port_offset = 0
            if enable_port_offset_store:
                value = get_actor_data_from_ray_internal_kv("manager", "port_offset")
                if value is not None:
                    self.port_offset = int(value)

        self.inflight_num_prefill = 0
        self.inflight_num_decode = 0

    def init_placement_group(self,
                             placement_group_name: str,
                             engine_args,
                             backend_type: BackendType,
                             init_server: bool = False,
                             block: bool = True) -> PlacementGroup:
        # num_cpus=2+(0/1), for Llumlet + AsyncPutQueueActor + (ApiServerActor)
        if not BackendType.is_sim_backend(backend_type):
            # num_gpus=world_size, for world_size Workers
            world_size = get_engine_world_size(engine_args, backend_type)
            placement_group = initialize_placement_group(placement_group_name, num_cpus=2+int(init_server),
                                                         num_gpus=world_size, detached=True, block=block)
        else:
            placement_group = initialize_placement_group(placement_group_name, num_cpus=2+int(init_server),
                                                         num_gpus=0, detached=True, block=block)

        return placement_group

    def get_instance_deployment_states(self, instance_id: str):
        pg_state = list_placement_groups(filters=[("name", "=", get_placement_group_name(instance_id))])
        pg_created = len(pg_state) == 1 and pg_state[0]["state"] == "CREATED"
        server_state = list_actors(filters=[("name", "=", get_server_name(instance_id))])
        server_alive = len(server_state) == 1 and server_state[0]["state"] == "ALIVE"
        instance_state = list_actors(filters=[("name", "=", get_instance_name(instance_id))])
        instance_alive = len(instance_state) == 1 and instance_state[0]["state"] == "ALIVE"

        return pg_created, server_alive, instance_alive

    def get_cluster_deployment(self) -> Tuple[Dict[str, PlacementGroup], Dict[str, APIServerActor], Dict[str, Llumlet]]:
        curr_pgs: Dict[str, PlacementGroup] = {}
        curr_servers: Dict[str, PlacementGroup] = {}
        curr_instances: Dict[str, Llumlet] = {}

        created_pg_states = list_placement_groups(filters=[("state", "=", "CREATED")])
        for created_pg_state in created_pg_states:
            instance_id = created_pg_state["name"].split("_")[-1]
            curr_pgs[instance_id] = ray.util.get_placement_group(created_pg_state["name"])

        alive_actor_states = list_actors(filters=[("state", "=", "ALIVE")])
        for alive_actor_state in alive_actor_states:
            if alive_actor_state["name"].startswith(SERVER_NAME_PREFIX):
                instance_id = alive_actor_state["name"].split("_")[-1]
                curr_servers[instance_id] = ray.get_actor(alive_actor_state["name"], namespace="llumnix")
            elif alive_actor_state["name"].startswith(INSTANCE_NAME_PREFIX):
                instance_id = alive_actor_state["name"].split("_")[-1]
                curr_instances[instance_id] = ray.get_actor(alive_actor_state["name"], namespace="llumnix")

        return curr_pgs, curr_servers, curr_instances

    def clear_instance_ray_resources(self, instance_id: str):
        if not remove_placement_group(instance_id):
            logger.debug("Failed to remove placement group {}.".format(instance_id))
        if not kill_server(instance_id):
            logger.debug("Failed to kill server {}.".format(instance_id))
        if not kill_instance(instance_id):
            logger.debug("Failed to kill instance {}.".format(instance_id))

    def _get_next_instance_type(self, cur_num_prefill, cur_num_decode, pd_ratio) -> str:
        instance_type = InstanceType.NO_CONSTRAINTS

        if self.enable_pd_disagg:
            # Note: There are no instances simultaneously in inflight_num_prefill and cur_num_prefill as
            # inflight_num will decrease before scaling up the instances. The same applies to num_decode.
            total_num_prefill = self.inflight_num_prefill + cur_num_prefill
            total_num_decode = self.inflight_num_decode + cur_num_decode

            if total_num_prefill == 0:
                instance_type = InstanceType.PREFILL
            elif total_num_decode == 0:
                instance_type = InstanceType.DECODE
            else:
                # compute distance if launch prefill or decode
                normal_distance = pd_ratio[0] - pd_ratio[1]

                base_num_ratio = min(total_num_prefill//pd_ratio[0], total_num_decode//pd_ratio[1])
                total_num_prefill = total_num_prefill - base_num_ratio * pd_ratio[0]
                total_num_decode = total_num_decode - base_num_ratio * pd_ratio[1]

                if total_num_prefill + total_num_decode == 0:
                    instance_type = InstanceType.PREFILL
                else:
                    distance_if_prefill = total_num_prefill + 1 - total_num_decode
                    distance_if_decode = total_num_prefill - (total_num_decode + 1)
                    gap_to_normal_if_prefill = abs(distance_if_prefill - normal_distance)
                    gap_to_normal_if_decode = abs(distance_if_decode - normal_distance)
                    instance_type = InstanceType.PREFILL if gap_to_normal_if_prefill <= gap_to_normal_if_decode \
                        else InstanceType.DECODE

        return instance_type

    def _get_next_instance_args(self, instance_args) -> InstanceArgs:
        assert not self.enablde_engine_pd_disagg, \
            "Currently not support engine based pd-disaggregation in global launch mode."

        next_instance_args: InstanceArgs = copy.deepcopy(instance_args)
        cur_num_prefill = len(self.global_scheduler.dispatch_scheduler.available_dispatch_instance_set)
        cur_num_decode = len(self.global_scheduler.instance_id_set -
                                self.global_scheduler.dispatch_scheduler.available_dispatch_instance_set)
        next_instance_args.instance_type = self._get_next_instance_type(cur_num_prefill, cur_num_decode, self.pd_ratio)
        return next_instance_args

    def _get_next_entrypoints_args(self, entrypoints_args: EntrypointsArgs) -> EntrypointsArgs:
        next_entrypoints_args = copy.deepcopy(entrypoints_args)
        if self.enable_port_increment:
            next_entrypoints_args.port += self.port_offset
            next_entrypoints_args.request_output_queue_port += self.port_offset
            self.port_offset += 1
            if self.enable_port_offset_store:
                put_actor_data_to_ray_internal_kv("manager", "port_offset", self.port_offset)
        return next_entrypoints_args

    def init_server_and_instance(self, instance_id: str, entrypoints_args: EntrypointsArgs,
                                 instance_args: InstanceArgs, engine_args, backend_type: BackendType,
                                 placement_group: PlacementGroup, instance_finish_cb: Callable = None,
                                 server_finish_cb: Callable = None):
        async def done_scale_up(instance_args: InstanceArgs, entrypoint_args: EntrypointsArgs):
            try:
                manager = ray.get_actor(get_manager_name(), namespace="llumnix")
                await instance.is_ready.remote()
                await server.run.remote(manager, instance_id, instance)
                self.inflight_num_prefill -= 1 if instance_args.instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode -= 1 if instance_args.instance_type == InstanceType.DECODE else 0
                if instance_finish_cb:
                    # manager.scale_up will be called here after the instance is ready
                    instance_finish_cb(instance_id, instance, instance_args)
                if server_finish_cb:
                    server_finish_cb(instance_id, server)
                logger.info("Launcher init_server_and_instance done, instance_id: {}, instance_type: {}, "
                            "api_server_port: {}, request_output_queue_port: {}".format(instance_id,
                            instance_args.instance_type, entrypoint_args.port,
                            entrypoint_args.request_output_queue_port))
            # pylint: disable=broad-except
            except Exception as e:
                self.inflight_num_prefill -= 1 if instance_args.instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode -= 1 if instance_args.instance_type == InstanceType.DECODE else 0
                logger.error("Unexpected exception occurs: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))
                self.clear_instance_ray_resources(instance_id)

        request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)
        next_instance_args = self._get_next_instance_args(instance_args)
        instance = self.init_instance(instance_id, next_instance_args, placement_group,
                                       request_output_queue_type, backend_type, engine_args)
        next_entrypoints_args = self._get_next_entrypoints_args(entrypoints_args)
        server = self.init_server(get_server_name(instance_id), placement_group, next_entrypoints_args)

        self.inflight_num_prefill += 1 if next_instance_args.instance_type == InstanceType.PREFILL else 0
        self.inflight_num_decode += 1 if next_instance_args.instance_type == InstanceType.DECODE else 0
        asyncio.create_task(done_scale_up(next_instance_args, next_entrypoints_args))

    def init_server(self, server_name: str, placement_group: PlacementGroup,
                    entrypoints_args: EntrypointsArgs) -> APIServerActor:
        fastapi_server = APIServerActor.from_args(server_name, placement_group, entrypoints_args)
        return fastapi_server

    def init_instance(self,
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
