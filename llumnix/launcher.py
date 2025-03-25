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
from typing import Callable, List, Tuple

import ray
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
from llumnix.utils import (initialize_placement_group,
                           get_manager_name, get_server_name,
                           get_data_from_ray_internal_kv, put_data_to_ray_internal_kv,
                           load_engine_args)
from llumnix.internal_config import PDDConfig


logger = init_logger(__name__)


class Launcher:
    def __init__(self,
                 global_scheduler: GlobalScheduler,
                 enable_port_increment: bool,
                 enable_port_offset_store: bool,
                 load_registered_service: bool,
                 load_registered_service_path: str,
                 pdd_config: PDDConfig):
        self.global_scheduler = global_scheduler
        self.enable_port_increment = enable_port_increment
        self.enable_port_offset_store = enable_port_offset_store
        self.pdd_config = pdd_config
        self.load_registered_service = load_registered_service
        self.load_registered_service_path = load_registered_service_path

        self.manager_actor_handle = None

        if enable_port_increment:
            self.port_offset = 0
            if enable_port_offset_store:
                # TODO(s5u13b): Do not use ray interval kv.
                value = get_data_from_ray_internal_kv("manager.port_offset")
                self.port_offset = int(value)

        if self.load_registered_service:
            self.engine_args_dict = {}
            if not self.pdd_config.enable_pd_disagg and not self.pdd_config.enable_engine_pd_disagg:
                instance_type_list = ['no_constraints']
            else:
                instance_type_list = ['prefill', 'decode']
            for instance_type in instance_type_list:
                self.engine_args_dict[instance_type] = load_engine_args(instance_type, self.load_registered_service_path)

        self.inflight_num_prefill_instance = 0
        self.inflight_num_decode_instance = 0

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
            seperate_gpu_groups = backend_type != BackendType.BLADELLM
            placement_group = initialize_placement_group(placement_group_name, num_cpus=3+int(init_server),
                                                         num_gpus=world_size, detached=True, block=block,
                                                         seperate_gpu_groups=seperate_gpu_groups)
        else:
            placement_group = initialize_placement_group(placement_group_name, num_cpus=2+int(init_server),
                                                         num_gpus=0, detached=True, block=block)

        return placement_group

    def init_server_and_instance(self,
                                 instance_id: str,
                                 entrypoints_args: EntrypointsArgs,
                                 instance_args: InstanceArgs,
                                 engine_args,
                                 backend_type: BackendType,
                                 placement_group: PlacementGroup,
                                 scale_up_callback: Callable = None,
                                 scale_down_callback: Callable = None):
        async def done_scale_up(instance_args: InstanceArgs, entrypoint_args: EntrypointsArgs):
            try:
                if not self.manager_actor_handle:
                    self.manager_actor_handle = ray.get_actor(get_manager_name(), namespace="llumnix")
                await instance.is_ready.remote()
                await server.run.remote(self.manager_actor_handle, instance_id, instance)
                self.inflight_num_prefill_instance -= 1 if instance_args.instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode_instance -= 1 if instance_args.instance_type == InstanceType.DECODE else 0
                if scale_up_callback:
                    # manager.scale_up will be called here after the instance is ready
                    scale_up_callback(instance_id, instance, instance_args)
                logger.info("Init server and instance done, instance_id: {}, instance_type: {}, "
                            "api_server_port: {}, request_output_queue_port: {}".format(
                                instance_id, instance_args.instance_type,
                                entrypoint_args.port, entrypoint_args.request_output_queue_port))
            # pylint: disable=broad-except
            except Exception as e:
                self.inflight_num_prefill_instance -= 1 if instance_args.instance_type == InstanceType.PREFILL else 0
                self.inflight_num_decode_instance -= 1 if instance_args.instance_type == InstanceType.DECODE else 0
                logger.error("Unexpected exception occurs: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))
                scale_down_callback(instance_id)

        request_output_queue_type = QueueType(entrypoints_args.request_output_queue_type)
        next_instance_args = self._get_next_instance_args(instance_args)
        next_entrypoints_args = self._get_next_entrypoints_args(entrypoints_args)
        next_engine_args = self._get_next_engine_args(engine_args, next_instance_args.instance_type)
        instance = self.init_instance(instance_id, next_instance_args, placement_group,
                                      request_output_queue_type, backend_type, next_engine_args)
        server = self.init_server(get_server_name(instance_id), placement_group, next_entrypoints_args)

        self.inflight_num_prefill_instance += 1 if next_instance_args.instance_type == InstanceType.PREFILL else 0
        self.inflight_num_decode_instance += 1 if next_instance_args.instance_type == InstanceType.DECODE else 0
        asyncio.create_task(done_scale_up(next_instance_args, next_entrypoints_args))

    def init_server(self,
                    server_name: str,
                    placement_group: PlacementGroup,
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

    def _get_next_instance_args(self, instance_args: InstanceArgs) -> InstanceArgs:
        if not self.pdd_config.enable_pd_disagg and not self.pdd_config.enable_engine_pd_disagg:
            return instance_args

        next_instance_args: InstanceArgs = copy.deepcopy(instance_args)
        cur_num_prefill = len(self.global_scheduler.dispatch_scheduler.available_dispatch_instance_set)
        cur_num_decode = len(self.global_scheduler.instance_id_set -
                                self.global_scheduler.dispatch_scheduler.available_dispatch_instance_set)
        next_instance_args.instance_type = self._get_next_instance_type(cur_num_prefill, cur_num_decode, self.pdd_config.pd_ratio)

        return next_instance_args

    def _get_next_entrypoints_args(self, entrypoints_args: EntrypointsArgs) -> EntrypointsArgs:
        if not self.enable_port_increment:
            return entrypoints_args

        next_entrypoints_args = copy.deepcopy(entrypoints_args)
        if self.enable_port_increment:
            next_entrypoints_args.port += self.port_offset
            next_entrypoints_args.request_output_queue_port += self.port_offset
            self.port_offset += 1
            if self.enable_port_offset_store:
                put_data_to_ray_internal_kv("manager.port_offset", self.port_offset)

        return next_entrypoints_args

    def _get_next_engine_args(self, engine_args, instance_type: str):
        if not self.load_registered_service:
            return engine_args

        new_engine_args = self.engine_args_dict[instance_type]

        return new_engine_args

    def _get_next_instance_type(self,
                                cur_num_prefill: int,
                                cur_num_decode: int,
                                pd_ratio: List[int]) -> str:
        if not self.pdd_config.enable_pd_disagg and not self.pdd_config.enable_engine_pd_disagg:
            return InstanceType.NO_CONSTRAINTS

        # Note: There are no instances simultaneously in inflight_num_prefill and cur_num_prefill as
        # inflight_num will decrease before scaling up the instances. The same applies to num_decode.
        total_num_prefill = self.inflight_num_prefill_instance + cur_num_prefill
        total_num_decode = self.inflight_num_decode_instance + cur_num_decode

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
