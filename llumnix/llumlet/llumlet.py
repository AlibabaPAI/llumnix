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
from typing import List, Union, Iterable, Any
import time

import ray
import ray.actor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator, InstanceType
from llumnix.backends.backend_interface import BackendInterface, BackendType, EngineState
from llumnix.backends.utils import init_backend_engine
from llumnix.llumlet.migration_coordinator import MigrationCoordinator
from llumnix.server_info import ServerInfo
from llumnix.queue.queue_type import QueueType
from llumnix.arg_utils import InstanceArgs, LlumnixEngineArgs
from llumnix.ray_utils import get_instance_name, log_actor_ray_info
from llumnix.constants import CHECK_ENGINE_STATE_INTERVAL
from llumnix.metrics.timestamps import set_timestamp
from llumnix.metrics.llumlet_metrics import LlumletMetrics
from llumnix.utils import RequestIDType
from llumnix.constants import NUM_GPUS_VLLM_GPU_ACTOR, NUM_GPUS_VLLM_V1_GPU_ACTOR, NUM_GPUS_BLADELLM_GPU_ACTOR

logger = init_logger(__name__)


class Llumlet:
    def __init__(self,
                 instance_id: str,
                 instance_args: InstanceArgs,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 llumnix_engine_args: LlumnixEngineArgs) -> None:
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.instance_id = instance_id
        logger.info("Llumlet(instance_id={}, backend_type={})".format(self.instance_id, llumnix_engine_args.backend_type))
        self.instance_args: InstanceArgs = instance_args
        self.enable_migration = instance_args.enable_migration
        self.actor_name = get_instance_name(instance_id)

        self.instance_load_calculator = InstanceLoadCalculator(instance_args=self.instance_args)

        self.backend_engine: BackendInterface = init_backend_engine(self.instance_id,
                                                                    placement_group,
                                                                    request_output_queue_type,
                                                                    instance_args,
                                                                    llumnix_engine_args)
        if self.enable_migration:
            self.migration_coordinator = MigrationCoordinator(
                self.instance_id,
                self.backend_engine,
                llumnix_engine_args.backend_type,
                instance_args.request_migration_policy,
                instance_args.migration_last_stage_max_blocks,
                instance_args.migration_max_stages,
            )
        self.llumlet_metrics = LlumletMetrics()

        asyncio.create_task(self._check_engine_state_loop())

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]})"

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  instance_args: InstanceArgs,
                  placement_group: PlacementGroup,
                  request_output_queue_type: QueueType,
                  engine_args: LlumnixEngineArgs):
        backend_type = engine_args.backend_type
        assert backend_type in [BackendType.VLLM, BackendType.BLADELLM, BackendType.VLLM_V1, BackendType.SIM_VLLM], \
            f'unimplemented backend {BackendType}'
        # There could be some cuda related imports or codes inside the llm engine of llumlet, so we allocate gpu to llumlet.
        if backend_type == BackendType.VLLM:
            num_gpus = NUM_GPUS_VLLM_GPU_ACTOR
        elif backend_type == BackendType.VLLM_V1:
            num_gpus = NUM_GPUS_VLLM_V1_GPU_ACTOR
        elif backend_type == BackendType.BLADELLM:
            num_gpus = NUM_GPUS_BLADELLM_GPU_ACTOR
        else: # backend_type == BackendType.SIM_VLLM
            num_gpus = 0
        llumlet_class = ray.remote(num_cpus=1,
                                    num_gpus=num_gpus,
                                    name=get_instance_name(instance_id),
                                    namespace='llumnix',
                                    lifetime="detached")(cls).options(
                                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                                            placement_group=placement_group,
                                            placement_group_bundle_index=0,
                                            placement_group_capture_child_tasks=True
                                        )
                                    )
        llumlet = llumlet_class.remote(instance_id,
                                       instance_args,
                                       placement_group,
                                       request_output_queue_type,
                                       engine_args)

        return llumlet

    async def _check_engine_state_loop(self):
        while True:
            await asyncio.sleep(CHECK_ENGINE_STATE_INTERVAL)
            if self.backend_engine.state == EngineState.CRASHED:
                logger.error("Llumlet {} detected backend engine crashed. Stopping...".format(self.instance_id))
                # pylint: disable=protected-access
                self.backend_engine._stop_event.set()
                await asyncio.sleep(0)
                self.stop()

    def stop(self):
        if self.backend_engine.state == EngineState.RUNNING:
            self.backend_engine.stop()
        self_actor = ray.get_actor(name=self.actor_name, namespace="llumnix")
        ray.kill(self_actor)

    # TODO(KuilongCui): only the metrics-related information needs to be synchronously loaded for the manager
    def get_instance_info(self) -> InstanceInfo:
        instance_info: InstanceInfo = self.backend_engine.engine.instance_info
        instance_info.instance_type = self.instance_args.instance_type
        instance_info.enable_defrag = self.instance_args.enable_defrag
        self.instance_load_calculator.compute_instance_load(instance_info)
        return instance_info

    async def is_ready(self):
        await self.backend_engine.is_ready()
        return True

    async def get_instance_type(self) -> InstanceType:
        await self.backend_engine.is_ready()
        return self.instance_args.instance_type

    async def get_engine_disagg_inst_id(self) -> str:
        await self.backend_engine.is_ready()
        return self.backend_engine.engine_disagg_inst_id

    async def generate(self, request_id: RequestIDType, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        set_timestamp(server_info, 'llumlet_generate_timestamp', time.time())
        self.llumlet_metrics.llumlet_request_qps.increase(
            labels={"instance_id": self.instance_id}
        )
        await self.backend_engine.add_request(
            request_id, server_info, expected_steps, *args, **kwargs
        )

    async def abort(self, request_id: Union[RequestIDType, Iterable[RequestIDType]]) -> None:
        if isinstance(request_id, (str, int)):
            request_id = (request_id,)
        request_ids = set(request_id)
        await self.backend_engine.abort_request(request_ids)

    async def migrate_out(self, dst_instance_actor: ray.actor.ActorHandle, dst_instance_id: str) -> List[RequestIDType]:
        return await self.migration_coordinator.migrate_out(dst_instance_actor, dst_instance_id)

    def execute_engine_method(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return executor(*args, **kwargs)

    async def execute_engine_method_async(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return await executor(*args, **kwargs)

    def execute_migration_method(self, method, *args, **kwargs):
        executor = getattr(self.migration_coordinator, method)
        return executor(*args, **kwargs)

    async def execute_migration_method_async(self, method, *args, **kwargs):
        executor = getattr(self.migration_coordinator, method)
        return await executor(*args, **kwargs)
    
    async def call_engine_utility_async(self, method: str, *args) -> Any:
        # As per the hint, the target object containing utility functions
        # is self.backend_engine.engine.
        target_engine = self.backend_engine.engine

        try:
            executor = getattr(target_engine, method)
        except AttributeError as e:
            logger.error(f"Utility method '{method}' not found on backend engine {target_engine}.")
            raise e

        if asyncio.iscoroutinefunction(executor):
            return await executor(*args, **kwargs)
        else:
            return executor(*args, **kwargs)