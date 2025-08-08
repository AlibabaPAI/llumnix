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
from typing import List, Tuple, Union, Iterable, Any, Optional

import ray
import ray.actor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator
from llumnix.backends.backend_interface import BackendBaseInterface
from llumnix.backends.utils import init_backend_engine, EngineState
from llumnix.llumlet.migration_coordinator import MigrationCoordinator
from llumnix.request_processing_context import RequestProcessingContext
from llumnix.queue.queue_type import QueueType
from llumnix.arg_utils import InstanceArgs, LlumnixEngineArgs
from llumnix.ray_utils import (
    LlumnixActor,
    log_actor_ray_info,
    get_llumnix_actor_handle,
    get_llumnix_actor_name,
)
from llumnix.constants import CHECK_ENGINE_STATE_INTERVAL
from llumnix.metrics.llumlet_metrics import LlumletMetrics
from llumnix.utils import (MigrationType, RequestIDType, BackendType, InstanceContext,
                           InstanceType, UnitStatus)
from llumnix.constants import NUM_GPUS_VLLM_GPU_ACTOR, NUM_GPUS_BLADELLM_GPU_ACTOR

logger = init_logger(__name__)


class Llumlet:
    def __init__(
        self,
        instance_id: str,
        instance_args: InstanceArgs,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        llumnix_engine_args: LlumnixEngineArgs,
        dp_rank: int = 0,
        dp_rank_local: Optional[int] = None
    ) -> None:
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.instance_id = instance_id
        self.instance_type = instance_args.instance_type
        logger.info("Llumlet(instance_id={}, backend_type={}, instance_type={})".format(
            self.instance_id, llumnix_engine_args.backend_type, self.instance_type))
        self.instance_args: InstanceArgs = instance_args
        self.placement_group = placement_group
        self.actor_handle = get_llumnix_actor_handle(LlumnixActor.INSTANCE, instance_id)

        self.instance_load_calculator = InstanceLoadCalculator(instance_args=instance_args)

        self.backend_engine: BackendBaseInterface = init_backend_engine(
            instance_id,
            placement_group,
            request_output_queue_type,
            instance_args,
            llumnix_engine_args,
            dp_rank,
            dp_rank_local,
        )

        self.enable_routine_migration = instance_args.enable_routine_migration
        self.enable_pre_stop_migration = instance_args.enable_pre_stop_migration
        if self.enable_routine_migration or self.enable_pre_stop_migration:
            self.migration_coordinator = MigrationCoordinator(
                self.instance_id,
                self.backend_engine,
                llumnix_engine_args.backend_type,
                instance_args.max_migration_concurrency,
                instance_args.request_migration_policy,
                instance_args.migration_last_stage_max_blocks,
                instance_args.migration_max_stages,
                instance_args.enable_engine_migration_interface,
            )
        self.llumlet_metrics = LlumletMetrics()
        self.unit_status = UnitStatus.HEALTHY

        asyncio.create_task(self._check_engine_state_loop())

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]}, instance_type={self.instance_type})"

    @classmethod
    def from_args(
        cls,
        instance_id: str,
        instance_args: InstanceArgs,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        engine_args: LlumnixEngineArgs,
        dp_rank: int = 0,
        dp_rank_local: Optional[int] = None
    ) -> ray.actor.ActorHandle:
        backend_type = engine_args.backend_type
        assert backend_type in [BackendType.VLLM, BackendType.BLADELLM, BackendType.VLLM_V1, BackendType.SIM_VLLM], \
            f'unimplemented backend {BackendType}'
        # There could be some cuda related imports or codes inside the llm engine of llumlet, so we allocate gpu to llumlet.
        bundle_index = 0
        if backend_type == BackendType.VLLM:
            num_gpus = NUM_GPUS_VLLM_GPU_ACTOR
        elif backend_type == BackendType.VLLM_V1:
            num_gpus = engine_args.get_world_size()
            if dp_rank > 0:
                bundle_index = dp_rank
        elif backend_type == BackendType.BLADELLM:
            num_gpus = NUM_GPUS_BLADELLM_GPU_ACTOR
        else: # backend_type == BackendType.SIM_VLLM
            num_gpus = 0
        llumlet_class = ray.remote(
            num_cpus=1,
            num_gpus=num_gpus,
            name=get_llumnix_actor_name(LlumnixActor.INSTANCE, instance_id),
            namespace='llumnix',
            lifetime="detached"
        )(cls).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=True
            )
        )
        llumlet = llumlet_class.remote(
            instance_id,
            instance_args,
            placement_group,
            request_output_queue_type,
            engine_args,
            dp_rank,
            dp_rank_local
        )

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
        ray.kill(self.actor_handle)

    # TODO(KuilongCui): only the metrics-related information needs to be synchronously loaded for the manager
    def get_instance_info(self) -> InstanceInfo:
        instance_info: InstanceInfo = self.backend_engine.get_instance_info()
        instance_info.instance_type = self.instance_args.instance_type
        instance_info.unit_status = self.unit_status
        instance_info.unit_id = self.instance_id.split("_")[0]
        instance_info.enable_defrag = self.instance_args.enable_defrag
        self.instance_load_calculator.compute_instance_load(instance_info)
        if self.enable_routine_migration or self.enable_pre_stop_migration:
            instance_info.has_migration_slot = self.migration_coordinator.has_migration_slot()
            instance_info.is_migrating = self.migration_coordinator.is_migrating()
        return instance_info

    async def is_ready(self):
        await self.backend_engine.is_ready()
        return self.unit_status == UnitStatus.HEALTHY

    def set_unit_status(self, status: UnitStatus) -> None:
        if status == UnitStatus.TERMINATING and not self.enable_pre_stop_migration:
            logger.info("Instance will be TERMINATING and pre-stop migration is disabled, so directly stop it.")
            status = UnitStatus.STOPPED
        logger.info("Llumlet(instance_id={}, instance_type={}) change unit_status {} -> {}.".format(
            self.instance_id, self.instance_args.instance_type, self.unit_status, status))
        self.unit_status = status

    def get_unit_status(self) -> UnitStatus:
        return self.unit_status

    async def get_instance_type(self) -> InstanceType:
        await self.backend_engine.is_ready()
        return self.instance_args.instance_type

    async def get_engine_context(self) -> InstanceContext:
        await self.backend_engine.is_ready()
        return self.backend_engine.get_engine_context()

    async def get_engine_context_and_instance_info(self) -> Tuple[InstanceContext, InstanceInfo]:
        instance_context = await self.get_engine_context()
        instance_info = self.backend_engine.get_instance_info()
        return instance_context, instance_info

    async def generate(
        self,
        request_id: RequestIDType,
        request_processing_context: RequestProcessingContext,
        expected_steps: int,
        *args,
        **kwargs,
    ) -> None:
        request_processing_context.add_trace_timeline("llumlet_generate_timestamp")
        self.llumlet_metrics.llumlet_request_qps.increase(
            labels={"instance_id": self.instance_id}
        )
        await self.backend_engine.add_request(
            request_id, request_processing_context, expected_steps, *args, **kwargs
        )

    async def abort(self, request_id: Union[RequestIDType, Iterable[RequestIDType]]) -> None:
        if isinstance(request_id, (str, int)):
            request_id = (request_id,)
        request_ids = set(request_id)
        await self.backend_engine.abort_request(request_ids)

    async def migrate_out(
        self,
        dst_instance_actor: ray.actor.ActorHandle,
        dst_instance_context: InstanceContext,
        migration_type: Optional[MigrationType] = None
    ) -> None:
        async def _inner_migrate_out(
            dst_instance_actor: ray.actor.ActorHandle,
            dst_instance_context: InstanceContext,
            migration_type: Optional[MigrationType] = None
        ) -> List[RequestIDType]:
            migrated_request_ids = await self.migration_coordinator.migrate_out(
                dst_instance_actor, dst_instance_context, migration_type)

            if migration_type == MigrationType.PRE_STOP_MIGRATION:
                if self.unit_status not in [UnitStatus.MIGRATING, UnitStatus.STOPPED]:
                    self.set_unit_status(UnitStatus.MIGRATING)
                instance_info: InstanceInfo = self.backend_engine.get_instance_info()
                num_running_requests = instance_info.num_running_requests
                num_waiting_requests = instance_info.num_waiting_requests

                if len(migrated_request_ids) > 0:
                    logger.info("Llumlet(instance_id={}, instance_type={}) is doing {}, left ruuning requests: {}, " \
                        "left waiting requests: {}.".format(self.instance_id, self.instance_args.instance_type,
                        migration_type, num_running_requests, num_waiting_requests))
                if num_running_requests == 0 and num_waiting_requests == 0:
                    if self.unit_status != UnitStatus.STOPPED:
                        self.set_unit_status(UnitStatus.STOPPED)
                        logger.info("Llumlet(instance_id={}, instance_type={}) is stopped.".format(
                            self.instance_id, self.instance_args.instance_type))

            return migrated_request_ids

        asyncio.create_task(_inner_migrate_out(dst_instance_actor, dst_instance_context, migration_type))

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
        """Call engine utility function for backend vLLM v1."""
        # As per the hint, the target object containing utility functions is self.backend_engine.engine.
        target_engine = self.backend_engine.engine

        try:
            executor = getattr(target_engine, method)
        except AttributeError as e:
            logger.error("Utility method '{}' not found on backend engine {}.".format(method, target_engine))
            raise e

        if asyncio.iscoroutinefunction(executor):
            return await executor(*args)
        return executor(*args)
