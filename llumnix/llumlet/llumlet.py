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
import traceback
from typing import List, Union, Iterable
import time

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, BackendType, EngineState
from llumnix.backends.utils import init_backend_engine, get_engine_world_size
from llumnix.llumlet.migration_coordinator import MigrationCoordinator, MigrationStatus
from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler
from llumnix.server_info import ServerInfo
from llumnix.internal_config import MigrationConfig
from llumnix.queue.queue_type import QueueType
from llumnix.llumlet.request import LlumnixRequest, RequestStatus
from llumnix.utils import get_instance_name

logger = init_logger(__name__)

CHECK_ENGINE_STATE_INTERVAL = 1.0


class Llumlet:
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 backend_type: BackendType,
                 engine_args,
                 profiling_result_file_path: str = None) -> None:
        try:
            self.node_id = ray.get_runtime_context().get_node_id()
            logger.info("Llumlet backend type: {}".format(backend_type))
            self.instance_id = instance_id
            self.actor_name = get_instance_name(instance_id)
            self.backend_engine: BackendInterface = init_backend_engine(instance_id,
                                                                        placement_group,
                                                                        request_output_queue_type,
                                                                        migration_config,
                                                                        backend_type,
                                                                        engine_args,
                                                                        profiling_result_file_path)
            self.migration_coordinator = MigrationCoordinator(self.backend_engine,
                                                              migration_config.last_stage_max_blocks,
                                                              migration_config.max_stages)
            self.migration_scheduler = LocalMigrationScheduler(migration_config.request_migration_policy,
                                                               self.backend_engine)
            self.log_requests = True

            asyncio.create_task(self._check_engine_state_loop())
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("failed to initialize Llumlet: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
            raise

    def __repr__(self):
        return f"{self.__class__.__name__}(node_id={self.node_id[:5]},instance_id={self.instance_id[:5]})"

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  placement_group: PlacementGroup,
                  request_output_queue_type: QueueType,
                  migration_config: MigrationConfig,
                  backend_type: BackendType,
                  engine_args,
                  profiling_result_file_path: str = None):
        try:
            assert backend_type in [backend_type.VLLM, backend_type.BLADELLM, backend_type.SIM_VLLM], \
                f'unimplemented backend {backend_type}'
            num_gpus = 0
            if backend_type == backend_type.BLADELLM:
                world_size = get_engine_world_size(engine_args, backend_type)
                num_gpus = world_size
            # TODO(s5u13b): Check the max_concurrency.
            llumlet_class = ray.remote(num_cpus=1,
                                       num_gpus=num_gpus,
                                       name=get_instance_name(instance_id),
                                       namespace='llumnix',
                                       max_concurrency=4,
                                       lifetime="detached")(cls).options(
                                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                                placement_group=placement_group,
                                                placement_group_bundle_index=0,
                                                placement_group_capture_child_tasks=True
                                            )
                                        )
            llumlet = llumlet_class.remote(instance_id,
                                           placement_group,
                                           request_output_queue_type,
                                           migration_config,
                                           backend_type,
                                           engine_args,
                                           profiling_result_file_path)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("failed to initialize Llumlet: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
            raise

        return llumlet

    async def _check_engine_state_loop(self):
        while True:
            await asyncio.sleep(CHECK_ENGINE_STATE_INTERVAL)
            if self.backend_engine.state == EngineState.CRASHED:
                logger.error("Llumlet ({}) detected backend engine crashed. Stopping...".format(self.instance_id))
                # pylint: disable=protected-access
                self.backend_engine._stop_event.set()
                await asyncio.sleep(0)
                self_actor = ray.get_actor(self.actor_name)
                ray.kill(self_actor)

    async def migrate_out(self, dst_instance_name: str) -> List[str]:
        migrate_out_requests = self.migration_scheduler.get_migrate_out_requests()
        if len(migrate_out_requests) == 0:
            return []

        for migrate_out_request in migrate_out_requests:
            migrate_out_request.is_migrating = True

        migrated_request_list = []
        for migrate_out_request in migrate_out_requests:
            migrated_request = await self._migrate_out_one_request(migrate_out_request, dst_instance_name)
            migrated_request_list.extend(migrated_request)
            if len(migrated_request) == 0 and migrate_out_request.eom:
                break
        return migrated_request_list

    async def _migrate_out_one_request(self, migrate_out_request: LlumnixRequest, dst_instance_name: str) -> List[LlumnixRequest]:
        try:
            t0 = time.time()
            migrate_in_ray_actor = ray.get_actor(dst_instance_name, namespace='llumnix')
            dst_instance_id = dst_instance_name[len("instance_"):]
            logger.info("{}->{} begin migrate out".format(self.instance_id, dst_instance_id))
            migrated_request = []

            if migrate_out_request.status == RequestStatus.RUNNING:
                migrate_out_request.migration_start_time = time.time()
                status = await self.migration_coordinator.migrate_out_running_request(migrate_in_ray_actor, migrate_out_request)
            elif migrate_out_request.status == RequestStatus.WAITING:
                migrate_out_request.migration_start_time = time.time()
                status = await self.migration_coordinator.migrate_out_waiting_request(migrate_in_ray_actor, migrate_out_request)
            else:
                return migrated_request

            if status == MigrationStatus.FINISHED:
                await migrate_in_ray_actor.execute_engine_method.remote("commit_dst_request", migrate_out_request)
                self.backend_engine.free_src_request(migrate_out_request)
                self.backend_engine.remove_migrating_out_request_last_stage(migrate_out_request)
                migrated_request.append(migrate_out_request.request_id)
            else: # ABORTED_SRC or ABORTED_DST
                migrate_out_request.reset_migration_args_src()
                migrate_out_request.reset_status()
                # If dst aborts itself, dst proactively frees the pre allocated cache in migrate_in_pre_alloc.
                if status == MigrationStatus.ABORTED_SRC:
                    await migrate_in_ray_actor.execute_migration_method.remote("free_dst_pre_alloc_cache", migrate_out_request.request_id)
            t1 = time.time()
            logger.info("{}->{} migrate done, migrate request {}, migration status: {}, len: {} blocks, cost: {} ms" \
                        .format(self.instance_id, dst_instance_id, migrated_request, status, \
                                sum(migrate_out_request.stage_num_blocks_list), (t1 - t0)*1000))
        except ray.exceptions.RayActorError:
            logger.info("instance {} is dead".format(dst_instance_name[len("instance_"):]))
            raise
        # pylint: disable=W0703
        except Exception as e:
            logger.error("unexpected exception occurs: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
            raise
        return migrated_request

    def get_instance_info(self) -> InstanceInfo:
        return self.backend_engine.engine.instance_info

    def is_ready(self) -> bool:
        return True

    def get_all_request_ids(self) -> List[str]:
        return self.backend_engine.get_all_request_ids()

    def generate(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        if hasattr(server_info, 'request_timestamps'):
            server_info.request_timestamps.llumlet_generate_timestamp = time.time()
        self.backend_engine.add_request(request_id, server_info, expected_steps, *args, **kwargs)

    def abort(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        return self.backend_engine.abort_request(request_ids)

    def clear_migration_states(self, is_migrate_in: bool) -> None:
        logger.info("instance {} clear_migration_states, is_migrate_in: {}".format(self.instance_id, is_migrate_in))
        if is_migrate_in:
            # If migrate out instance dies during migration, migrate in instance directly free the pre-allocated cache of the migrating in request.
            logger.info("clear_migration_states: free_dst_pre_alloc_cache")
            self.backend_engine.free_dst_pre_alloc_cache()
        else:
            # If migrate in instance dies during migration, migrate out instance should add the migrating out request in last stage.
            # back to the running request queue.
            migrating_out_requests_last_stage = self.backend_engine.pop_migrating_out_requests_last_stage()
            for backend_request in migrating_out_requests_last_stage:
                logger.info("clear_migration_states: add request {} back to engine".format(backend_request.request_id))
                assert backend_request.status in [RequestStatus.WAITING_MIGRATING, RequestStatus.RUNNING_MIGRATING], \
                    "The status of request in migrating_out_requests_last_stage should be \
                     RequestStatus.WAITING_MIGRATING or RequestStatus.RUNNING_MIGRATING"
                if backend_request.status == RequestStatus.RUNNING_MIGRATING:
                    self.backend_engine.add_running_request(backend_request)
                else: # WAITING_MIGRATING
                    self.backend_engine.add_waiting_request(backend_request)

    def execute_migration_method(self, method, *args, **kwargs):
        executor = getattr(self.migration_coordinator, method)
        return executor(*args, **kwargs)

    def execute_engine_method(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return executor(*args, **kwargs)
