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
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy

from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, BackendType, EngineState
from llumnix.backends.utils import init_backend_engine, initialize_placement_group
from llumnix.llumlet.migration_coordinator import MigrationCoordinator, MigrationStatus
from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler
from llumnix.server_info import ServerInfo
from llumnix.internal_config import MigrationConfig
from llumnix.queue.queue_type import QueueType

logger = init_logger(__name__)


class Llumlet:
    def __init__(self,
                 instance_id: str,
                 output_queue_type: QueueType,
                 backend_type: BackendType,
                 migration_config: MigrationConfig,
                 *args,
                 **kwargs) -> None:
        try:
            self.instance_id = instance_id
            self.actor_name = f"instance_{instance_id}"
            self.backend_engine: BackendInterface = init_backend_engine(self.instance_id,
                                                                        output_queue_type,
                                                                        backend_type,
                                                                        migration_config,
                                                                        *args,
                                                                        **kwargs)
            self.migration_coordinator = MigrationCoordinator(self.backend_engine,
                                                            migration_config.last_stage_max_blocks,
                                                            migration_config.max_stages)
            self.migration_scheduler = LocalMigrationScheduler(migration_config.request_migration_policy,
                                                            self.backend_engine)
            self.log_requests = True

            self.check_state_thread = asyncio.create_task(self.check_state())
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Failed to initialize llumlet: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))

    @classmethod
    def from_args(cls,
                  output_queue_type: QueueType,
                  disable_fixed_node_init_instance: bool,
                  detached: bool,
                  node_id: str,
                  instance_id: str,
                  backend_type: BackendType,
                  migration_config: MigrationConfig,
                  world_size: int,
                  *args,
                  **kwargs):
        lifetime = "detached" if detached else None
        assert backend_type in [backend_type.VLLM, backend_type.SIM_VLLM, backend_type.BLADELLM], f'unimplemented backend {backend_type}'
        actor_name = f"instance_{instance_id}"
        if backend_type in [backend_type.VLLM, backend_type.BLADELLM]:
            if disable_fixed_node_init_instance:
                # TODO(s5u13b): Support placement_group lifetime management when the migration backend is gloo.
                placement_group = initialize_placement_group(world_size, detached=detached)
                kwargs["placement_group"] = placement_group
                engine_class = ray.remote(num_cpus=1,
                                          name=actor_name,
                                          namespace='llumnix',
                                          max_concurrency=4,
                                          lifetime=lifetime)(cls).options(
                                                scheduling_strategy=PlacementGroupSchedulingStrategy(
                                                    placement_group=placement_group,
                                                    placement_group_bundle_index=0,
                                                )
                                            )
            else:
                kwargs["node_id"] = node_id
                engine_class = ray.remote(num_cpus=1,
                                          name=actor_name,
                                          namespace='llumnix',
                                          max_concurrency=4,
                                          lifetime=lifetime)(cls).options(
                                                scheduling_strategy=NodeAffinitySchedulingStrategy(
                                                    node_id=node_id,
                                                    soft=False,
                                                )
                                            )
        else: # backend_type == backend_type.SIM_VLLM:
            kwargs["node_id"] = node_id
            engine_class = ray.remote(num_cpus=1,
                                      name=actor_name,
                                      namespace='llumnix',
                                      max_concurrency=4,
                                      lifetime=lifetime)(cls).options(
                                            scheduling_strategy=NodeAffinitySchedulingStrategy(
                                                node_id=node_id,
                                                soft=False,
                                            )
                                        )
        llumlet = engine_class.remote(instance_id, output_queue_type, backend_type, migration_config, *args, **kwargs)
        return llumlet

    async def check_state(self):
        while True:
            await asyncio.sleep(1)
            if self.backend_engine.state == EngineState.CRASHED:
                logger.warning("llumlet ({}) detected backend engine crashed. Stopping...".format(self.instance_id))
                # pylint: disable=protected-access
                self.backend_engine._stop_event.set()
                self_actor = ray.get_actor(self.actor_name)
                ray.kill(self_actor)

    async def migrate_out(self, dst_instance_name: str, num_requests: int) -> List[str]:
        try:
            migrate_in_ray_actor = ray.get_actor(dst_instance_name, namespace='llumnix')
            dst_instance_id = dst_instance_name[len("instance_"):]
            migrated_request_list = []
            continue_migrate = True
            while continue_migrate and len(migrated_request_list) < num_requests:
                t0 = time.time()
                migrate_out_request = self.migration_scheduler.get_migrate_out_request()
                if migrate_out_request is not None:
                    logger.info("migrate_out {}".format(migrate_out_request.request_id))
                if migrate_out_request is None:
                    return migrated_request_list
                logger.info("{}->{} begin migrate out {}".format(self.instance_id, dst_instance_id, migrate_out_request.request_id))
                status = await self.migration_coordinator.migrate_out_multistage(migrate_in_ray_actor, migrate_out_request)
                if status == MigrationStatus.FINISHED_DONE:
                    await migrate_in_ray_actor.execute_engine_method.remote("commit_dst_request", migrate_out_request)
                    self.backend_engine.free_src_request(migrate_out_request)
                    migrated_request_list.append(migrate_out_request.request_id)
                    migrate_out_request.stage_timestamps.append(time.time())
                    self.backend_engine.remove_migrating_out_request_last_stage(migrate_out_request)
                else:
                    migrate_out_request.reset_migration_args()
                    await migrate_in_ray_actor.execute_migration_method.remote("free_dst_pre_alloc_cache", migrate_out_request.request_id)
                    continue_migrate = False
                t1 = time.time()
                logger.info("{}->{} migrate done, migrate request {}, status:{}, len:{} blocks, cost:{} ms" \
                    .format(self.instance_id, dst_instance_id, migrated_request_list, status, \
                    sum(migrate_out_request.stage_num_blocks_list), (t1 - t0)*1000))
        except ray.exceptions.RayActorError:
            logger.info("[migrate_out] instance {} is dead".format(dst_instance_name[len("instance_"):]))
            raise
        return migrated_request_list

    def get_instance_info(self) -> InstanceInfo:
        return self.backend_engine.engine.instance_info

    def is_ready(self) -> bool:
        return True

    def get_all_request_ids(self) -> List[str]:
        return self.backend_engine.get_all_request_ids()

    def generate(
        self,
        request_id: str,
        server_info: ServerInfo,
        expected_steps: int,
        *args,
        **kwargs,
    ) -> None:
        # This should not be used for logging, as it is monotonic time.
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
                self.backend_engine.add_running_request(backend_request)

    def execute_migration_method(self, method, *args, **kwargs):
        executor = getattr(self.migration_coordinator, method)
        return executor(*args, **kwargs)

    def execute_engine_method(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return executor(*args, **kwargs)
    
