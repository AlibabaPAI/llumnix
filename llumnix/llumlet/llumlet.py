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

from typing import List, Dict, Union, Iterable
import time
from collections import defaultdict
import ray
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.backends.utils import init_backend_engine, initialize_cluster
from llumnix.llumlet.migration_coordinator import MigrationCoordinator, MigrationStatus
from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler
from llumnix.server_info import ServerInfo
from llumnix.config import MigrationConfig

logger = init_logger(__name__)


class Llumlet:
    def __init__(self,
                 instance_id: str,
                 backend_type: BackendType,
                 migration_config: MigrationConfig,
                 *args,
                 **kwargs) -> None:
        self.instance_id = instance_id
        self.actor_name = f"instance_{instance_id}"
        self.backend_engine: BackendInterface = init_backend_engine(self.instance_id,
                                                                    backend_type,
                                                                    migration_config,
                                                                    *args,
                                                                    **kwargs)
        self.migration_coordinator = MigrationCoordinator(self.backend_engine,
                                                          migration_config.last_stage_max_blocks,
                                                          migration_config.max_stages)
        self.migration_scheduler = LocalMigrationScheduler(migration_config.migrate_policy,
                                                           self.backend_engine)
        self.log_requests = True
        self.instance_info = None

    @classmethod
    def from_args(cls,
                  fixed_node_init: bool,
                  instance_id: str,
                  backend_type: BackendType,
                  world_size: int,
                  migration_config: MigrationConfig,
                  *args,
                  **kwargs):
        llumlet = None
        assert backend_type in [backend_type.VLLM, backend_type.SIM_VLLM], f'unimplemented backend {backend_type}'
        if backend_type == backend_type.VLLM:
            if not fixed_node_init:
                placement_group = initialize_cluster(world_size)
                kwargs["placement_group"] = placement_group
                engine_class = ray.remote(num_cpus=1,
                                          name=f"instance_{instance_id}",
                                          namespace='llumnix',
                                          max_concurrency=4)(cls).options(
                                                scheduling_strategy=PlacementGroupSchedulingStrategy(
                                                    placement_group=placement_group,
                                                    placement_group_bundle_index=0,))
            else:
                placement_group = None
                kwargs["placement_group"] = placement_group
                engine_class = ray.remote(num_cpus=1,
                                          name=f"instance_{instance_id}",
                                          namespace='llumnix',
                                          max_concurrency=4)(cls).options(
                                                scheduling_strategy=NodeAffinitySchedulingStrategy(
                                                    node_id=ray.get_runtime_context().get_node_id(),
                                                    soft=False,))
        else: # backend_type == backend_type.SIM_VLLM:
            engine_class = ray.remote(num_cpus=1,
                                      name=f"instance_{instance_id}",
                                      namespace='llumnix',
                                      max_concurrency=4)(cls).options(
                                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                                            node_id=ray.get_runtime_context().get_node_id(),
                                            soft=False,))
        llumlet = engine_class.remote(instance_id, backend_type, migration_config, *args, **kwargs)
        # circular dependency
        # engine_manager = ray.get_actor(MANAGER_ACTOR_NAME, namespace='llumnix')
        # retry_manager_ray_call_by_ray_get(engine_manager.scale_up.remote, 'scale_up', instance_id, llumlet)
        llumlet.run_engine_loop.remote()
        return llumlet

    def migrate_out(self, dst_instance_name: str) -> List[str]:
        try:
            t0 = time.time()
            migrate_in_ray_actor = ray.get_actor(dst_instance_name, namespace='llumnix')
            dst_instance_id = dst_instance_name[len("instance_"):]
            logger.info("{}->{} begin migrate out".format(self.instance_id, dst_instance_id))
            migrate_out_request = self.migration_scheduler.get_migrate_out_request()
            migrated_request_list = []
            if migrate_out_request is None:
                return migrated_request_list
            status = self.migration_coordinator.migrate_out_multistage(migrate_in_ray_actor, migrate_out_request)
            if status == MigrationStatus.FINISHED_DONE:
                self.backend_engine.free_src_request(migrate_out_request.backend_request)
                migrated_request_list.append(migrate_out_request.request_id)
                migrate_out_request.stage_timestamps.append(time.time())
                self.backend_engine.free_request_states(migrate_out_request.request_id)
                self.backend_engine.remove_migrating_out_request_last_stage(migrate_out_request.backend_request)
            else:
                ray.get(migrate_in_ray_actor.execute_migration_method.remote("free_dst_pre_alloc_cache", migrate_out_request.request_id))
            t1 = time.time()
            logger.info("{}->{} migrate done, migrate request {}, status:{}, len:{} blocks, cost:{} ms" \
                  .format(self.instance_id, dst_instance_id, migrated_request_list, status, \
                   sum(migrate_out_request.stage_num_blocks_list), (t1 - t0)*1000))
        except ray.exceptions.RayActorError:
            logger.info("[migrate_out] instance {} is dead".format(dst_instance_name[len("instance_"):]))
            raise
        return migrated_request_list

    def get_instance_info(self) -> InstanceInfo:
        return self.instance_info

    def get_actor_name(self) -> str:
        return self.actor_name

    def get_instance_id(self) -> str:
        return self.instance_id

    def is_ready(self) -> bool:
        return True

    def get_all_request_ids(self) -> List[str]:
        return self.backend_engine.get_all_request_ids()

    def generate(
        self,
        request_id: str,
        server_info: ServerInfo,
        *args,
        **kwargs,
    ) -> None:
        # This should not be used for logging, as it is monotonic time.
        self.backend_engine.add_request(request_id, server_info, *args, **kwargs)

    def abort(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        return self.backend_engine.abort_request(request_ids)

    def run_engine_loop(self) -> None:
        while True:
            request_outputs, instance_info, server_infos = self.backend_engine.step()
            self.instance_info = instance_info
            if len(request_outputs) == 0:
                time.sleep(0.01)
            else:
                self._put_request_output_to_server(request_outputs, server_infos)

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

    def _put_request_output_to_server(self, request_outputs, server_infos: List[ServerInfo]) -> None:
        server_request_outputs = defaultdict(list)
        server_queue: Dict[str, RayQueue] = {}
        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output, server_info in zip(request_outputs, server_infos):
            server_id = server_info.server_id
            request_output_queue = server_info.request_output_queue
            server_request_outputs[server_id].append(request_output)
            if server_id not in server_queue:
                server_queue[server_id] = request_output_queue
        for server_id, req_outputs in server_request_outputs.items():
            try:
                server_queue[server_id].actor.put_nowait_batch.remote(req_outputs)
            except ray.exceptions.RayActorError:
                logger.info("Server {} is dead".format(server_id))
                request_ids = [req_output.request_id for req_output in req_outputs]
                self.abort(request_ids)

    def execute_migration_method(self, method, *args, **kwargs):
        executor = getattr(self.migration_coordinator, method)
        return executor(*args, **kwargs)

    def execute_engine_method(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return executor(*args, **kwargs)
