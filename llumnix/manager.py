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
import time
import csv
import os
from typing import Dict, List, Tuple, Union, Iterable
from collections import defaultdict
import traceback
from functools import partial
import ray

from llumnix.llumlet.llumlet import Llumlet
from llumnix.logger import init_logger
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.global_scheduler.migration_scheduler import PairMigrationConstraints
from llumnix.global_scheduler.migration_filter import CustomFilter
from llumnix.instance_info import InstanceInfo
from llumnix.arg_utils import ManagerArgs, EntrypointsArgs, DeploymentArgs
from llumnix.server_info import ServerInfo
from llumnix.backends.backend_interface import BackendType
from llumnix.utils import (random_uuid, clear_gloo_backend_state, remove_placement_group,
                           get_instance_name, INSTANCE_NAME_PREFIX, MANAGER_NAME,
                           run_async_func_sync)
from llumnix.entrypoints.utils import DeploymentMode
from llumnix.utils import initialize_placement_group
from llumnix.backends.utils import get_engine_world_size
from llumnix.queue.queue_type import QueueType

logger = init_logger(__name__)

CLEAR_REQUEST_INSTANCE_INTERVAL = 3600
NO_INSTANCE_RETRY_INTERVAL = 1.0
WAIT_ALL_MIGRATIONS_DONE_INTERVAL = 1.0
WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS = 1.0
AUTO_DEPLOYMENT_INTERVAL = 1.0

# TODO(s5u13b): Fix the logger when manager failover.
# TODO(s5u13b): Handle exception of ray operations.
# TODO(s5u13b): Update the documents of global deployment.


class Manager:
    def __init__(self,
                 manager_args: ManagerArgs,
                 work_dir: str,
                 entrypoints_args: EntrypointsArgs = None,
                 engine_args = None,
                 deployment_args: DeploymentArgs = None
                 ) -> None:
        os.chdir(work_dir)
        self.actor_name = MANAGER_NAME
        self.manager_args = manager_args
        # engine_args and entrypoints_args are used in global deployment.
        self.entrypoints_args = entrypoints_args
        self.engine_args = engine_args
        self.deployment_args = deployment_args

        assert deployment_args is None or \
            (deployment_args is not None and entrypoints_args is not None and engine_args is not None)

        # deployment args
        if deployment_args is not None:
            self.deployment_mode: DeploymentMode = deployment_args.deployment_mode
            self.backend_type: BackendType = deployment_args.backend_type
        self.max_instances = manager_args.max_instances
        self.min_instances = manager_args.min_instances

        # scheduling args
        self.enable_migration = manager_args.enable_migration
        self.enable_scaling = manager_args.enable_scaling
        self.enable_pd_disagg = manager_args.enable_pd_disagg
        self.polling_interval = manager_args.polling_interval
        self.pair_migration_frequency = manager_args.pair_migration_frequency
        self.scaling_interval = manager_args.scaling_interval

        global_scheduler_config = manager_args.create_global_scheduler_config()
        self.global_scheduler = GlobalScheduler(global_scheduler_config)

        # log args
        self.log_requests = not manager_args.disable_log_requests_manager
        self.log_instance_info = manager_args.log_instance_info
        if self.log_instance_info:
            self._init_instance_info_csv(manager_args)
            self.instance_last_logged_empty = {}

        # instance states
        self.num_instances = 0
        self.instances: Dict[str, Llumlet] = {}
        self.instance_migrating: Dict[str, bool] = {}
        self.pending_rebuild_migration_instances = 0

        # request states
        self.request_instance: Dict[str, str] = {}

        # migration states
        self.num_instance_info_updates = 0
        self.migrating = False

        # auto-scaling states
        self.scale_up_time = -1
        self.scale_down_time = -1
        self.scaling_up = False
        self.scaling_down = False
        self.last_check_scale_time = time.time()

        # tasks
        # When manager starts, it automatically connects to all existing instances.
        run_async_func_sync(self._connect_to_instances())
        asyncio.create_task(self._update_instance_info_loop(self.polling_interval))
        asyncio.create_task(self._clear_request_instance_loop(CLEAR_REQUEST_INSTANCE_INTERVAL))

    async def generate(self, request_id: str, server_info: ServerInfo, *args, **kwargs,) -> None:
        while self.num_instances == 0:
            logger.warning("[generate] no instance available temporarily, sleep {}s, "
                           "and regenerate request {}".format(NO_INSTANCE_RETRY_INTERVAL, request_id))
            await asyncio.sleep(NO_INSTANCE_RETRY_INTERVAL)

        instance_id, request_expected_steps = self.global_scheduler.dispatch()
        try:
            if hasattr(server_info, 'request_timestamps'):
                server_info.request_timestamps.manager_generate_timestamp = time.time()
            await self.instances[instance_id].generate.remote(request_id, server_info, request_expected_steps, *args, **kwargs)
            if self.log_requests:
                logger.info("[generate] manager received request {}".format(request_id))
                logger.info("[generate] dispath request {} to instance {}".format(request_id, instance_id))
                self.request_instance[request_id] = instance_id
        except (ray.exceptions.RayActorError, KeyError):
            logger.info("[generate] instance {} is dead, regenerate request {}".format(instance_id, request_id))
            self.scale_down(instance_id)

    async def abort(self, request_id: Union[str, Iterable[str]]) -> None:
        def abort_done_callback(instance_id: str, request_ids: List[str], fut):
            ret = fut.result()[0]
            if not isinstance(ret, (ray.exceptions.RayActorError, KeyError)):
                if self.log_requests:
                    logger.info("[abort] abort requests: {}.".format(request_ids))
                for req_id in request_ids:
                    if req_id in self.request_instance:
                        del self.request_instance[req_id]
                    else:
                        logger.warning("[abort] request {} is not in request_instance".format(req_id))
            else:
                logger.info("[abort] instance {} is dead".format(instance_id))
                self.scale_down(instance_id)

        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        instance_requests = defaultdict(list)
        for req_id in request_ids:
            # Requests will be free by instance when finished, so it is acceptable to miss aborted requests.
            if req_id in self.request_instance:
                instance_id = self.request_instance[req_id]
                instance_requests[instance_id].append(req_id)
        tasks = []
        for instance_id, request_ids in instance_requests.items():
            task = asyncio.gather(self.instances[instance_id].abort.remote(request_ids), return_exceptions=True)
            task.add_done_callback(partial(abort_done_callback, instance_id, request_ids))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _update_instance_info_loop(self, interval: float) -> None:
        def update_instance_info_done_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, ray.exceptions.RayActorError):
                if ret is not None:
                    instance_infos.append(ret)
                    self.global_scheduler.update_instance_infos([ret])
            else:
                logger.info("[_update_instance_info_loop] instance {} is dead".format(instance_id))
                self.scale_down(instance_id)
                logger.info("[_update_instance_info_loop] dead instances: {}.".format(ret))
                logger.info("[_update_instance_info_loop] dead instances: {}.".format(self.instances))

        while True:
            try:
                await asyncio.sleep(interval)
                tasks = []
                instance_infos = []
                for instance_id, instance in self.instances.items():
                    # Use asyncio.gather to wrap ray remote call to add done callback, asyncio.create_task will get error.
                    task = asyncio.gather(instance.get_instance_info.remote(), return_exceptions=True)
                    task.add_done_callback(partial(update_instance_info_done_callback, instance_id))
                    tasks.append(task)
                await asyncio.gather(*tasks, return_exceptions=True)
                self.num_instance_info_updates += 1
                # Push migrate when the instance_info have updated a certain number of times.
                if self.enable_migration and self.num_instance_info_updates != 0 \
                    and self.num_instance_info_updates % self.pair_migration_frequency == 0:
                    asyncio.create_task(self._push_migrations())
                if self.log_instance_info:
                    self._log_instance_infos_to_csv(instance_infos)
            # pylint: disable=W0703
            except Exception as e:
                logger.error("[_update_instance_info_loop] unexpected exception occurs: {}".format(e))
                logger.error("[_update_instance_info_loop] exception traceback: {}".format(traceback.format_exc()))

    async def _push_migrations(self) -> None:
        if self.enable_pd_disagg:
            asyncio.create_task(self._migrate(PairMigrationConstraints.PREFILL_2_DECODING))
            asyncio.create_task(self._migrate(PairMigrationConstraints.DECODING_2_DECODING))
        else:
            asyncio.create_task(self._migrate(PairMigrationConstraints.NO_CONSTRAINTS))

    async def _migrate(self, pair_migration_type: PairMigrationConstraints) -> None:
        # TODO(s5u13b): Remove the migration done callback through decentralized migration refactoring.
        async def migrate_done_callback(ret, migrate_instance_pair: Tuple[str, str]) -> None:
            if migrate_instance_pair[0] in self.instance_migrating:
                self.instance_migrating[migrate_instance_pair[0]] = False
            if migrate_instance_pair[1] in self.instance_migrating:
                self.instance_migrating[migrate_instance_pair[1]] = False
            if isinstance(ret, (ray.exceptions.RayActorError, ray.exceptions.RayTaskError, KeyError)):
                has_error_pair = await self._check_instance_error(migrate_instance_pair)
                for i, has_error in enumerate(has_error_pair):
                    # Instance without error should clear migration states.
                    # TODO(s5u13b): Fix the clear_migration_states to adapt to the many-to-many migration.
                    if not has_error:
                        try:
                            await self.instances[migrate_instance_pair[i]].clear_migration_states.remote(is_migrate_in=bool(i))
                        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError, KeyError):
                            has_error = True
                for i, has_error in enumerate(has_error_pair):
                    if has_error:
                        instance_id = migrate_instance_pair[i]
                        logger.info("[_migrate] instance {} is dead".format(instance_id))
                        self.scale_down(instance_id)
            else:
                migrate_out_request_ids = ret
                if migrate_out_request_ids:
                    migrate_out_request_id = migrate_out_request_ids[0]
                    self.request_instance[migrate_out_request_id] = migrate_instance_pair[1]
                logger.info("[_migrate] {}->{} migrate done, migrate request {}".format(
                    migrate_instance_pair[0], migrate_instance_pair[1], migrate_out_request_ids))
        def migrate_done_callback_wrapper(migrate_instance_pair: Tuple[str, str], fut) -> None:
            ret = fut.result()
            loop = asyncio.get_event_loop()
            loop.create_task(migrate_done_callback(ret, migrate_instance_pair))

        try:
            migrate_instance_pairs = self.global_scheduler.pair_migration(pair_migration_type)
            migration_tasks = []
            for _, migrate_instance_pair in enumerate(migrate_instance_pairs):
                migrate_out_instance_id, migrate_in_instance_id = migrate_instance_pair
                if self.instance_migrating[migrate_out_instance_id] or self.instance_migrating[migrate_in_instance_id]:
                    continue
                self.instance_migrating[migrate_out_instance_id] = True
                self.instance_migrating[migrate_in_instance_id] = True
                migrate_in_instance_name = get_instance_name(migrate_in_instance_id)
                task = asyncio.gather(self.instances[migrate_out_instance_id].migrate_out.remote(migrate_in_instance_name),
                                      return_exceptions=True)
                task.add_done_callback(partial(migrate_done_callback_wrapper, migrate_instance_pair))
                migration_tasks.append(task)
            await asyncio.gather(*migration_tasks, return_exceptions=True)
        # pylint: disable=W0703
        except Exception as e:
            logger.error("[_migrate] unexpected exception occurs: {}".format(e))
            logger.error("[_migrate] exception traceback: {}".format(traceback.format_exc()))

    async def _rebuild_migrate_backend(self) -> None:
        # Wait for all instances to finish migration
        while any(self.instance_migrating.values()):
            await asyncio.sleep(WAIT_ALL_MIGRATIONS_DONE_INTERVAL)

        # During rebuilding migration backend, disable migrate
        origin_config = self.enable_migration
        self.enable_migration = False

        async def run_task(alive_instances: List[str], task_name: str, *args, **kwargs):
            tasks = []
            for instance_name in alive_instances:
                llumlet_handle = self.instances[instance_name]
                tasks.append(llumlet_handle.execute_engine_method.remote("_run_workers", task_name, *args, **kwargs))
            rets = await asyncio.gather(*tasks, return_exceptions=True)
            dead_instances = set()
            for instance_name, ret in zip(alive_instances, rets):
                if isinstance(ret, ray.exceptions.RayActorError):
                    dead_instances.add(instance_name)
            if len(dead_instances) > 0:
                self.scale_down(dead_instances, rebuild_migrate_backend=False)
                if self.manager_args.migration_backend == 'gloo':
                    clear_gloo_backend_state()
            return dead_instances

        alive_instances = sorted(self.instances.keys())
        pending_task = self.pending_rebuild_migration_instances
        group_name = None

        if self.manager_args.migration_backend == 'gloo':
            clear_gloo_backend_state()

        while len(alive_instances) > 0 and self.pending_rebuild_migration_instances > 0:
            dead_instances = set()
            group_name = random_uuid()
            instance_rank = {instance_id: index for index, instance_id in enumerate(alive_instances)}
            dead_instances.update(await run_task(alive_instances, "rebuild_migration_backend",
                                                                  instance_rank, group_name))
            if len(dead_instances) == 0 and self.pending_rebuild_migration_instances == pending_task:
                dead_instances.update(await run_task(alive_instances, "warmup"))
            if len(dead_instances) == 0:
                self.pending_rebuild_migration_instances -= pending_task
            alive_instances = sorted(set(self.instances.keys()) - dead_instances)
            pending_task = self.pending_rebuild_migration_instances

        if len(alive_instances) == 0:
            self.pending_rebuild_migration_instances = 0
            group_name = None

        migration_filter: CustomFilter = self.global_scheduler.migration_scheduler\
            .migration_filter.get_filter("migration_backend_init_filter")
        migration_filter.set_filter_condtition(
            src_filter=lambda instance_info: instance_info.instance_id in alive_instances,
            dst_filter=lambda instance_info: instance_info.instance_id in alive_instances)

        logger.info("[rebuild_migrate_backend] rebuild {} migration backend done, group_name: {}, alive instance ({}): {}"
            .format(self.manager_args.migration_backend, group_name, len(alive_instances), alive_instances))

        # Restore migrate config
        self.enable_migration = origin_config

    def scale_up(self, instance_id: Union[str, Iterable[str]], llumlet_actor_handles: List["ray.actor.ActorHandle"]) -> None:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)

        indeed_update = False
        no_pending_instance = (self.pending_rebuild_migration_instances == 0)

        for idx, ins_id in enumerate(instance_ids):
            if ins_id not in self.instances:
                indeed_update = True
                self.instances[ins_id] = llumlet_actor_handles[idx]
                self.instance_migrating[ins_id] = False
                if self.log_instance_info:
                    self.instance_last_logged_empty[ins_id] = False
                self.pending_rebuild_migration_instances += 1
        self.global_scheduler.scale_up(instance_ids)
        self.num_instances = len(self.instances)

        # When scaling up, we need to rebuild the migration backend. But if initially self.pending_rebuild_migration_instances != 0,
        # a coroutine is already handling the changes in the number of instances in the cluster and it will account for the changes
        # caused by this scale-up (see rebuild_migrate_backend for details). Therefore, we simply return in this case. Specifically,
        # for RPC, the Ray actor handle is used for the migration cache, so there is no need to rebuild the group.
        if self.enable_migration and self.manager_args.migration_backend in ['gloo', 'nccl'] \
            and indeed_update and no_pending_instance:
            asyncio.create_task(self._rebuild_migrate_backend())

        return self.num_instances

    def scale_down(self, instance_id: Union[str, Iterable[str]], rebuild_migrate_backend: bool = True) -> None:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)

        indeed_update = False
        no_pending_instance = self.pending_rebuild_migration_instances == 0

        for ins_id in instance_ids:
            if ins_id in self.instances:
                indeed_update = True
                if ins_id in self.instances:
                    del self.instances[ins_id]
                else:
                    logger.warning("[scale_down] instance {} is not in self.instances".format(ins_id))
                if ins_id in self.instance_migrating:
                    del self.instance_migrating[ins_id]
                else:
                    logger.warning("[scale_down] instance {} is not in self.instance_migrating".format(ins_id))
                if not remove_placement_group(ins_id):
                    logger.warning("[scale_down] failed to remove placement group of instance {}".format(ins_id))
                if self.log_instance_info:
                    if ins_id in self.instance_last_logged_empty:
                        del self.instance_last_logged_empty[ins_id]
                    else:
                        logger.warning("[scale_down] instance {} is not in self.instance_last_logged_empty".format(ins_id))
                self.pending_rebuild_migration_instances += 1
        self.global_scheduler.scale_down(instance_ids)
        self.num_instances = len(self.instances)

        if self.enable_migration and self.manager_args.migration_backend in ['gloo', 'nccl']:
            if len(self.instances) == 0:
                self.pending_rebuild_migration_instances = 0
                if self.manager_args.migration_backend == 'gloo':
                    clear_gloo_backend_state()
            elif indeed_update and no_pending_instance and rebuild_migrate_backend:
                asyncio.create_task(self._rebuild_migrate_backend())

        return self.num_instances

    async def _connect_to_instances(self):
        def connect_to_instances_done_callback(instance_id: str, instance_actor_handle: "ray.actor.ActorHandle", fut):
            ret = fut.result()[0]
            if not isinstance(ret, Exception):
                scale_up_instance_ids.append(instance_id)
                scale_up_instance_actor_handles.append(instance_actor_handle)
                logger.info("[_connect_to_instances] connect to instance {}.".format(instance_id))
            else:
                logger.warning("[_connect_to_instances] connect to instance {} failed, exception: {}".format(instance_id, ret))

        # Must set True despite set namespance to llumnix.
        actor_names_dict = ray.util.list_named_actors(all_namespaces=True)
        instance_actor_names = [actor_name_dict['name'] for actor_name_dict in actor_names_dict
                                if actor_name_dict['name'].startswith(INSTANCE_NAME_PREFIX)]
        instance_actor_handles = [ray.get_actor(actor_name, namespace='llumnix') for actor_name in instance_actor_names]
        scale_up_instance_ids = []
        scale_up_instance_actor_handles = []
        tasks = []
        for instance_actor_name, instance_actor_handle in zip(instance_actor_names, instance_actor_handles):
            instance_id = instance_actor_name[len('instance_'):]
            if instance_id not in self.instances:
                task = asyncio.gather(instance_actor_handle.is_ready.remote(), return_exceptions=True)
                task.add_done_callback(partial(connect_to_instances_done_callback, instance_id, instance_actor_handle))
                tasks.append(task)
        await asyncio.gather(*tasks)
        # The only function that can add instance actor handles to manager.
        self.scale_up(scale_up_instance_ids, scale_up_instance_actor_handles)

    @classmethod
    def from_args(cls,
                  manager_args: ManagerArgs,
                  entrypoints_args: EntrypointsArgs = None,
                  engine_args = None,
                  deployment_args: DeploymentArgs = None,
                  ) -> "Manager":
        manager_class = ray.remote(num_cpus=0,
                                   max_restarts=-1,
                                   name=MANAGER_NAME,
                                   namespace='llumnix',
                                   lifetime="detached")(cls)
        manager = manager_class.remote(manager_args,
                                       os.getcwd(),
                                       entrypoints_args,
                                       engine_args,
                                       deployment_args)

        return manager

    def init_llumlets(self,
                      request_output_queue_type: QueueType,
                      backend_type: BackendType,
                      engine_args,
                      instance_ids: List[str] = None
                      ) -> Tuple[List[str], List[Llumlet]]:
        manager_args = self.manager_args
        world_size = get_engine_world_size(engine_args, backend_type)

        instance_ids: List[str] = []
        llumlets: List[Llumlet] = []
        for _ in range(manager_args.initial_instances):
            instance_id = random_uuid()
            if not manager_args.profiling_result_file_path:
                # num_cpus=3, for Llumlet + AsyncPutQueueActor + ProxyActor, num_gpus=world_size, for Workers
                placement_group = initialize_placement_group(instance_id, num_cpus=3, num_gpus=world_size, detached=True)
                # TODO(s5u13b): Refine the order of arguments.
                llumlet = Llumlet.from_args(
                    request_output_queue_type,
                    instance_id,
                    backend_type,
                    world_size,
                    engine_manager_args.create_migration_config(),
                    placement_group,
                    engine_args,
                    placement_group,
                    engine_args,
                    *args,
                    **kwargs
                )
            else:
                assert backend_type == backend_type.VLLM, 'Only support the simulator backend for vLLM.'
                # num_cpus=1, for Llumlet + AsyncPutQueueActor
                placement_group = initialize_placement_group(instance_id, num_cpus=2, num_gpus=0, detached=True)
            llumlet = Llumlet.from_args(
                            instance_id,
                            placement_group,
                            request_output_queue_type,
                            manager_args.create_migration_config(),
                            backend_type,
                            engine_args,
                            manager_args.profiling_result_file_path)
            instance_ids.append(instance_id)
            llumlets.append(llumlet)

        return instance_ids, llumlets

    async def is_ready(self) -> bool:
        """Called by api server, return true when all the instances have been successfully created."""
        tasks = [llumlet.is_ready.remote() for llumlet in self.instances.values()]
        is_ready_list = await asyncio.gather(*tasks)
        return all(is_ready_list)

    async def _check_instance_error(self, migrate_instance_pairs: Tuple[str, str]) -> List[bool]:
        def check_instance_error_done_callback(idx: int, instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, (ray.exceptions.RayActorError, KeyError)):
                logger.info("[_check_instance_error] instance {} is alive".format(instance_id))
                results[idx] = False
            else:
                logger.info("[_check_instance_error] instance {} is dead".format(instance_id))
                results[idx] = True

        results = [None, None]
        tasks = []
        for idx, instance_id in enumerate(migrate_instance_pairs):
            task = asyncio.gather(self.instances[instance_id].is_ready.remote(), return_exceptions=True)
            task.add_done_callback(partial(check_instance_error_done_callback, idx, instance_id))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def _get_request_instance(self) -> None:
        def get_request_instance_done_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, ray.exceptions.RayActorError):
                instance_requests.append(ret)
                instance_ids.append(instance_id)
            else:
                logger.info("[_get_request_instance] instance {} is dead".format(instance_id))
                self.scale_down(instance_id)

        instance_requests = []
        instance_ids = []
        tasks = []
        for instance_id, instance_actor_handle in self.instances.items():
            task = asyncio.gather(instance_actor_handle.get_instance_info.remote(), return_exceptions=True)
            task.add_done_callback(partial(get_request_instance_done_callback, instance_id))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("[_get_request_instance] instance_ids: {}".format(instance_ids))
        logger.debug("[_get_request_instance] instance_requests: {}".format(instance_requests))
        for (instance_id, requests) in zip(instance_ids, instance_requests):
            for request_id in requests:
                self.request_instance[request_id] = instance_id

    async def _clear_request_instance_loop(self, interval: float):
        await self._get_request_instance()
        # Clear the request_instance at a certain interval to prevent memory leaking.
        while True:
            await asyncio.sleep(interval)
            self.request_instance = {}

    def _init_instance_info_csv(self, manager_args: ManagerArgs) -> None:
        # pylint: disable=consider-using-with
        self.instance_info_file = open(manager_args.log_filename + '_instance.csv', 'w', encoding='utf-8')
        self.instance_info_csv = csv.writer(self.instance_info_file)
        self.instance_info_csv.writerow([
            'timestamp',
            'instance_id',
            'step_id',
            'gpu_cache_usage',
            'num_available_gpu_blocks',
            'instance_load',
            'max_tot_tokens',
            'num_running_requests',
            'num_waiting_requests',
            'num_killed_requests',
            'inference_type',
            'bs',
            'profiling_data',
            'seq_lens',
            'num_instances',
            'num_seqs',
            'num_blocks_first_waiting_request',
            'num_blocks_all_waiting_requests',
            'waiting_time_first_waiting_request'])

    def _log_instance_infos_to_csv(self, instance_infos: List[InstanceInfo]) -> None:
        for instance_info in instance_infos:
            instance_id = instance_info.instance_id
            gpu_cache_usage = instance_info.gpu_cache_usage
            should_log = (gpu_cache_usage > 0) or (gpu_cache_usage == 0 and not self.instance_last_logged_empty[instance_id])
            if should_log:
                self.instance_last_logged_empty[instance_id] = (gpu_cache_usage == 0)
                self.instance_info_csv.writerow([
                    instance_info.timestamp,
                    instance_info.instance_id,
                    instance_info.step_id,
                    instance_info.gpu_cache_usage,
                    instance_info.num_available_gpu_blocks,
                    instance_info.instance_load_migrate,
                    instance_info.max_tot_tokens,
                    instance_info.num_running_requests,
                    instance_info.num_waiting_requests,
                    instance_info.num_killed_requests,
                    instance_info.inference_type,
                    instance_info.num_batched_tokens,
                    instance_info.profiling_data,
                    instance_info.running_seq_lens,
                    self.num_instances,
                    instance_info.num_seqs,
                    instance_info.num_blocks_first_waiting_request,
                    instance_info.num_blocks_all_waiting_requests,
                    instance_info.waiting_time_first_waiting_request])
        self.instance_info_file.flush()
