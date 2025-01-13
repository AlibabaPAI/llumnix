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
import random
import time
import csv
import os
from typing import Dict, List, Tuple, Union, Iterable
from collections import defaultdict
import traceback
from functools import partial

import ray
import ray.actor
from ray.util.state import list_placement_groups, list_actors

from llumnix.llumlet.llumlet import Llumlet
from llumnix.logging.logger import init_logger
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.global_scheduler.migration_scheduler import PairMigrationConstraints
from llumnix.global_scheduler.migration_filter import CustomFilter
from llumnix.instance_info import InstanceInfo
from llumnix.arg_utils import ManagerArgs, EntrypointsArgs, InstanceArgs, LaunchArgs
from llumnix.server_info import ServerInfo
from llumnix.backends.backend_interface import BackendType
from llumnix.utils import (random_uuid, clear_gloo_backend_state, get_instance_name,
                           get_manager_name, INSTANCE_NAME_PREFIX, get_placement_group_name,
                           run_async_func_sync,)
from llumnix.entrypoints.utils import LaunchMode
from llumnix.queue.queue_type import QueueType
from llumnix.constants import (CLEAR_REQUEST_INSTANCE_INTERVAL, NO_INSTANCE_RETRY_INTERVAL,
                               WAIT_ALL_MIGRATIONS_DONE_INTERVAL, AUTO_SCALE_UP_INTERVAL,
                               WAIT_PLACEMENT_GROUP_TIMEOUT, CHECK_DEPLOYMENT_STATES_INTERVAL,
                               WATCH_DEPLOYMENT_INTERVAL, WATCH_DEPLOYMENT_INTERVAL_PENDING_INSTANCE)
from llumnix.launcher import Launcher


logger = init_logger(__name__)

# TODO(s5u13b): Handle exception of ray operations.


class Manager:
    def __init__(self,
                 entrypoints_args: EntrypointsArgs,
                 manager_args: ManagerArgs,
                 instance_args: InstanceArgs,
                 engine_args,
                 launch_args: LaunchArgs,
                 work_dir: str,
                 ) -> None:
        os.chdir(work_dir)
        self.job_id = ray.get_runtime_context().get_job_id()
        self.worker_id = ray.get_runtime_context().get_worker_id()
        self.actor_id = ray.get_runtime_context().get_actor_id()
        self.node_id = ray.get_runtime_context().get_node_id()
        logger.info("Manager(job_id={}, worker_id={}, actor_id={}, node_id={})".format(
                        self.job_id, self.worker_id, self.actor_id, self.node_id))
        self.actor_name = get_manager_name()
        self.manager_args = manager_args

        # used in global deployment.
        self.entrypoints_args = entrypoints_args
        self.instance_args = instance_args
        self.engine_args = engine_args
        self.launch_args = launch_args

        # launch args
        if launch_args is not None:
            self.launch_mode: LaunchMode = launch_args.launch_mode
            self.backend_type: BackendType = launch_args.backend_type

        # migration args
        self.enable_migration = manager_args.enable_migration
        self.pair_migration_frequency = manager_args.pair_migration_frequency
        self.enable_pd_disagg = manager_args.enable_pd_disagg

        # scaling args
        self.enable_scaling = manager_args.enable_scaling
        self.max_instances = manager_args.max_instances
        self.min_instances = manager_args.min_instances
        self.scaling_interval = manager_args.scaling_interval
        self.scaling_policy = manager_args.scaling_policy
        self.scale_up_threshold = manager_args.scale_up_threshold
        self.scale_down_threshold = manager_args.scale_down_threshold

        self.polling_interval = manager_args.polling_interval

        self.is_group_kind_migration_backend = manager_args.is_group_kind_migration_backend
        global_scheduler_config = manager_args.create_global_scheduler_config(self.is_group_kind_migration_backend)
        self.global_scheduler = GlobalScheduler(global_scheduler_config)

        self.launcher: Launcher = Launcher(self.global_scheduler, manager_args.enable_port_increment,
                                           manager_args.enable_port_offset_store, manager_args.enable_pd_disagg,
                                           manager_args.enable_engine_pd_disagg, manager_args.pd_ratio)

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

        if hasattr(self, "launch_mode") and self.launch_mode == LaunchMode.GLOBAL:
            assert self.entrypoints_args is not None and self.engine_args is not None
            self.last_timeout_instance_id = None
            asyncio.create_task(self._auto_scale_up_loop(AUTO_SCALE_UP_INTERVAL))
            asyncio.create_task(self._check_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))
            if self.manager_args.enable_pd_disagg:
                asyncio.create_task(self._check_pd_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL))

    def __repr__(self):
        # Customizing prefixes for Actor logs.
        return f"{self.__class__.__name__}(node_id={self.node_id[:5]})"

    async def generate(self, request_id: str, server_info: ServerInfo, *args, **kwargs,) -> None:
        while self.num_instances == 0:
            logger.warning("No instance available now, sleep {}s, "
                           "and regenerate request {}.".format(NO_INSTANCE_RETRY_INTERVAL, request_id))
            await asyncio.sleep(NO_INSTANCE_RETRY_INTERVAL)

        instance_id, request_expected_steps = self.global_scheduler.dispatch()
        try:
            if hasattr(server_info, 'request_timestamps'):
                server_info.request_timestamps.manager_generate_timestamp = time.time()
            await self.instances[instance_id].generate.remote(request_id, server_info, request_expected_steps, *args, **kwargs)
            if self.log_requests:
                logger.info("manager receive request {}".format(request_id))
                logger.info("dispath request {} to instance {}".format(request_id, instance_id))
                self.request_instance[request_id] = instance_id
        except (ray.exceptions.RayActorError, KeyError):
            logger.info("Instance {} is dead, regenerate request {}.".format(instance_id, request_id))
            self.scale_down(instance_id)

    async def abort(self, request_id: Union[str, Iterable[str]]) -> None:
        def abort_done_callback(instance_id: str, request_ids: List[str], fut):
            ret = fut.result()[0]
            if not isinstance(ret, (ray.exceptions.RayActorError, KeyError)):
                if self.log_requests:
                    logger.info("Abort requests: {}.".format(request_ids))
                for req_id in request_ids:
                    if req_id in self.request_instance:
                        del self.request_instance[req_id]
                    else:
                        logger.warning("request {} is not in request_instance".format(req_id))
            else:
                logger.info("Instance {} is dead.".format(instance_id))
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
                logger.info("Instance {} is dead.".format(instance_id))
                self.scale_down(instance_id)

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
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

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
                        logger.info("Instance {} is dead.".format(instance_id))
                        self.scale_down(instance_id)
            else:
                migrate_out_request_ids = ret[0]
                if migrate_out_request_ids:
                    migrate_out_request_id = migrate_out_request_ids[0]
                    self.request_instance[migrate_out_request_id] = migrate_instance_pair[1]
                logger.info("instance {}->{} migrate done, migrate request {}".format(
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
            logger.error("Unexpected exception: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))

    async def _auto_scale_up_loop(self, interval: float) -> None:
        while True:
            try:
                new_pg = None
                if self.last_timeout_instance_id is not None:
                    last_timeout_pg_name = get_placement_group_name(self.last_timeout_instance_id)
                    last_timeout_pg_states = list_placement_groups(filters=[("name", "=", last_timeout_pg_name)])
                    if len(last_timeout_pg_states) > 0:
                        new_instance_id = self.last_timeout_instance_id
                        # pending, created(without server and instance) or rescheduling
                        new_pg = ray.util.get_placement_group(last_timeout_pg_name)
                    # reset
                    self.last_timeout_instance_id = None
                pending_pg_states = list_placement_groups(filters=[("state", "=", "PENDING")])
                pending_pg_states.extend(list_placement_groups(filters=[("state", "=", "RESCHEDULING")]))
                for pending_pg_state in pending_pg_states:
                    instance_id = pending_pg_state["name"].split("_")[-1]
                    if new_pg is not None and instance_id == new_instance_id:
                        continue
                    self.scale_down(instance_id)
                if new_pg is None:
                    new_instance_id = random_uuid()
                    new_pg = self.launcher.init_placement_group(get_placement_group_name(new_instance_id), self.engine_args, self.backend_type,
                                                        init_server=True, block=False)
                try:
                    await asyncio.wait_for(new_pg.ready(), WAIT_PLACEMENT_GROUP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug("Waiting for new placement group {} ready timeout.".format(new_instance_id))
                    # After timeout, the new placement group might be pending,
                    # created(without server and instance), rescheduling.
                    self.last_timeout_instance_id = new_instance_id
                    await asyncio.sleep(interval)
                    continue
                self.launcher.init_server_and_instance(new_instance_id, self.entrypoints_args, self.instance_args,
                                                       self.engine_args, self.backend_type, new_pg,
                                                       instance_finish_cb=self.scale_up)
                logger.info("Deploy server and instance to new placement group done, instance_id: {}.".format(new_instance_id))
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    # TODO(KuilongCui): Add comments for this function.
    async def _rebuild_migration_backend(self) -> None:
        # Wait for all instances to finish migration
        while any(self.instance_migrating.values()):
            await asyncio.sleep(WAIT_ALL_MIGRATIONS_DONE_INTERVAL)

        # During rebuilding migration backend, disable migration.
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
                self.scale_down(dead_instances, rebuild_migration_backend=False)
                clear_gloo_backend_state()
            return dead_instances

        alive_instances = sorted(self.instances.keys())
        pending_task = self.pending_rebuild_migration_instances
        group_name = None
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

        logger.info("Rebuild migration backend done, group_name: {}, alive instance ({}): {}."
            .format(group_name, len(alive_instances), alive_instances))

        # Restore migrate config
        self.enable_migration = origin_config

    def scale_up(self, instance_id: Union[str, Iterable[str]],
                 instance_actor_handle: Union[ray.actor.ActorHandle, List[ray.actor.ActorHandle]],
                 instance_arg: Union[InstanceArgs, Iterable[InstanceArgs]]) -> None:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
            instance_actor_handle = [instance_actor_handle,]
            instance_arg = [instance_arg,]
        instance_ids = list(instance_id)
        instance_actor_handles = list(instance_actor_handle)
        instance_args = list(instance_arg)

        indeed_update = False
        no_pending_instance = (self.pending_rebuild_migration_instances == 0)

        for idx, ins_id in enumerate(instance_ids):
            if ins_id not in self.instances:
                indeed_update = True
                self.instances[ins_id] = instance_actor_handles[idx]
                self.instance_migrating[ins_id] = False
                if self.log_instance_info:
                    self.instance_last_logged_empty[ins_id] = False
                self.pending_rebuild_migration_instances += 1
        self.global_scheduler.scale_up(instance_ids, instance_args)
        self.num_instances = len(self.instances)

        # When scaling up, we need to rebuild the migration backend. But if initially self.pending_rebuild_migration_instances != 0,
        # a coroutine is already handling the changes in the number of instances in the cluster and it will account for the changes
        # caused by this scale-up (see rebuild_migration_backend for details). Therefore, we simply return in this case.
        # Specifically, for not group kind migration backend, there is no need to rebuild the group.
        if self.enable_migration and self.is_group_kind_migration_backend \
            and indeed_update and no_pending_instance:
            asyncio.create_task(self._rebuild_migration_backend())

        return self.num_instances

    def scale_down(self, instance_id: Union[str, Iterable[str]], rebuild_migration_backend: bool = True) -> None:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)

        indeed_update = False
        no_pending_instance = self.pending_rebuild_migration_instances == 0

        for ins_id in instance_ids:
            self.launcher.clear_instance_ray_resources(ins_id)
            if ins_id in self.instances:
                indeed_update = True
                if ins_id in self.instances:
                    del self.instances[ins_id]
                else:
                    logger.debug("instance {} is not in instances".format(ins_id))
                if ins_id in self.instance_migrating:
                    del self.instance_migrating[ins_id]
                else:
                    logger.debug("instance {} is not in instance_migrating".format(ins_id))
                if self.log_instance_info:
                    if ins_id in self.instance_last_logged_empty:
                        del self.instance_last_logged_empty[ins_id]
                    else:
                        logger.debug("instance {} is not in instance_last_logged_empty".format(ins_id))
                self.pending_rebuild_migration_instances += 1
        self.global_scheduler.scale_down(instance_ids)
        self.num_instances = len(self.instances)

        if self.enable_migration and self.is_group_kind_migration_backend:
            if len(self.instances) == 0:
                self.pending_rebuild_migration_instances = 0
                clear_gloo_backend_state()
            elif indeed_update and no_pending_instance and rebuild_migration_backend:
                asyncio.create_task(self._rebuild_migration_backend())

        return self.num_instances

    async def _connect_to_instances(self):
        def connect_to_instances_done_callback(instance_id: str, instance_actor_handle: "ray.actor.ActorHandle", fut):
            ret = fut.result()[0]
            if not isinstance(ret, Exception):
                scale_up_instance_ids.append(instance_id)
                scale_up_instance_actor_handles.append(instance_actor_handle)
                scale_up_instance_args.append(ret)
                logger.info("Connect to instance {}".format(instance_id))
            else:
                logger.warning("Connect to instance {} failed, exception: {}".format(instance_id, ret))

        # Must set True despite set namespance to llumnix.
        actor_names_dict = ray.util.list_named_actors(all_namespaces=True)
        instance_actor_names = [actor_name_dict['name'] for actor_name_dict in actor_names_dict
                                if actor_name_dict['name'].startswith(INSTANCE_NAME_PREFIX)]
        instance_actor_handles = [ray.get_actor(actor_name, namespace='llumnix') for actor_name in instance_actor_names]
        scale_up_instance_ids = []
        scale_up_instance_actor_handles = []
        scale_up_instance_args = []
        tasks = []
        for instance_actor_name, instance_actor_handle in zip(instance_actor_names, instance_actor_handles):
            instance_id = instance_actor_name[len('instance_'):]
            if instance_id not in self.instances:
                task = asyncio.gather(instance_actor_handle.get_instance_args.remote(), return_exceptions=True)
                task.add_done_callback(partial(connect_to_instances_done_callback, instance_id, instance_actor_handle))
                tasks.append(task)
        await asyncio.gather(*tasks)
        # The only function that can add instance actor handles to manager.
        self.scale_up(scale_up_instance_ids, scale_up_instance_actor_handles, scale_up_instance_args)

    @classmethod
    def from_args(cls,
                  entrypoints_args: EntrypointsArgs,
                  manager_args: ManagerArgs,
                  instance_args: InstanceArgs,
                  engine_args,
                  launch_args: LaunchArgs,
                  ) -> "Manager":
        manager_class = ray.remote(num_cpus=1,
                                   max_restarts=-1,
                                   name=get_manager_name(),
                                   namespace="llumnix",
                                   lifetime="detached")(cls)
        manager = manager_class.remote(
            entrypoints_args,
            manager_args,
            instance_args,
            engine_args,
            launch_args,
            os.getcwd())
        return manager

    def init_instances(self,
                       request_output_queue_type: QueueType,
                       backend_type: BackendType,
                       instance_args: InstanceArgs,
                       engine_args
                      ) -> Tuple[List[str], List[Llumlet]]:
        instance_ids: List[str] = []
        instances: List[Llumlet] = []
        for _ in range(self.manager_args.initial_instances):
            instance_id = random_uuid()
            placement_group = self.launcher.init_placement_group(get_placement_group_name(instance_id), engine_args, backend_type)
            instance = self.launcher.init_instance(instance_id, instance_args, placement_group, request_output_queue_type,
                                           backend_type, engine_args)
            instance_ids.append(instance_id)
            instances.append(instance)

        self.scale_up(instance_ids, instances, [instance_args]*len(instance_ids))

        return instance_ids, instances

    def _inner_check_pd_deployment(self) -> str:
        prefill_instance_ids = self.global_scheduler.dispatch_scheduler.available_dispatch_instance_set
        cur_num_prefill = len(prefill_instance_ids)
        decode_instance_ids = self.global_scheduler.instance_id_set - prefill_instance_ids
        cur_num_decode = len(decode_instance_ids)

        scale_down_instance_id = ""
        if cur_num_prefill == 0 and cur_num_decode > 0:
            scale_down_instance_id = random.choice(list(decode_instance_ids))
            logger.info("[_inner_check_pd_deployment] pd_ratio: {}, cur_num_prefill: {}, cur_num_decode: {}, "
                        "all decode, scale down decode instance {}".format(self.manager_args.pd_ratio,
                        cur_num_prefill, cur_num_decode, scale_down_instance_id))

        if cur_num_decode == 0 and cur_num_prefill > 0:
            scale_down_instance_id = random.choice(list(prefill_instance_ids))
            logger.info("[_inner_check_pd_deployment] pd_ratio: {}, cur_num_prefill: {}, cur_num_decode: {}, "
                        "all prefill, scale down prefill instance {}".format(self.manager_args.pd_ratio,
                        cur_num_prefill, cur_num_decode, scale_down_instance_id))

        if scale_down_instance_id:
            self.scale_down(scale_down_instance_id)

        return scale_down_instance_id

    # TODO(KuilongCui): currently, only one naive state check policy is implemented, which prevents the
    # cluster from consisting entirely of prefill or decode instances.
    async def _check_pd_deployment_states_loop(self, interval: float) -> None:
        previous_penging_pg_names = None

        while True:
            try:
                pending_pg_states = list_placement_groups(filters=[("state", "=", "PENDING")])
                rescheduling_pg_states = list_placement_groups(filters=[("state", "=", "RESCHEDULING")])
                all_penging_pg_names = [pg.name for pg in pending_pg_states]

                if previous_penging_pg_names and len(rescheduling_pg_states) == 0 :
                    new_pending_pg_states = list_placement_groups(filters=[("state", "=", "PENDING")])
                    all_new_penging_pg_names = [pg.name for pg in new_pending_pg_states]
                    if len(set(previous_penging_pg_names).difference(set(all_new_penging_pg_names))) == 0:
                        self._inner_check_pd_deployment()
                    previous_penging_pg_names = all_new_penging_pg_names
                else:
                    previous_penging_pg_names = all_penging_pg_names

                await asyncio.sleep(interval)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    async def _check_deployment_states_loop(self, interval: float) -> None:
        async def watch_instance_deployment_states(instance_id: str):
            # There might be some delays of calling _init_server_and_instance, so sleep first.
            await asyncio.sleep(WATCH_DEPLOYMENT_INTERVAL)
            wait_pending_instance_time = 0.0
            while True:
                instance_state = list_actors(filters=[("name", "=", get_instance_name(instance_id))])
                instance_pending_creation = len(instance_state) == 1 and instance_state[0]["state"] == "PENDING_CREATION"
                if not instance_pending_creation:
                    break
                await asyncio.sleep(WATCH_DEPLOYMENT_INTERVAL)
                wait_pending_instance_time += WATCH_DEPLOYMENT_INTERVAL
                if wait_pending_instance_time >= WATCH_DEPLOYMENT_INTERVAL_PENDING_INSTANCE:
                    break
            pg_created, server_alive, instance_alive = self.launcher.get_instance_deployment_states(instance_id)
            if pg_created and (not server_alive or not instance_alive):
                logger.warning("Instance {} deployment states incorrect, states: (pg {}, server {}, instance {})"
                               .format(instance_id, pg_created, server_alive, instance_alive))
                self.scale_down(instance_id)

        while True:
            try:
                curr_pgs, curr_servers, curr_instances = self.launcher.get_cluster_deployment()
                assert len(curr_pgs) >= max(len(curr_servers), len(curr_instances))
                tasks = []
                for instance_id in curr_pgs:
                    if instance_id not in curr_servers or instance_id not in curr_instances:
                        tasks.append(asyncio.create_task(watch_instance_deployment_states(instance_id)))
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(interval)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    async def is_ready(self) -> bool:
        """Called by api server, return true when all the instances have been successfully created."""
        tasks = [instance.is_ready.remote() for instance in self.instances.values()]
        is_ready_list = await asyncio.gather(*tasks)
        return all(is_ready_list)

    async def _check_instance_error(self, migrate_instance_pairs: Tuple[str, str]) -> List[bool]:
        def check_instance_error_done_callback(idx: int, instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, (ray.exceptions.RayActorError, KeyError)):
                logger.info("Instance {} is alive.".format(instance_id))
                results[idx] = False
            else:
                logger.info("Instance {} is dead.".format(instance_id))
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
                logger.info("Instance {} is dead.".format(instance_id))
                self.scale_down(instance_id)

        instance_requests = []
        instance_ids = []
        tasks = []
        for instance_id, instance_actor_handle in self.instances.items():
            task = asyncio.gather(instance_actor_handle.get_all_request_ids.remote(), return_exceptions=True)
            task.add_done_callback(partial(get_request_instance_done_callback, instance_id))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("instance_ids: {}".format(instance_ids))
        logger.debug("instance_requests: {}".format(instance_requests))
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
            'dispatch_load_metric',
            'migration_load_metric',
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
                    instance_info.dispatch_load_metric,
                    instance_info.migration_load_metric,
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
