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
from typing import Dict, List, Tuple, Union, Iterable, Optional
from functools import partial

import ray
import ray.actor
from ray.util.placement_group import PlacementGroup
import ray.exceptions
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from llumnix.llumlet.llumlet import Llumlet
from llumnix.logging.logger import init_logger
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.global_scheduler.migration_scheduler import PairMigrationConstraints
from llumnix.global_scheduler.migration_filter import CustomFilter
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.arg_utils import (
    ManagerArgs,
    EntrypointsArgs,
    InstanceArgs,
    LaunchArgs,
    LlumnixEngineArgs,
)
from llumnix.server_info import ServerInfo
from llumnix.backends.backend_interface import BackendType
from llumnix.utils import (
    random_uuid,
    run_coroutine_in_new_thread,
    async_wrapper,
    ray_get_with_timeout,
    asyncio_wait_for_with_timeout,
    RequestIDType,
)
from llumnix.ray_utils import (
    get_manager_name,
    INSTANCE_NAME_PREFIX,
    get_placement_group_name,
    log_actor_ray_info,
    execute_actor_method_async_with_retries,
    get_scaler_name,
)
from llumnix.entrypoints.utils import LaunchMode
from llumnix.constants import NO_INSTANCE_RETRY_GENERATE_INTERVAL, WAIT_ALL_MIGRATIONS_DONE_INTERVAL
from llumnix.metrics.timestamps import set_timestamp
from llumnix.entrypoints.api_server_actor import APIServerActor

logger = init_logger(__name__)

# TODO(s5u13b): Handle exception of ray operations.
# TODO(s5u13b): Refactor manager to divide functions into different classes.


class Manager:
    def __init__(self,
                 entrypoints_args: EntrypointsArgs,
                 manager_args: ManagerArgs,
                 instance_args: InstanceArgs,
                 engine_args: LlumnixEngineArgs,
                 launch_args: LaunchArgs,
                 work_dir: str,
                 ) -> None:
        os.chdir(work_dir)
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.manager_args = manager_args

        # used in global launch
        self.entrypoints_args = entrypoints_args # not used
        self.instance_args = instance_args
        self.engine_args = engine_args # not used
        self.launch_args = launch_args # not used

        # scaling args
        # avoid circular import
        # pylint: disable=import-outside-toplevel
        from llumnix.scaler import Scaler
        self.scaler: Scaler = ray.get_actor(get_scaler_name(), namespace="llumnix")

        # launch args
        if launch_args is not None:
            self.launch_mode: LaunchMode = launch_args.launch_mode
            self.backend_type: BackendType = launch_args.backend_type

        # migration args
        self.enable_migration = manager_args.enable_migration
        self.pair_migration_frequency = manager_args.pair_migration_frequency
        self.is_group_kind_migration_backend = manager_args.is_group_kind_migration_backend

        # prefill-decode disaggregation args
        self.enable_pd_disagg = manager_args.enable_pd_disagg

        # scheduling states
        self.polling_interval = manager_args.polling_interval
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
        self.instance_id_2_engine_disagg_inst_id: Dict[str, str] = {}
        self.pgs: Dict[str, PlacementGroup] = {}
        self.servers: Dict[str, APIServerActor] = None
        if hasattr(self, "launch_mode") and self.launch_mode == LaunchMode.GLOBAL:
            self.servers = {}
        self.all_instances_not_migrating = True
        self.pending_rebuild_migration_instances = 0

        # migration states
        self.num_instance_info_updates = 0
        self.migrating = False

        # When manager starts, it automatically connects to all existing instances.
        run_coroutine_in_new_thread(self._connect_to_instances(), blocking=True)
        asyncio.create_task(self._poll_instance_info_loop(self.polling_interval))

    async def generate(self, request_id: RequestIDType, server_info: ServerInfo, *args, **kwargs) -> None:
        while self.num_instances == 0:
            logger.warning("No instance available now, sleep {}s, "
                           "and regenerate request {}.".format(NO_INSTANCE_RETRY_GENERATE_INTERVAL, request_id))
            await asyncio.sleep(NO_INSTANCE_RETRY_GENERATE_INTERVAL)
        prefill_instance_id, request_expected_steps = self.global_scheduler.dispatch(InstanceType.PREFILL)
        if self.manager_args.enable_engine_pd_disagg:
            # Only used in bladellm now
            decode_instance_id, _ = self.global_scheduler.dispatch(InstanceType.DECODE)
            kwargs["decode_instance_id"] = self.instance_id_2_engine_disagg_inst_id.get(
                decode_instance_id, None
            )
        set_timestamp(server_info, 'manager_generate_timestamp', time.time())
        try:
            asyncio.create_task(
                asyncio_wait_for_with_timeout(
                    async_wrapper(
                        self.instances[prefill_instance_id].generate.remote,
                        request_id, server_info, request_expected_steps, *args, **kwargs
                    )
                )
            )
        # pylint: disable=broad-except
        except Exception as e:
            logger.exception("Failed to generate request {} by instance {}, unexcepted exception: {}".format(
                request_id, prefill_instance_id, e))
            self.scale_down(prefill_instance_id)
            await asyncio.create_task(self.generate(request_id, server_info, *args, **kwargs))
        if self.log_requests:
            logger.info("manager receive request {}".format(request_id))
            logger.info("dispath request {} to instance {}".format(request_id, prefill_instance_id))
            if self.manager_args.enable_engine_pd_disagg:
                logger.info("dispatch request {} to decode instance {}".format(request_id, decode_instance_id))

    @classmethod
    def from_args(cls,
                  entrypoints_args: EntrypointsArgs,
                  manager_args: ManagerArgs,
                  instance_args: InstanceArgs,
                  engine_args: LlumnixEngineArgs,
                  launch_args: LaunchArgs,
                  ) -> "Manager":
        manager_class = ray.remote(
            num_cpus=1,
            max_restarts=-1,
            name=get_manager_name(),
            namespace="llumnix",
            lifetime="detached"
        )(cls).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )
        )
        manager = manager_class.remote(
            entrypoints_args,
            manager_args,
            instance_args,
            engine_args,
            launch_args,
            os.getcwd()
        )
        return manager

    async def is_ready(self) -> bool:
        """Called by api server, return true when all the instances have been successfully created."""
        tasks = [
            asyncio_wait_for_with_timeout(instance.is_ready.remote())
            for instance in self.instances.values()
        ]
        # Note that llumnix run server and scale up instance in manager after instance is ready,
        # so the waiting time here will not include the initialization time of instance.
        is_ready_list = await asyncio.gather(*tasks, return_exceptions=True)
        return all(is_ready_list)

    async def _poll_instance_info_loop(self, interval: float) -> None:
        async def get_instance_info_done_callback(ret, instance_id: str):
            if not isinstance(ret, Exception):
                if ret is not None:
                    instance_infos.append(ret)
                    self.global_scheduler.update_instance_infos([ret])
            else:
                if isinstance(ret, ray.exceptions.RayActorError):
                    logger.info("Instance {} is dead.".format(instance_id))
                elif isinstance(ret, asyncio.TimeoutError):
                    logger.error("Instance {} is hang, please check the cause.".format(instance_id))
                else:
                    logger.exception("Failed to poll instance info of instance {}, "
                                     "unexpected exception: {}".format(instance_id, ret))
                await self.scale_down(instance_id)

        def get_instance_info_done_callback_wrapper(instance_id: str, fut) -> None:
            ret = fut.result()[0]
            loop = asyncio.get_event_loop()
            loop.create_task(get_instance_info_done_callback(ret, instance_id))

        while True:
            await asyncio.sleep(interval)
            tasks = []
            instance_infos = []
            for instance_id, instance in self.instances.items():
                # Use asyncio.gather to wrap ray remote call to add done callback, asyncio.create_task will get error.
                task = asyncio.gather(
                    asyncio_wait_for_with_timeout(instance.get_instance_info.remote()),
                    return_exceptions=True
                )
                task.add_done_callback(partial(get_instance_info_done_callback_wrapper, instance_id))
                tasks.append(task)
            if self.num_instance_info_updates % 1000 == 0:
                logger.debug("Polling instance infos of {} instances starts.".format(self.num_instances))
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.num_instance_info_updates % 1000 == 0:
                logger.debug("Polling instance infos of {} instances ends.".format(self.num_instances))
            self.num_instance_info_updates += 1
            self._update_all_instances_not_migrating(instance_infos)
            # Push migrate when the instance_info have updated a certain number of times.
            if self.enable_migration and self.num_instance_info_updates != 0 \
                and self.num_instance_info_updates % self.pair_migration_frequency == 0:
                asyncio.create_task(self._push_migrations())
            if self.log_instance_info:
                self._log_instance_infos_to_csv(instance_infos)

    async def _push_migrations(self) -> None:
        if self.enable_pd_disagg:
            asyncio.create_task(self._migrate(PairMigrationConstraints.PREFILL_2_DECODE))
            asyncio.create_task(self._migrate(PairMigrationConstraints.DECODE_2_DECODE))
        else:
            asyncio.create_task(self._migrate(PairMigrationConstraints.NO_CONSTRAINTS))

    async def _migrate(self, pair_migration_type: PairMigrationConstraints) -> None:
        # TODO(s5u13b): Remove the migration done callback through decentralized migration refactoring.
        async def migrate_done_callback(ret, migrate_instance_pair: Tuple[str, str]) -> None:
            if isinstance(ret, Exception):
                has_error_pair = await self._check_instance_error(migrate_instance_pair)
                for i, has_error in enumerate(has_error_pair):
                    # Instance without error should clear migration states.
                    instance_id = migrate_instance_pair[i]
                    if not has_error:
                        try:
                            # TODO(s5u13b): Fix the clear_migration_states to adapt to the many-to-many migration.
                            await asyncio_wait_for_with_timeout(
                                self.instances[instance_id].clear_migration_states.remote(is_migrate_in=bool(i))
                            )
                        except Exception as e: # pylint: disable=broad-except
                            if isinstance(e, ray.exceptions.RayActorError):
                                logger.info("Instance {} is dead.".format(instance_id))
                            elif isinstance(e, asyncio.TimeoutError):
                                logger.error("Instance {} is hang, please check the cause.".format(instance_id))
                            else:
                                logger.exception("Failed to clear migration states of instance {}, "
                                                 "unexpected exception: {}".format(instance_id, e))
                            has_error = True
                for i, has_error in enumerate(has_error_pair):
                    if has_error:
                        instance_id = migrate_instance_pair[i]
                        await self.scale_down(instance_id)
            else:
                migrate_out_request_ids = ret
                if not self.enable_pd_disagg:
                    logger.info("Instance {}->{} migrate done, migrate request {}".format(
                        migrate_instance_pair[0], migrate_instance_pair[1], migrate_out_request_ids))

        def migrate_done_callback_wrapper(migrate_instance_pair: Tuple[str, str], fut) -> None:
            ret = fut.result()[0]
            loop = asyncio.get_event_loop()
            loop.create_task(migrate_done_callback(ret, migrate_instance_pair))

        # If encounter error during migration, to make manager keep running, we do not raise exception.
        try:
            migrate_instance_pairs = self.global_scheduler.pair_migration(pair_migration_type)
            migration_tasks = []
            for _, migrate_instance_pair in enumerate(migrate_instance_pairs):
                src_instance_id, dst_instance_id = migrate_instance_pair
                dst_instance_actor_handle = self.instances[dst_instance_id]
                task = asyncio.gather(
                    asyncio_wait_for_with_timeout(
                        self.instances[src_instance_id].migrate_out.remote(
                            dst_instance_id, dst_instance_actor_handle
                        )
                    ),
                    return_exceptions=True
                )
                task.add_done_callback(partial(migrate_done_callback_wrapper, migrate_instance_pair))
                migration_tasks.append(task)
            if len(migration_tasks) > 0 and not self.enable_pd_disagg:
                logger.info("{} migration tasks starts.".format(len(migration_tasks)))
            await asyncio.gather(*migration_tasks, return_exceptions=True)
            if len(migration_tasks) > 0 and not self.enable_pd_disagg:
                logger.info("{} migration tasks ends.".format(len(migration_tasks)))
        # pylint: disable=W0703
        except Exception as e:
            logger.exception("Error during migrate, unexpected exception: {}".format(e))
            logger.critical("Manager encouters error during migrate, manager keeps running, "
                            "please check the cause as soon as possible!")

    def scale_up(self,
                 instance_id: Union[str, Iterable[str]],
                 instance_actor_handle: Union[ray.actor.ActorHandle, Iterable[ray.actor.ActorHandle]],
                 instance_type: Union[InstanceType, Iterable[InstanceType]],
                 placement_group: Union[PlacementGroup, Iterable[PlacementGroup]],
                 server: Union[ray.actor.ActorHandle, Iterable[ray.actor.ActorHandle]] = None) -> None:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
            instance_actor_handle = [instance_actor_handle,]
            instance_type = [instance_type,]
            placement_group = [placement_group,]
            server = [server,] if server is not None else None

        instance_ids = list(instance_id)
        instance_actor_handles: List[Llumlet] = list(instance_actor_handle)
        instance_types = list(instance_type)
        placement_groups = list(placement_group)
        servers = list(server) if server is not None else None

        indeed_update = False
        no_pending_instance = (self.pending_rebuild_migration_instances == 0)

        for idx, ins_id in enumerate(instance_ids):
            if ins_id not in self.instances:
                instance_actor = instance_actor_handles[idx]
                if self.manager_args.enable_engine_pd_disagg:
                    try:
                        self.instance_id_2_engine_disagg_inst_id[ins_id] = \
                            ray_get_with_timeout(instance_actor.get_engine_disagg_inst_id.remote())
                    # pylint: disable=broad-except
                    except Exception as e:
                        if isinstance(e, ray.exceptions.RayActorError):
                            logger.warning("Failed to scale up instance {}, instance is dead.".format(ins_id))
                        elif isinstance(e, ray.exceptions.GetTimeoutError):
                            logger.error("Failed to scale up instance {}, instance is hang, "
                                        "please check the cause.".format(ins_id))
                        else:
                            logger.exception("Error during scale up instance {}, "
                                            "unexpected exception: {}".format(ins_id, e))
                        continue
                    logger.info("Bind instance id {} with engine instance id {}.".format(
                        ins_id, self.instance_id_2_engine_disagg_inst_id[ins_id]))
                indeed_update = True
                self.instances[ins_id] = instance_actor
                self.pgs[ins_id] = placement_groups[idx]
                if self.servers is not None and servers is not None:
                    self.servers[ins_id] = servers[idx]
                if self.log_instance_info:
                    self.instance_last_logged_empty[ins_id] = False
                self.pending_rebuild_migration_instances += 1

        if indeed_update:
            self.global_scheduler.scale_up(instance_ids, instance_types)
            self.num_instances = len(self.instances)

        # When scaling up, we need to rebuild the migration backend. But if initially self.pending_rebuild_migration_instances != 0,
        # a coroutine is already handling the changes in the number of instances in the cluster and it will account for the changes
        # caused by this scale-up (see rebuild_migration_backend for details). Therefore, we simply return in this case.
        # Specifically, for not group kind migration backend, there is no need to rebuild the group.
        if self.enable_migration and self.is_group_kind_migration_backend \
            and indeed_update and no_pending_instance and not self.instance_args.simulator_mode:
            asyncio.create_task(self._rebuild_migration_backend())

        return self.num_instances

    async def scale_down(self, instance_id: Union[str, Iterable[str]], rebuild_migration_backend: bool = True) -> None:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)

        indeed_update = False
        no_pending_instance = self.pending_rebuild_migration_instances == 0

        for ins_id in instance_ids:
            await execute_actor_method_async_with_retries(
                self.scaler.clear_instance_ray_resources.remote, 'Scaler', 'clear_instance_ray_resources', ins_id
            )
            if ins_id in self.instances:
                indeed_update = True
                if ins_id in self.instances:
                    self.instances.pop(ins_id)
                    self.instance_id_2_engine_disagg_inst_id.pop(ins_id, None)
                    if ins_id in self.pgs:
                        self.pgs.pop(ins_id)
                    else:
                        logger.warning("instance {} is not in pgs".format(ins_id))
                    if self.servers:
                        if ins_id in self.servers:
                            self.servers.pop(ins_id)
                        else:
                            logger.warning("instance {} is not in servers".format(ins_id))
                else:
                    logger.warning("instance {} is not in instances".format(ins_id))
                if self.log_instance_info:
                    if ins_id in self.instance_last_logged_empty:
                        del self.instance_last_logged_empty[ins_id]
                    else:
                        logger.warning("instance {} is not in instance_last_logged_empty".format(ins_id))
                self.pending_rebuild_migration_instances += 1

        self.global_scheduler.scale_down(instance_ids)
        self.num_instances = len(self.instances)

        if self.enable_migration and self.is_group_kind_migration_backend:
            if len(self.instances) == 0:
                self.pending_rebuild_migration_instances = 0
                await execute_actor_method_async_with_retries(
                    self.scaler.clear_gloo_backend_ray_resources.remote, 'Scaler', 'clear_gloo_backend_ray_resources',
                )
            elif indeed_update and no_pending_instance and rebuild_migration_backend and not self.instance_args.simulator_mode:
                asyncio.create_task(self._rebuild_migration_backend())

        asyncio.create_task(
            execute_actor_method_async_with_retries(
                self.scaler.clear_instance_ray_resources.remote, 'Scaler', 'clear_instance_ray_resources', instance_ids
            )
        )

        return self.num_instances

    def get_num_prefill_decode_instances(self):
        num_prefill_instances = len(self.global_scheduler.prefill_instance_info)
        num_decode_instances = len(self.global_scheduler.decode_instance_info)

        return num_prefill_instances, num_decode_instances

    def get_prefill_decode_instance_id_set(self):
        prefill_instance_id_set = set(self.global_scheduler.prefill_instance_info.keys())
        decode_instance_id_set = set(self.global_scheduler.decode_instance_info.keys())
        return prefill_instance_id_set, decode_instance_id_set

    async def _rebuild_migration_backend(self) -> None:
        # During rebuilding migration backend, disable migration.
        origin_config = self.enable_migration
        self.enable_migration = False

        # Wait for all instances to finish migration
        while not self.all_instances_not_migrating:
            await asyncio.sleep(WAIT_ALL_MIGRATIONS_DONE_INTERVAL)

        async def run_task(alive_instances: List[str], task_name: str, *args, **kwargs):
            tasks = []
            for instance_name in alive_instances:
                llumlet_handle = self.instances[instance_name]
                tasks.append(
                    asyncio_wait_for_with_timeout(
                        llumlet_handle.execute_engine_method.remote("_run_workers", task_name, *args, **kwargs),
                    )
                )
            rets = await asyncio.gather(*tasks, return_exceptions=True)
            dead_instances = set()
            for instance_name, ret in zip(alive_instances, rets):
                if isinstance(ret, Exception):
                    instance_id = instance_name[:len(INSTANCE_NAME_PREFIX)]
                    if isinstance(ret, ray.exceptions.RayActorError):
                        logger.info("Instance {} is dead.".format(instance_id))
                    elif isinstance(ret, asyncio.TimeoutError):
                        logger.error("Instance {} is hang, please check the cause.".format(instance_id))
                    else:
                        logger.exception("Failed to run task {} for instance {}, "
                                         "unexpected exception: {}".format(task_name, instance_id, ret))
                    dead_instances.add(instance_name)
            if len(dead_instances) > 0:
                await self.scale_down(dead_instances, rebuild_migration_backend=False)
                await execute_actor_method_async_with_retries(
                    self.scaler.clear_gloo_backend_ray_resources.remote, 'Scaler', 'clear_gloo_backend_ray_resources',
                )
            return dead_instances

        alive_instances = sorted(self.instances.keys())
        pending_task = self.pending_rebuild_migration_instances
        group_name = None
        await execute_actor_method_async_with_retries(
            self.scaler.clear_gloo_backend_ray_resources.remote, 'Scaler', 'clear_gloo_backend_ray_resources',
        )

        while len(alive_instances) > 0 and self.pending_rebuild_migration_instances > 0:
            dead_instances = set()
            group_name = random_uuid()
            instance_rank = {instance_id: index for index, instance_id in enumerate(alive_instances)}
            dead_instances.update(await run_task(
                alive_instances, "rebuild_migration_backend", instance_rank, group_name))
            if len(dead_instances) == 0 and self.pending_rebuild_migration_instances == pending_task:
                dead_instances.update(await run_task(alive_instances, "warmup"))
            if len(dead_instances) == 0:
                self.pending_rebuild_migration_instances -= pending_task
            alive_instances = sorted(set(self.instances.keys()) - dead_instances)
            pending_task = self.pending_rebuild_migration_instances

        if len(alive_instances) == 0:
            self.pending_rebuild_migration_instances = 0
            group_name = None

        migration_filter: CustomFilter = \
            self.global_scheduler.migration_scheduler.migration_filter.get_filter("migration_backend_init_filter")
        migration_filter.set_filter_condtition(
            src_filter=lambda instance_info: instance_info.instance_id in alive_instances,
            dst_filter=lambda instance_info: instance_info.instance_id in alive_instances,
        )

        logger.info("Rebuild migration backend done, group_name: {}, alive instance ({}): {}."
            .format(group_name, len(alive_instances), alive_instances))

        # Restore migrate config
        self.enable_migration = origin_config

    def _update_all_instances_not_migrating(self, instance_infos: List[InstanceInfo]) -> None:
        instance_migrating: Dict[str, Optional[bool]] = {ins_info.instance_id: ins_info.migrating for ins_info in instance_infos}
        self.all_instances_not_migrating = not any(
            instance_id not in instance_migrating or instance_migrating[instance_id]
                for instance_id in self.instances
        )

    async def _connect_to_instances(self):
        def connect_to_instance_done_callback(instance_id: str, instance_actor_handle: Llumlet, fut):
            ret = fut.result()[0]
            if not isinstance(ret, Exception):
                try:
                    placement_groups.append(ray.util.get_placement_group(get_placement_group_name(instance_id)))
                    instance_ids.append(instance_id)
                    instances.append(instance_actor_handle)
                    instance_types.append(ret)
                    logger.info("Connect to instance {}".format(instance_id))
                except Exception as e: # pylint: disable=broad-except
                    if isinstance(e, ValueError):
                        logger.warning("Failed to connect to instance {}, placement group not found.".format(instance_id))
                    else:
                        logger.exception("Error during connect to instance {}, "
                                         "unexpected exception: {}".format(instance_id, e))
            else:
                if isinstance(ret, ray.exceptions.RayActorError):
                    logger.warning("Failed to connect to instance {}, instance is dead.".format(instance_id))
                elif isinstance(ret, asyncio.TimeoutError):
                    logger.error("Failed to connect to instance {}, instance is hang, "
                                 "please check the cause.".format(instance_id))
                else:
                    logger.exception("Error during connect to instance {}, "
                                     "unexpected exception: {}".format(instance_id, ret))

        # Must set True despite set namespance to llumnix.
        actor_infos = ray.util.list_named_actors(all_namespaces=True)
        instance_actor_names = [actor_info['name'] for actor_info in actor_infos
                                if actor_info['name'].startswith(INSTANCE_NAME_PREFIX)]
        available_instance_actor_names = []
        available_instance_actor_handles: List[Llumlet] = []
        for actor_name in instance_actor_names:
            try:
                instance_actor_handle = ray.get_actor(actor_name, namespace='llumnix')
                available_instance_actor_names.append(actor_name)
                available_instance_actor_handles.append(instance_actor_handle)
            except Exception as e: # pylint: disable=broad-except
                instance_id = actor_name[len(INSTANCE_NAME_PREFIX):]
                if isinstance(e, ValueError):
                    logger.warning("Failed to connect to instance {}, actor not found.".format(instance_id))
                else:
                    logger.exception("Error during connect to instance {}, "
                                     "unexpected exception: {}".format(instance_id, e))
        instance_ids = []
        instances = []
        instance_types = []
        placement_groups = []
        tasks = []
        for instance_actor_name, instance_actor_handle in \
            zip(available_instance_actor_names,available_instance_actor_handles):
            instance_id = instance_actor_name[len(INSTANCE_NAME_PREFIX):]
            if instance_id not in self.instances:
                task = asyncio.gather(
                    asyncio_wait_for_with_timeout(instance_actor_handle.get_instance_type.remote()),
                    return_exceptions=True
                )
                task.add_done_callback(
                    partial(connect_to_instance_done_callback, instance_id, instance_actor_handle)
                )
                tasks.append(task)
        await asyncio.gather(*tasks)
        # The only function that can add instance actor handles to manager.
        self.scale_up(instance_ids, instances, instance_types, placement_groups)

    async def _check_instance_error(self, migrate_instance_pairs: Tuple[str, str]) -> List[bool]:
        def check_instance_error_done_callback(idx: int, instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, Exception):
                logger.info("Instance {} is alive.".format(instance_id))
                results[idx] = False
            else:
                if isinstance(ret, ray.exceptions.RayActorError):
                    logger.info("Instance {} is dead.".format(instance_id))
                elif isinstance(ret, asyncio.TimeoutError):
                    logger.error("Instance {} is hang, please check the cause.".format(instance_id))
                else:
                    logger.exception("Failed to check instance {} error, "
                                     "unexpected exception: {}".format(instance_id, ret))
                results[idx] = True

        results = [None, None]
        tasks = []
        for idx, instance_id in enumerate(migrate_instance_pairs):
            task = asyncio.gather(
                asyncio_wait_for_with_timeout(self.instances[instance_id].is_ready.remote()),
                return_exceptions=True
            )
            task.add_done_callback(partial(check_instance_error_done_callback, idx, instance_id))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

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
            'waiting_time_first_waiting_request',
        ])

    def _log_instance_infos_to_csv(self, instance_infos: List[InstanceInfo]) -> None:
        for instance_info in instance_infos:
            instance_id = instance_info.instance_id
            gpu_cache_usage = instance_info.gpu_cache_usage
            should_log = (gpu_cache_usage > 0) or (gpu_cache_usage == 0 and \
                                                   instance_id in self.instance_last_logged_empty and \
                                                   not self.instance_last_logged_empty[instance_id])
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
                    instance_info.waiting_time_first_waiting_request
                ])
        self.instance_info_file.flush()
