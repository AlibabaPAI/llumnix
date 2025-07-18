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
import csv
import os
from typing import Dict, List, Tuple, Union, Iterable
from functools import partial

import ray
import ray.actor

from llumnix.llumlet.llumlet import Llumlet
from llumnix.logging.logger import init_logger
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.global_scheduler.migration_filter import CustomFilter
from llumnix.instance_info import InstanceInfo
from llumnix.arg_utils import (
    ManagerArgs,
    EntrypointsArgs,
    InstanceArgs,
    LaunchArgs,
    LlumnixEngineArgs,
)
from llumnix.metrics.manager_metrics import ManagerMetrics
from llumnix.request_processing_context import RequestProcessingContext
from llumnix.utils import (
    random_uuid,
    asyncio_wait_for_ray_remote_call_with_timeout,
    RequestIDType,
    log_instance_exception,
    BackendType,
    LaunchMode,
    InstanceType,
    InstanceContext,
)
from llumnix.ray_utils import (
    get_manager_name,
    log_actor_ray_info,
    get_scaler_name,
)
from llumnix.constants import (
    NO_INSTANCE_RETRY_GENERATE_INTERVAL,
    WAIT_ALL_MIGRATIONS_DONE_INTERVAL,
    MAX_ACTOR_METHOD_RETRIES,
)

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
        self.launch_mode: LaunchMode = launch_args.launch_mode
        self.backend_type: BackendType = launch_args.backend_type
        logger.info("Launch mode: {}, backend type: {}".format(self.launch_mode, self.backend_type))

        # migration args
        self.enable_migration = manager_args.enable_migration
        self.pair_migration_frequency = manager_args.pair_migration_frequency
        self.is_group_kind_migration_backend = manager_args.is_group_kind_migration_backend

        # prefill-decode disaggregation args
        self.enable_pd_disagg = manager_args.enable_pd_disagg

        # scheduling states
        self.instance_context: Dict[str, InstanceContext] = {}
        self.polling_interval = manager_args.polling_interval
        global_scheduler_config = manager_args.create_global_scheduler_config()
        self.global_scheduler = GlobalScheduler(global_scheduler_config)

        # log args
        self.log_instance_info = manager_args.log_instance_info
        if self.log_instance_info:
            self._init_instance_info_csv(manager_args)
            self.instance_last_logged_empty = {}

        # instance states
        self.num_instances = 0
        self.instances: Dict[str, Llumlet] = {}
        self.pending_rebuild_migration_instances = 0

        # migration states
        self.num_instance_info_updates = 0
        self.migrating = False

        # metrics
        self.manager_metrics = ManagerMetrics()

        asyncio.create_task(self._poll_instance_info_loop(self.polling_interval))

    async def generate(self, request_id: RequestIDType, request_processing_context: RequestProcessingContext, *args, **kwargs) -> Tuple[str, str]:
        def choose_destination_instance(prefill_instance_id: str, decode_instance_id: str, dispatch_kwargs: Dict):
            if self.backend_type == BackendType.BLADELLM:
                if self.manager_args.enable_engine_pd_disagg:
                    dispatch_kwargs["decode_instance_id"] = self.instance_context[decode_instance_id].local_engine_id
                elif self.manager_args.enable_engine_semi_pd_disagg:
                    dispatch_kwargs["semi_p_inst_id"] = self.instance_context[prefill_instance_id].local_engine_id
                    dispatch_kwargs["semi_d_inst_id"] = self.instance_context[decode_instance_id].local_engine_id

            if self.backend_type == BackendType.VLLM_V1 and self.manager_args.enable_engine_pd_disagg:
                dispatch_kwargs["llumnix_scheduler"] = True
                dispatch_kwargs["prefill_kvt_engine_available_port"] = \
                    self.instance_context[prefill_instance_id].kvt_engine_available_port
                dispatch_kwargs["prefill_engine_host"] = self.instance_context[prefill_instance_id].engine_host
                dispatch_kwargs["prefill_instance_id"] = prefill_instance_id
                dispatch_kwargs["decode_instance_id"] = decode_instance_id

            if (self.backend_type == BackendType.BLADELLM and self.manager_args.enable_engine_semi_pd_disagg) or \
                (self.backend_type == BackendType.VLLM_V1 and self.manager_args.enable_engine_pd_disagg):
                target_instance_id = decode_instance_id
            else:
                target_instance_id = prefill_instance_id
            return target_instance_id

        self.manager_metrics.manager_request_qps.increase(
            labels={"server_id": request_processing_context.server_id}
        )
        while self.num_instances == 0:
            logger.warning("No instance available now, sleep {}s, and regenerate request {}.".format(
                NO_INSTANCE_RETRY_GENERATE_INTERVAL, request_id))
            await asyncio.sleep(NO_INSTANCE_RETRY_GENERATE_INTERVAL)

        prefill_instance_id, decode_instance_id, request_expected_steps = \
            self.global_scheduler.dispatch(request_id, dispatch_context=kwargs)
        target_instance_id = choose_destination_instance(prefill_instance_id, decode_instance_id, kwargs)
        request_processing_context.add_trace_timeline('manager_generate_timestamp')
        asyncio.create_task(
            self._generate_with_exception_handling(
                request_id, request_processing_context, request_expected_steps, target_instance_id, *args, **kwargs
            )
        )
        return prefill_instance_id, decode_instance_id

    async def _generate_with_exception_handling(self,
                                                request_id: RequestIDType,
                                                request_processing_context: RequestProcessingContext,
                                                request_expected_steps: int,
                                                target_instance_id: str,
                                                *args, **kwargs):
        try:
            await asyncio_wait_for_ray_remote_call_with_timeout(
                self.instances[target_instance_id].generate,
                request_id, request_processing_context, request_expected_steps, *args, **kwargs
            )
        # pylint: disable=broad-except
        except Exception as e:
            log_instance_exception(e, target_instance_id, "generate", request_id)
            self.scale_down(target_instance_id)
            await asyncio.create_task(self.generate(request_id, request_processing_context, *args, **kwargs))

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
        )(cls)
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
            asyncio_wait_for_ray_remote_call_with_timeout(instance.is_ready)
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
                log_instance_exception(ret, instance_id, "_poll_instance_info_loop")
                self.scale_down(instance_id)

        def get_instance_info_done_callback_wrapper(instance_id: str, fut) -> None:
            ret = fut.result()[0]
            loop = asyncio.get_event_loop()
            loop.create_task(get_instance_info_done_callback(ret, instance_id))

        while True:
            try:
                await asyncio.sleep(interval)
                tasks = []
                instance_infos = []
                for instance_id, instance in self.instances.items():
                    # Use asyncio.gather to wrap ray remote call to add done callback, asyncio.create_task will get error.
                    task = asyncio.gather(
                        asyncio_wait_for_ray_remote_call_with_timeout(instance.get_instance_info),
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
                # Push migrate when the instance_info have updated a certain number of times.
                if self.enable_migration and self.num_instance_info_updates != 0 \
                    and self.num_instance_info_updates % self.pair_migration_frequency == 0:
                    asyncio.create_task(self._migrate())
                if self.log_instance_info:
                    self._log_instance_infos_to_csv(instance_infos)
            # pylint: disable=W0703
            except Exception:
                logger.critical(
                    "Manager get error in _poll_instance_info_loop, manager keeps running, please check the cause!",
                    exc_info=True, stack_info=True
                )

    async def _migrate(self):
        # If encounter error during migration, to make manager keep running, we do not raise exception.
        try:
            instance_infos = self.global_scheduler.instance_info
            migrate_tasks = self.global_scheduler.push_migrations(instance_infos)

            for task in migrate_tasks:
                migration_type, migrate_pairs = task
                for src_instance_id, dst_instance_id in migrate_pairs:
                    dst_instance_actor = self.instances[dst_instance_id]
                    asyncio.create_task(
                        asyncio_wait_for_ray_remote_call_with_timeout(
                            self.instances[src_instance_id].migrate_out,
                            dst_instance_actor, dst_instance_id, migration_type
                        )
                    )
        # pylint: disable=broad-except
        except Exception:
            logger.critical(
                "Manager get error in _migrate, manager keeps running, please check the cause!",
                exc_info=True, stack_info=True
            )

    async def _get_engine_context(self, ins_id: str, instance_actor: Llumlet) -> InstanceContext:
        try:
            engine_context = await asyncio_wait_for_ray_remote_call_with_timeout(
                instance_actor.get_engine_context)
            self.instance_context[ins_id] = engine_context
            logger.info("Bind instance id {} with engine context {}.".format(ins_id, engine_context))
            return engine_context
        # pylint: disable=broad-except
        except Exception as e:
            log_instance_exception(e, ins_id, "scale_up")
            raise e

    async def _get_available_instances(self,
                                instance_ids: List[str],
                                instance_actor_handles: List[Llumlet],
                                instance_types: List[InstanceType]
                                ) -> List[str]:
        available_instance_ids, available_instance_actors, available_instance_types = [], [], []

        def self_assign_id_success_callback(fut,
                                            instance_idx: int,
                                            scale_up_info: List[List],
                                            available_scale_up_info: List[List]):
            ret = fut.result()[0]
            if not isinstance(ret, Exception):
                for item_idx, scale_up_info_item in enumerate(scale_up_info):
                    available_scale_up_info[item_idx].append(scale_up_info_item[instance_idx])

        tasks = []
        for ins_idx, ins_info in enumerate(zip(instance_ids, instance_actor_handles)):
            ins_id, ins_actor = ins_info
            if ins_id not in self.instances:
                task = asyncio.gather(self._get_engine_context(ins_id, ins_actor), return_exceptions=True)
                task.add_done_callback(
                    partial(
                        self_assign_id_success_callback,
                        instance_idx=ins_idx,
                        scale_up_info=[instance_ids, instance_actor_handles, instance_types],
                        available_scale_up_info=[available_instance_ids, available_instance_actors, available_instance_types])
                    )
                tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

        return available_instance_ids, available_instance_actors, available_instance_types

    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    async def scale_up(self,
                       instance_id: Union[str, Iterable[str]],
                       instance_actor_handle: Union[ray.actor.ActorHandle, Iterable[ray.actor.ActorHandle]],
                       instance_type: Union[InstanceType, Iterable[InstanceType]]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
            instance_actor_handle = [instance_actor_handle,]
            instance_type = [instance_type,]

        instance_ids: List[str] = list(instance_id)
        instance_actor_handles: List[Llumlet] = list(instance_actor_handle)
        instance_types: List[InstanceType] = list(instance_type)

        no_pending_instance = (self.pending_rebuild_migration_instances == 0)
        indeed_update = not set(instance_ids).issubset(self.instances.keys())

        if indeed_update:
            available_instance_ids, available_instance_actors, available_instance_types = \
                await self._get_available_instances(instance_ids, instance_actor_handles, instance_types)

            self.global_scheduler.scale_up(available_instance_ids, available_instance_types)

            for idx, ins_id in enumerate(available_instance_ids):
                if ins_id not in self.instances:
                    instance_actor = available_instance_actors[idx]
                    self.instances[ins_id] = instance_actor
                    if self.log_instance_info:
                        self.instance_last_logged_empty[ins_id] = False
                    self.pending_rebuild_migration_instances += 1

            self.num_instances = len(self.instances)

        # When scaling up, we need to rebuild the migration backend. But if initially self.pending_rebuild_migration_instances != 0,
        # a coroutine is already handling the changes in the number of instances in the cluster and it will account for the changes
        # caused by this scale-up (see rebuild_migration_backend for details). Therefore, we simply return in this case.
        # Specifically, for not group kind migration backend, there is no need to rebuild the group.
        if self.enable_migration and self.is_group_kind_migration_backend \
            and indeed_update and no_pending_instance and not self.instance_args.simulator_mode:
            asyncio.create_task(self._rebuild_migration_backend())

        return self.num_instances

    @ray.method(max_task_retries=MAX_ACTOR_METHOD_RETRIES)
    def scale_down(self, instance_id: Union[str, Iterable[str]], rebuild_migration_backend: bool = True) -> None:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids: List[str] = list(instance_id)

        indeed_update = False
        no_pending_instance = self.pending_rebuild_migration_instances == 0

        for ins_id in instance_ids:
            if ins_id in self.instances:
                indeed_update = True
                self.pending_rebuild_migration_instances += 1
                self.instances.pop(ins_id)
                self.instance_context.pop(ins_id, None)
            else:
                logger.warning("instance {} is not in instances".format(ins_id))

            if self.log_instance_info:
                if ins_id in self.instance_last_logged_empty:
                    del self.instance_last_logged_empty[ins_id]
                else:
                    logger.warning("instance {} is not in instance_last_logged_empty".format(ins_id))

        if indeed_update:
            self.global_scheduler.scale_down(instance_ids)
            self.num_instances = len(self.instances)

        asyncio.create_task(
            asyncio_wait_for_ray_remote_call_with_timeout(
                self.scaler.scale_down, instance_ids,
                exc_handling=True,
            )
        )

        if self.enable_migration and self.is_group_kind_migration_backend:
            if len(self.instances) == 0:
                self.pending_rebuild_migration_instances = 0
                asyncio.create_task(
                    asyncio_wait_for_ray_remote_call_with_timeout(
                        self.scaler.clear_gloo_backend_ray_resources,
                        exc_handling=True,
                    )
                )
            elif indeed_update and no_pending_instance and rebuild_migration_backend and not self.instance_args.simulator_mode:
                asyncio.create_task(self._rebuild_migration_backend())

        return self.num_instances

    async def _rebuild_migration_backend(self) -> None:
        # During rebuilding migration backend, disable migration.
        origin_config = self.enable_migration
        self.enable_migration = False

        # Wait for all instances to finish migration
        while not self.global_scheduler.all_instances_not_migrating():
            await asyncio.sleep(WAIT_ALL_MIGRATIONS_DONE_INTERVAL)

        async def run_task(alive_instances: List[str], task_name: str, *args, **kwargs):
            tasks = []
            dead_instances = set()

            for instance_id in alive_instances:
                llumlet_handle = self.instances.get(instance_id, None)
                if llumlet_handle is not None:
                    tasks.append(
                        asyncio_wait_for_ray_remote_call_with_timeout(
                            llumlet_handle.execute_engine_method_async, "_run_workers_async", task_name, *args, **kwargs
                        )
                    )
                else:
                    dead_instances.add(instance_id)
                    break

            if len(dead_instances) == 0:
                rets = await asyncio.gather(*tasks, return_exceptions=True)
                for instance_id, ret in zip(alive_instances, rets):
                    if isinstance(ret, Exception):
                        log_instance_exception(ret, instance_id, "_rebuild_migration_backend")
                        dead_instances.add(instance_id)

            if len(dead_instances) > 0:
                self.scale_down(dead_instances, rebuild_migration_backend=False)
                await asyncio_wait_for_ray_remote_call_with_timeout(self.scaler.clear_gloo_backend_ray_resources)
            return dead_instances

        await asyncio_wait_for_ray_remote_call_with_timeout(self.scaler.clear_gloo_backend_ray_resources)
        alive_instances = sorted(self.instances.keys())
        pending_task = self.pending_rebuild_migration_instances
        group_name = None

        while len(alive_instances) > 0 and self.pending_rebuild_migration_instances > 0:
            dead_instances = set()
            group_name = random_uuid()
            instance_rank = {instance_id: index for index, instance_id in enumerate(alive_instances)}
            # TODO(KuilongCui): Fix potential self.instances update between bellow two awaits.
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
            self.global_scheduler.migration_scheduler.migration_base_filter.get_filter("migration_backend_init_filter")
        migration_filter.set_filter_condtition(
            src_filter=lambda instance_info: instance_info.instance_id in alive_instances,
            dst_filter=lambda instance_info: instance_info.instance_id in alive_instances,
        )

        logger.info(
            "Rebuild migration backend done, group_name: {}, alive instance ({}): {}.".format(
                group_name, len(alive_instances), alive_instances
            )
        )

        # Restore migrate config
        self.enable_migration = origin_config

    async def _connect_to_instances(self):
        instance_ids, instances, instance_types = \
            await asyncio_wait_for_ray_remote_call_with_timeout(self.scaler.get_instances)
        await self.scale_up(instance_ids, instances, instance_types)

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
            'kv_blocks_ratio',
            'remaining_steps',
            'adaptive_decode',
            'miss_waiting_tokens',
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
                    instance_info.waiting_time_first_waiting_request,
                    instance_info.kv_blocks_ratio,
                    instance_info.remaining_steps,
                    instance_info.adaptive_decode,
                    instance_info.miss_waiting_tokens,
                ])
        self.instance_info_file.flush()
