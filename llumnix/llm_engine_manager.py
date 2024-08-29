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
import ray

from llumnix.llumlet.llumlet import Llumlet
from llumnix.logger import init_logger
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.instance_info import InstanceInfo
from llumnix.config import GlobalSchedulerConfig
from llumnix.arg_utils import EngineManagerArgs
from llumnix.backends.profiling import ProfilingDatabase
from llumnix.server_info import ServerInfo
from llumnix.backends.backend_interface import BackendType
from llumnix.utils import random_uuid


logger = init_logger(__name__)

MANAGER_ACTOR_NAME = 'manager'
CLEARING_INTERVAL = 3600
RETRIES_INTERVALS = 5.0

# TODO(s5u13b): Fix the logger when manager failover.


class LLMEngineManager:
    def __init__(self,
                 engine_manager_args: EngineManagerArgs,
                 global_scheduler_config: GlobalSchedulerConfig,
                 work_dir: str,
                 log_requests: bool = True,
                 profiling_database: ProfilingDatabase = None) -> None:
        os.chdir(work_dir)
        self.actor_name = MANAGER_ACTOR_NAME
        self.engine_manager_args = engine_manager_args
        self.profiling_database = profiling_database

        self.log_requests = log_requests

        self.num_instances = 0
        self.enable_migration = engine_manager_args.enable_migration
        self.enable_scaling = engine_manager_args.enable_scaling
        self.max_instances = engine_manager_args.max_instances
        self.min_instances = engine_manager_args.min_instances

        logger.info("LLMEngineManager starts")
        logger.info("enable_migration: {}".format(self.enable_migration))
        logger.info("num_instances: {}".format(self.num_instances))
        logger.info("max_instances: {}, min_instances: {}".format(self.max_instances, self.min_instances))

        # TODO(s5u13b): refactor auto-scaling

        self.instances: Dict[str, Llumlet] = {}
        self.instance_migrating: Dict[str, bool] = {}
        self.pending_rebuild_migration_instances = 0
        self.global_scheduler = GlobalScheduler(global_scheduler_config)
        # When manager starts, it automatically connects to all existing instances.
        self._connect_to_instances()

        self.polling_interval = engine_manager_args.polling_interval
        asyncio.create_task(self._update_instance_info_loop(self.polling_interval))

        # args
        self.pair_migration_frequency = engine_manager_args.pair_migration_frequency
        self.scaling_interval = engine_manager_args.scaling_interval

        # request states
        self.request_instance: Dict[str, str] = {}
        self.clearing_interval = CLEARING_INTERVAL
        asyncio.create_task(self._clear_request_instance_loop(self.clearing_interval))

        # migrate states
        self.num_instance_info_updates = 0
        self.migrating = False

        # auto-scaling states
        self.scale_up_time = -1
        self.scale_down_time = -1
        self.scaling_up = False
        self.scaling_down = False
        self.last_check_scale_time = time.time() + 100

        self.log_instance_info = engine_manager_args.log_instance_info
        if self.log_instance_info:
            self._init_instance_info_csv(engine_manager_args)

    async def generate(
            self,
            request_id: str,
            server_info: ServerInfo,
            *args,
            **kwargs,) -> None:
        while self.num_instances == 0:
            logger.info("No instance available temporarily, sleep {}s, "
                        "and retry generate request {} again....".format(RETRIES_INTERVALS, request_id))
            await asyncio.sleep(RETRIES_INTERVALS)
        instance_id = self.global_scheduler.dispatch()
        try:
            await self.instances[instance_id].generate.remote(request_id, server_info, *args, **kwargs)
            if self.log_requests:
                logger.info("received request {}.".format(request_id))
                logger.info("dispath to instance {}".format(instance_id))
                self.request_instance[request_id] = instance_id
        except (ray.exceptions.RayActorError, KeyError):
            logger.info("[generate] instance {} is dead, regenerate request {}".format(instance_id, request_id))
            self.scale_down(instance_id)

    async def abort(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        instance_requests = defaultdict(list)
        for req_id in request_ids:
            if req_id in self.request_instance:
                instance_id = self.request_instance[req_id]
                instance_requests[instance_id].append(req_id)
        for instance_id, request_ids in instance_requests.items():
            try:
                # Requests will be free by instance when finished, so it is acceptable to miss aborted requests.
                await self.instances[instance_id].abort.remote(request_ids)
                if self.log_requests:
                    logger.info("abort requests: {}.".format(request_ids))
                    for req_id in request_ids:
                        if req_id in self.request_instance:
                            del self.request_instance[req_id]
            except (ray.exceptions.RayActorError, KeyError):
                logger.info("[abort] instance {} is dead".format(instance_id))
                self.scale_down(instance_id)

    async def _get_request_instance(self) -> None:
        logger.info("_get_request_instance:")
        tasks = [instance_actor_handle.get_all_request_ids.remote() for instance_actor_handle in self.instances.values()]
        instance_ids = list(self.instances.keys())
        rets = await asyncio.gather(*tasks, return_exceptions=True)
        instance_requests = []
        instance_id_list = []
        for idx, ret in enumerate(rets):
            if not isinstance(ret, ray.exceptions.RayActorError):
                instance_requests.append(ret)
                instance_id_list.append(instance_ids[idx])
            else:
                instance_id = instance_ids[idx]
                logger.info("[_get_request_instance] instance {} is dead".format(instance_id))
                self.scale_down(instance_id)
        logger.info("instance_id_list: {}".format(instance_id_list))
        logger.info("instance_requests: {}".format(instance_requests))
        for (instance_id, requests) in zip(instance_id_list, instance_requests):
            for request_id in requests:
                self.request_instance[request_id] = instance_id

    async def _update_instance_info_loop(self, interval: float) -> None:
        while True:
            try:
                await asyncio.sleep(interval)
                tasks = [instance.get_instance_info.remote() for instance in self.instances.values()]
                instance_ids = list(self.instances.keys())
                rets = await asyncio.gather(*tasks, return_exceptions=True)

                instance_info_list = []
                dead_instances = []
                for idx, ret in enumerate(rets):
                    if not isinstance(ret, ray.exceptions.RayActorError):
                        if ret is not None:
                            instance_info_list.append(ret)
                    else:
                        instance_id = instance_ids[idx]
                        dead_instances.append(instance_id)
                if len(dead_instances) > 0:
                    logger.info("[_update_instance_info_loop] dead instances: {}.".format(dead_instances))
                    self.scale_down(dead_instances)

                self.global_scheduler.update_instance_infos(instance_info_list)
                self.num_instance_info_updates += 1
                # Push migrate when the instance_info have updated a certain number of times.
                if self.enable_migration and self.num_instance_info_updates != 0 \
                    and self.num_instance_info_updates % self.pair_migration_frequency == 0:
                    asyncio.create_task(self._migrate())
                if self.log_instance_info:
                    self._log_instance_infos_to_csv(instance_info_list)
            # pylint: disable=W0703
            except Exception as e:
                logger.error("unexpected exception occurs: {}".format(e))
                logger.error("exception traceback: {}".format(traceback.format_exc()))

    async def _clear_request_instance_loop(self, interval: float):
        await self._get_request_instance()
        # Clear the request_instance at a certain interval to prevent memory leaking.
        while True:
            await asyncio.sleep(interval)
            self.request_instance = {}

    async def _post_migrate(self, rets: List[str], call_migrate_instance_pairs: List[Tuple[str, str]]) -> None:
        for i, ret in enumerate(rets):
            self.instance_migrating[call_migrate_instance_pairs[i][0]] = False
            self.instance_migrating[call_migrate_instance_pairs[i][1]] = False
            if isinstance(ret, (ray.exceptions.RayActorError, KeyError)):
                has_error_pair = await self._check_instance_error(call_migrate_instance_pairs[i])
                for j, has_error in enumerate(has_error_pair):
                    # Instance without error should clear migration states.
                    if not has_error:
                        try:
                            await self.instances[call_migrate_instance_pairs[i][j]].clear_migration_states.remote(is_migrate_in=bool(j))
                        except (ray.exceptions.RayActorError, KeyError):
                            has_error = True
                for j, has_error in enumerate(has_error_pair):
                    if has_error:
                        instance_id = call_migrate_instance_pairs[i][j]
                        logger.info("[_migrate] instance {} is dead".format(instance_id))
                        self.scale_down(instance_id)
            else:
                migrate_out_request_ids = ret
                if migrate_out_request_ids:
                    migrate_out_request_id = migrate_out_request_ids[0]
                    self.request_instance[migrate_out_request_id] = call_migrate_instance_pairs[i][1]
                logger.info("{}->{} migrate done, migrate request {}".format(
                    call_migrate_instance_pairs[i][0], call_migrate_instance_pairs[i][1], migrate_out_request_ids))

    async def _migrate(self) -> None:
        migrate_instance_pairs = self.global_scheduler.pair_migration()
        try:
            migration_tasks = []
            call_migrate_instance_pairs: List[Tuple[str, str]] = []
            for _, migrate_instance_pair in enumerate(migrate_instance_pairs):
                migrate_out_instance_id, migrate_in_instance_id = migrate_instance_pair
                if self.instance_migrating[migrate_out_instance_id] or self.instance_migrating[migrate_in_instance_id]:
                    continue
                self.instance_migrating[migrate_out_instance_id] = True
                self.instance_migrating[migrate_in_instance_id] = True
                migrate_in_instance_name = "instance_{}".format(migrate_in_instance_id)
                logger.info("{}->{} begin migrate out".format(migrate_out_instance_id, migrate_in_instance_id))
                call_migrate_instance_pairs.append(migrate_instance_pair)
                task = self.instances[migrate_out_instance_id].migrate_out.remote(migrate_in_instance_name)
                migration_tasks.append(task)
            # TODO(s5u13b): Migration failover could be implemented in Llumlet rather than manager.
            rets = await asyncio.gather(*migration_tasks, return_exceptions=True)
            await self._post_migrate(rets, call_migrate_instance_pairs)
        # pylint: disable=W0703
        except Exception as e:
            logger.error("unexpected exception occurs: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))

    async def rebuild_migrate_backend(self) -> None:
        # Wait for all instances to finish migration
        while any(self.instance_migrating.values()):
            await asyncio.sleep(0.1)

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

                if self.engine_manager_args.migration_backend == 'gloo':
                    try:
                        # clear gloo migrate backend intermediate state
                        ray.kill(ray.get_actor("gloo_queue", "llumnix"))
                    except ValueError:
                        # gloo_queue may not have been created yet; just ignore this error.
                        pass

            return dead_instances

        alive_instances = sorted(self.instances.keys())
        pending_task = self.pending_rebuild_migration_instances
        group_name = None

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

        logger.info("rebuild {} migrate backend done, group_name: {}, alive instance ({}): {}"
            .format(self.engine_manager_args.migration_backend, group_name, len(alive_instances), alive_instances))

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
                self.pending_rebuild_migration_instances += 1
        self.global_scheduler.scale_up(instance_ids)
        self.num_instances = len(self.instances)

        # When scaling up, we need to rebuild the migration backend. But if initially self.pending_rebuild_migration_instances != 0,
        # a coroutine is already handling the changes in the number of instances in the cluster and it will account for the changes
        # caused by this scale-up (see rebuild_migrate_backend for details). Therefore, we simply return in this case. Specifically,
        # for RPC, the Ray actor handle is used for the migration cache, so there is no need to rebuild the group.
        if self.engine_manager_args.migration_backend != 'rpc' and indeed_update and no_pending_instance:
            asyncio.create_task(self.rebuild_migrate_backend())

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
                del self.instances[ins_id]
                del self.instance_migrating[ins_id]
                self.pending_rebuild_migration_instances += 1
        self.global_scheduler.scale_down(instance_ids)
        self.num_instances = len(self.instances)

        if self.engine_manager_args.migration_backend != 'rpc':
            if len(self.instances) == 0:
                self.pending_rebuild_migration_instances = 0
            elif indeed_update and no_pending_instance and rebuild_migrate_backend:
                asyncio.create_task(self.rebuild_migrate_backend())

        return self.num_instances

    def _connect_to_instances(self):
        actor_names_dict = ray.util.list_named_actors(True)
        instance_actor_names = [actor_name_dict['name'] for actor_name_dict in actor_names_dict if actor_name_dict['name'] != MANAGER_ACTOR_NAME]
        instance_actor_handles = [ray.get_actor(actor_name, namespace='llumnix') for actor_name in instance_actor_names]
        scale_up_instance_ids = []
        scale_up_instance_actor_handles = []
        for instance_actor_name, instance_actor_handle in zip(instance_actor_names, instance_actor_handles):
            instance_id = instance_actor_name[len('instance_'):]
            if instance_id not in self.instances:
                try:
                    ray.get(instance_actor_handle.is_ready.remote())
                # pylint: disable=W0703
                except Exception as e:
                    logger.info("connect to instance {} abort, which may be not ready or alive, err: {}".format(instance_id, e))
                    continue
                logger.info("connect to instance {}.".format(instance_id))
                scale_up_instance_ids.append(instance_id)
                scale_up_instance_actor_handles.append(instance_actor_handle)
        # The only function that can add instance actor handles to manager.
        self.scale_up(scale_up_instance_ids, scale_up_instance_actor_handles)

    async def _check_instance_error(self, migrate_instance_pairs: Tuple[str, str]) -> List[bool]:
        results = [None, None]
        for idx, instance_id in enumerate(migrate_instance_pairs):
            try:
                await self.instances[instance_id].is_ready.remote()
                logger.info("[_check_instance_error] instance {} is alive".format(instance_id))
                results[idx] = False
            except (ray.exceptions.RayActorError, KeyError):
                logger.info("[_check_instance_error] instance {} is dead".format(instance_id))
                results[idx] = True
        return results

    @classmethod
    def from_args(cls,
                  engine_manager_args: EngineManagerArgs,
                  profiling_database: ProfilingDatabase=None) -> "LLMEngineManager":
        global_scheduler_config = engine_manager_args.create_global_scheduler_configs()
        # Init manager actor in 'llumnix' namespace to ensure that only one manager can be created.
        manager_class = ray.remote(num_cpus=0,
                                   max_restarts=-1,
                                   name=MANAGER_ACTOR_NAME,
                                   namespace='llumnix',
                                   lifetime="detached"
                                   )(cls)
        engine_manager = manager_class.remote(engine_manager_args,
                                              global_scheduler_config,
                                              os.getcwd(),
                                              log_requests=not engine_manager_args.disable_log_requests_manager,
                                              profiling_database=profiling_database)
        logger.info("engine_manager_args: {}".format(engine_manager_args))
        return engine_manager

    # TODO(s5u13b): Significant duplication with llumlet_utils.init_llumlets. Consider reducing duplicate codes.
    # TODO(s5u13b): Fix the logger when enabling init instance by manager.
    def init_llumlets(self,
                      engine_args,
                      node_id: str) -> Tuple[List[str], List[Llumlet]]:
        engine_manager_args = self.engine_manager_args
        engine_config = engine_args.create_engine_config()
        parallel_config = engine_config.parallel_config
        instance_ids: List[str] = []
        llumlets: List[Llumlet] = []
        for _ in range(engine_manager_args.initial_instances):
            instance_id = random_uuid()
            if not engine_manager_args.profiling_result_file_path:
                llumlet = Llumlet.from_args(
                    engine_manager_args.disable_fixed_node_init_instance,
                    True,
                    node_id,
                    instance_id,
                    BackendType.VLLM,
                    parallel_config.world_size,
                    engine_manager_args.create_migration_config(),
                    engine_args,
                )
            else:
                llumlet = Llumlet.from_args(
                    engine_manager_args.disable_fixed_node_init_instance,
                    True,
                    node_id,
                    instance_id,
                    BackendType.SIM_VLLM,
                    parallel_config.world_size,
                    engine_manager_args.create_migration_config(),
                    engine_manager_args.profiling_result_file_path,
                    engine_manager_args.gpu_type,
                    engine_args,
                )
            instance_ids.append(instance_id)
            llumlets.append(llumlet)

        return instance_ids, llumlets

    def get_actor_name(self) -> str:
        return self.actor_name

    async def is_ready(self) -> bool:
        """Called by api server, return true when all the instances have been successfully created."""
        tasks = [llumlet.is_ready.remote() for llumlet in self.instances.values()]
        is_ready_list = await asyncio.gather(*tasks)
        return all(is_ready_list)

    def _init_instance_info_csv(self, engine_manager_args: EngineManagerArgs) -> None:
        # pylint: disable=consider-using-with
        self.instance_info_file = open(engine_manager_args.log_filename + '_instance.csv', 'w', encoding='utf-8')
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
            'latency',
            'seq_lens',
            'num_instances',
            'num_seqs',
            'num_blocks_first_waiting_request',
            'num_blocks_all_waiting_requests',
            'waiting_time_first_waiting_request'])

    def _log_instance_infos_to_csv(self, instance_infos: List[InstanceInfo]) -> None:
        for instance_info in instance_infos:
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
                instance_info.latency,
                instance_info.running_seq_lens,
                self.num_instances,
                instance_info.num_seqs,
                instance_info.num_blocks_first_waiting_request,
                instance_info.num_blocks_all_waiting_requests,
                instance_info.waiting_time_first_waiting_request])
        self.instance_info_file.flush()
