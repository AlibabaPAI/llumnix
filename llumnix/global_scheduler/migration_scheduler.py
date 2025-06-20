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
from functools import partial
from typing import Dict, Optional, Tuple

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.load_computation import DummyLoad
from llumnix.manager import Manager
from llumnix.utils import asyncio_wait_for_with_timeout
from llumnix.global_scheduler.migration_filter import MigrationInstanceFilter, MigrationFilterConfig, CustomFilter, MigrationFilterPolicy, LoadFilter
from llumnix.global_scheduler.migration_policy import PairMigrationPolicyFactory
from llumnix.instance_info import InstanceType

logger = init_logger(__name__)


class MigrationScheduler:
    def __init__(self,
                 manager: Manager,
                 pair_migration_policy: str,
                 migrate_out_load_threshold: float,
                 is_group_kind_migration_backend: bool,
                 enable_pd_disagg: bool,
                 enable_engine_pd_disagg: bool,
                 enable_adaptive_pd: bool) -> None:
        self.manager: Manager = manager
        self.enable_pd_disagg = enable_pd_disagg
        self.enable_engine_pd_disagg = enable_engine_pd_disagg
        self.enable_adaptive_pd = enable_adaptive_pd

        assert self.enable_adaptive_pd or (self.enable_pd_disagg or self.enable_engine_pd_disagg)
        self._register_migration_backend_init_filter(is_group_kind_migration_backend)
        self._register_new_instance_filter()

        filter_config = MigrationFilterConfig(migrate_out_load_threshold=migrate_out_load_threshold)
        self.migration_base_filter = MigrationInstanceFilter(filter_config)

        self.pair_migration_policy = PairMigrationPolicyFactory.get_policy(
            pair_migration_policy, migrate_out_load_threshold=migrate_out_load_threshold)

        if self.enable_pd_disagg or self.enable_engine_pd_disagg:
            self.p_d_instance_filter = CustomFilter()
            self.p_d_instance_filter.set_filter_condtition(
                src_filter=lambda instance_info: instance_info.inference_type == InstanceType.PREFILL,
                dst_filter=lambda instance_info: instance_info.inference_type == InstanceType.DECODE
            )
            self.p2d_migration_filter = MigrationInstanceFilter(filter_config)
            self.p2d_migration_filter.register_filter("p2d_migration", self.p_d_instance_filter)

            self.d_d_instance_filter = CustomFilter()
            self.d_d_instance_filter.set_filter_condtition(
                src_filter=lambda instance_info: instance_info.inference_type == InstanceType.DECODE,
                dst_filter=lambda instance_info: instance_info.inference_type == InstanceType.DECODE
            )
            self.d2d_migration_filter = MigrationInstanceFilter(filter_config)
            self.d2d_migration_filter.register_filter("dd_intance", self.d_d_instance_filter)
            self.d2d_migration_filter.register_filter("dd_load_balance", LoadFilter)

        if self.enable_adaptive_pd:
            self.dynamic_p_any_filter = CustomFilter()
            self.dynamic_p_any_filter.set_filter_condtition(
                src_filter=lambda instance_info: instance_info.inference_type == InstanceType.PREFILL \
                    and instance_info.max_decode_batch_size > 0,
                dst_filter=lambda instance_info: not (instance_info.inference_type == InstanceType.PREFILL \
                    and instance_info.max_decode_batch_size == 0),
            )
            self.dynamic_p_any_migration_filter = MigrationInstanceFilter(filter_config)


        # if pair_migration_type == PairMigrationConstraints.NO_CONSTRAINTS:
        #     policy_filter = MigrationFilterPolicyFactory.get_policy("load")
        # elif pair_migration_type in [PairMigrationConstraints.PREFILL_2_DECODE, PairMigrationConstraints.DECODE_2_DECODE]:
        #     policy_filter = MigrationFilterPolicyFactory.get_policy('pdd')
        # else:
        #     raise ValueError(f"Unsupported pair migration type: {pair_migration_type}")
        # src_filter_conditions.append(policy_filter.filter_src_condition(self.filter_config, pair_migration_type))
        # dst_filter_conditions.append(policy_filter.filter_dst_condition(self.filter_config, pair_migration_type))


    def _register_migration_backend_init_filter(self, is_group_kind_migration_backend: bool) -> None:
        # some migration backends require init_process_group before passing the KV cache. Here, we add a filter
        # to prevent instances of migration backends that have not been initialized from participating in migration.
        migration_backend_init_filter = CustomFilter()
        migration_backend_init_filter.set_filter_condtition(
            src_filter=lambda _: not is_group_kind_migration_backend,
            dst_filter=lambda _: not is_group_kind_migration_backend)
        self.migration_base_filter.register_filter("migration_backend_init_filter", migration_backend_init_filter)

    def _register_new_instance_filter(self) -> None:
        # instances that have just been launched should be refused for migration due to the absence of load information.
        new_instance_filter = CustomFilter()
        new_instance_filter.set_filter_condtition(
            src_filter=lambda instance_info: not isinstance(instance_info.migration_load_metric, DummyLoad),
            dst_filter=lambda instance_info: not isinstance(instance_info.migration_load_metric, DummyLoad))
        self.migration_base_filter.register_filter("new_instance_filter", new_instance_filter)

    async def push_migrations(self, instance_info: Dict[str, InstanceInfo]) -> None:
        pass
        # if self.enable_pd_disagg:
        #     asyncio.create_task(self._migrate(instance_info, PD_MIGRATION))
        #     asyncio.create_task(self._migrate(instance_info, D2D))
        # else:
        #     asyncio.create_task(self._migrate(instance_info, NO2No))

        # if self.enable_adaptive_pd:
        #     asyncio.create_task(self._migrate(instance_info, P' to D or P'))
        #     asyncio.create_task(self._migrate(instance_info, D or P' or P))

    # migration_filter must ensure that the specific instance_info does not appear in both src and dst simultaneously
    def _pair_migration(self, instance_info: Dict[str, InstanceInfo], migration_filter: Optional[MigrationInstanceFilter]):
        available_src_instance_infos, available_dst_instance_infos \
            = self.migration_base_filter.filter_instances(instance_info)

        src_instance_infos = available_src_instance_infos
        if migration_filter:
            src_instance_infos = migration_filter.filter_src_instances(available_src_instance_infos)

        dst_instance_infos = available_dst_instance_infos
        if migration_filter:
            dst_instance_infos = migration_filter.filter_dst_instances(available_dst_instance_infos)

        assert set(src_instance_infos).isdisjoint(dst_instance_infos), f"src and dst migration instance should not" \
            f"overlap, but got overlap {set(src_instance_infos).intersection(dst_instance_infos)}"

        return self.pair_migration_policy.pair_migration(src_instance_infos, dst_instance_infos)

    async def _migrate(self,
                       instance_infos: Dict[str, InstanceInfo],
                       filter: MigrationInstanceFilter):
        # TODO(s5u13b): Remove the migration done callback through decentralized migration refactoring.
        async def migrate_done_callback(ret, migrate_instance_pair: Tuple[str, str]) -> None:
            if migrate_instance_pair[0] in self.manager.instance_migrating:
                self.manager.instance_migrating[migrate_instance_pair[0]] = False
            if migrate_instance_pair[1] in self.manager.instance_migrating:
                self.manager.instance_migrating[migrate_instance_pair[1]] = False
            if isinstance(ret, Exception):
                has_error_pair = await self.manager._check_instance_error(migrate_instance_pair)
                for i, has_error in enumerate(has_error_pair):
                    if has_error:
                        instance_id = migrate_instance_pair[i]
                        await self.manager.scale_down(instance_id)
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
            migrate_instance_pairs = self._pair_migration(instance_infos, filter)
            migration_tasks = []
            for _, migrate_instance_pair in enumerate(migrate_instance_pairs):
                src_instance_id, dst_instance_id = migrate_instance_pair
                if self.manager.instance_migrating[src_instance_id] or \
                    self.manager.instance_migrating[dst_instance_id]:
                    continue
                self.manager.instance_migrating[src_instance_id] = True
                self.manager.instance_migrating[dst_instance_id] = True
                dst_instance_actor = self.manager.instances[dst_instance_id]
                task = asyncio.gather(
                    asyncio_wait_for_with_timeout(
                        self.manager.instances[src_instance_id].migrate_out.remote(
                            dst_instance_actor, dst_instance_id
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
