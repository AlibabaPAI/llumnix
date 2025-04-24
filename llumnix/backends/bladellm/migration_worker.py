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

# pylint: disable=invalid-overridden-method

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List

import grpc
import torch
from google.protobuf import empty_pb2

from blade_llm.service.args import ServingArgs
from blade_llm.service.workers.local_worker import LocalWorker
from blade_llm.service.workers.base_worker import BaseWorker
from blade_llm.service.proto import bladellm_pb2
from blade_llm.module.parallel import nums_rank_per_node

from llumnix.backends.bladellm.migration_backend import get_migration_backend, WorkerRequestSyncGroup
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.internal_config import MigrationConfig
from llumnix.logging.logger import init_logger
from llumnix.constants import GRPC_MAX_MESSAGE_LENGTH
from llumnix.utils import get_ip_address, convert_bytes

logger = init_logger(__name__)


class MigrationWorker(migration_worker_pb2_grpc.MigrationWorkerServicer):
    def __init__(self, state_manager, instance_id: str, migration_config: MigrationConfig,
                 request_sync_group: WorkerRequestSyncGroup, base_worker: BaseWorker, rank: int,
                 args: ServingArgs) -> None:
        migration_worker_pb2_grpc.MigrationWorkerServicer.__init__(self)
        local_rank = rank % nums_rank_per_node()
        torch.cuda.set_device(f"cuda:{local_rank}")
        self.instance_id = instance_id
        self.migration_config = migration_config
        self.rank = rank
        self.state_manager = state_manager

        # ragged_flash_kv_arena
        num_layer = len(self.state_manager.kv_cache_arena.gpu_kv_cache)
        self.single_block_bytes = num_layer * self.state_manager.kv_cache_arena.gpu_kv_cache[0][0].nbytes

        # Assume request meta size in worker do not exceed 20MB
        request_meta_max_size = 1024 * 1024 * 20
        grpc_limit_migration_num_blocks = int((GRPC_MAX_MESSAGE_LENGTH - request_meta_max_size) / self.single_block_bytes)
        if migration_config.migration_buffer_blocks >= grpc_limit_migration_num_blocks:
            logger.warning("migration_buffer_blocks {} is too large, reset to grpc_limit_migration_num_blocks {}."
                           .format(migration_config.migration_buffer_blocks, grpc_limit_migration_num_blocks))
            migration_config.migration_buffer_blocks = grpc_limit_migration_num_blocks

        self.migration_backend = get_migration_backend(instance_id, rank, rank, migration_config,
                                                    request_sync_group, base_worker, state_manager, args)

        self.migration_grpc_addr = get_ip_address() + ":" + \
            str(self.migration_config.grpc_migration_backend_server_port+self.rank)

        asyncio.create_task(self._launch_grpc_service())

    async def _launch_grpc_service(self):
        options=[
            ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH),
        ]
        server = grpc.aio.server(migration_thread_pool=ThreadPoolExecutor(max_workers=2), options=options)
        migration_worker_pb2_grpc.add_MigrationWorkerServicer_to_server(self, server)
        server.add_insecure_port(self.migration_grpc_addr)
        await server.start()
        # TODO(KuilongCui): clear server state
        await server.wait_for_termination()

    # pylint: disable=unused-argument
    def is_ready(self, request, context):
        return empty_pb2.Empty()

    # pylint: disable=unused-argument
    def migrate_cache(self, request, context):
        try:
            start_time = time.time()
            self.migration_backend.migrate_cache(request, request.src_blocks, request.dst_blocks)
            end_time = time.time()
            total_kv_cache_size = len(request.src_blocks) * self.single_block_bytes
            # pylint: disable=invalid-name
            GB_bytes = 1024 * 1024 * 1024
            speed = total_kv_cache_size / GB_bytes / (end_time - start_time)
            logger.info("Migrate kv cache done, blocks_num: {}, total_kv_cache_size: {}, time: {:.2f}s, speed: {:.5f}GB/s."
                        .format(len(request.src_blocks), convert_bytes(total_kv_cache_size), end_time-start_time, speed))
        # pylint: disable=broad-except
        except Exception as e:
            logger.exception("Failed to migrate cache, request {}, error: {}".format(request, e))
        return empty_pb2.Empty()

    def do_send(self, request, context):
        return self.migration_backend.do_send(request, context)

    def rebuild_migration_backend(self, request, context) -> bool:
        self.migration_backend.destory_backend()
        return self.migration_backend.init_backend(None, None, None)

    def warmup(self, request, context):
        resp = migration_worker_pb2.WarmupResponse(is_ok=True)
        try:
            self.migration_backend.warmup()
        # pylint: disable=broad-except
        except Exception as e:
            resp.is_ok = False
            resp.error_msg = f"warmup failed: {e}"
        return resp


class MigrationLocalWorker(LocalWorker, MigrationWorker):
    def __init__(self, rank: int, serving_args: ServingArgs,
                 instance_id: str, migration_config: MigrationConfig,) -> None:
        try:
            LocalWorker.__init__(self, rank, serving_args)

            self.request_sync_group = WorkerRequestSyncGroup(self._engine._state_manager._request_groups,
                                                             self._req_tracker)
            MigrationWorker.__init__(self, self._engine._state_manager, instance_id, migration_config,
                                     self.request_sync_group, self, rank, serving_args)
        # pylint: disable=broad-except
        except Exception as e:
            logger.exception("Failed to initialize MigrationLocalWorker: {}".format(e))
            raise

    # used for wait_worker_ready
    async def info(self, req: empty_pb2.Empty) -> str:
        async with grpc.aio.insecure_channel(self.migration_grpc_addr) as channel:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)
            await stub.is_ready(empty_pb2.Empty())
        return await LocalWorker.info(self, req)

    async def barrier(self, request_group_ids: List[int]) -> str:
        self.request_sync_group.backup_request_group(request_group_ids)
        return bladellm_pb2.WorkerStepResponse(is_ok=True).SerializeToString()

    async def handle_rpc_call(self, request):
        method = request.method
        if method == "barrier":
            resp = await self.barrier(request.step.decode)
            self.send_response(resp)
        else:
            self.request_sync_group.update_migrated_in_request()
            rpc_responce = await super().handle_rpc_call(request)
            if method == "step":
                self.request_sync_group.remove_backup_request_group(
                    [finish_info.request_id for finish_info in request.step.latest_finished_ids])
            return rpc_responce
