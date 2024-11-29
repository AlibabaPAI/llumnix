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

from typing import Dict
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor
from google.protobuf import empty_pb2
import torch
import grpc

from blade_llm.service.args import ServingArgs
from blade_llm.utils.network import get_free_port

from llumnix.backends.bladellm.migration_backend import get_migration_backend
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.internal_config import MigrationConfig
from llumnix.logger import init_logger

logger = init_logger(__name__)

class MigrationWorker(migration_worker_pb2_grpc.MigrationWorkerServicer):
    def __init__(self, state_manager, instance_id: int, migration_config: MigrationConfig,
                 rank: int, args: ServingArgs) -> None:
        migration_worker_pb2_grpc.MigrationWorkerServicer.__init__(self)
        device = args.device if args.device else torch.cuda.device(rank)
        torch.cuda.set_device(device)
        self.instance_id = instance_id
        self.migration_config = migration_config
        self.migration_backend = get_migration_backend(instance_id, rank, rank, migration_config,
                                                       state_manager, args)

    # pylint: disable=unused-argument
    def migrate_cache(self, request: migration_worker_pb2.MigrateRequests, context) -> None:
        try:
            src_worker_handle = request.src_handlers[self._rank]
            self.migration_backend.migrate_cache(src_worker_handle, request.src_blocks, request.dst_blocks)
        # pylint: disable=broad-except
        except Exception as e:
            logger.info("[migrate_cache] rank: {}, {} is dead, err : {}.".format(self._rank, src_worker_handle, e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))

        return empty_pb2.Empty()

    def do_send(self, request: migration_worker_pb2.SendKvCacheRequest, context):
        return self.migration_backend.do_send(request, context)

    def rebuild_migration_backend(self, instance_rank: Dict[str, int], group_name: str) -> bool:
        self.migration_backend.destory_backend()
        return self.migration_backend.init_backend(None, None, None)

    def warmup(self, request, context):
        resp = migration_worker_pb2.WarmupResponse(is_ok=True)
        try:
            self.migration_backend.warmup()
        except Exception as e:
            resp.is_ok = False
            resp.error_msg = f"warmup failed: {e}"
        return resp
    
    def migrate_request_group(self, request: migration_worker_pb2.MigrateResGroupRequests, context):
        try:
            src_worker_handle = request.src_handlers[self._rank]
            self.migration_backend.migrate_request_group(src_worker_handle, request.id)
        # pylint: disable=broad-except
        except Exception as e:
            logger.info("[migrate_request_group] rank: {}, {} is dead, err : {}.".format(self._rank, src_worker_handle, e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
        
        return empty_pb2.Empty()
    
    def send_request_group(self, request: migration_worker_pb2.SendResourceGroupRequests, context):
        return self.migration_backend.send_request_group(request, context) 

    def shutdown(self) -> None:
        torch.cuda.synchronize()
        del self.migration_backend
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
