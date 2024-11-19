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
from concurrent.futures import ThreadPoolExecutor
from google.protobuf import empty_pb2
from multiprocessing import Process, set_start_method
from typing import List

from blade_llm.generation.kvcache.kv_transfer import TransferType
from blade_llm.service.workers.remote_worker import RemoteWorker, RemoteManager
from blade_llm.service.args import ServingArgs
from blade_llm.service.worker import _RemoteWorkerProcesses, _WorkerProcesses
from blade_llm.utils.network import get_free_port
from blade_llm.service.worker import worker_main
from blade_llm.service.workers.local_worker import LocalWorker
from blade_llm.service.workers.remote_worker import RemoteWorker

from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.internal_config import MigrationConfig
from llumnix.backends.bladellm.migration_worker import MigrationWorker
from llumnix.logger import init_logger

logger = init_logger(__name__)

class _WorkerProcessesLlumnix(_WorkerProcesses):
    def __init__(self, args: ServingArgs, instance_id: int, migration_config: MigrationConfig):
        super().__init__(args)
        self.instance_id = instance_id
        self.migration_config = migration_config

    def _spawn_workers(self):
        """Spawn backend servers in background processes for inference"""
        set_start_method("spawn", force=True)
        backends: List[Process] = []
        for i in range(self._args.tensor_parallel_size * self._args.pipeline_parallel_size):
            p = Process(target=worker_main, args=(i, self._args, self.instance_id, self.migration_config, 'shm:migrate_cache_test'))
            p.start()
            backends.append(p)
        return backends

# not ready
class _RemoteWorkerProcessesLlumnix(_RemoteWorkerProcesses):
    def __init__(self, args: ServingArgs, instance_id: int, migration_config: MigrationConfig):
        super().__init__(args)
        self.instance_id = instance_id
        self.migration_config = migration_config

def launch_worker(args: ServingArgs, instance_id: int, migration_config: MigrationConfig):
    if args.enable_remote_worker:
        # TODO(xinyi): not support RemoteWorkerProcesses
        return _RemoteWorkerProcessesLlumnix(args, instance_id, migration_config)
    else:
        return _WorkerProcessesLlumnix(args, instance_id, migration_config)

class MigrationLocalWorker(LocalWorker, MigrationWorker):
    def __init__(self, instance_id: int, worker_addr: str, migration_config: MigrationConfig,
                naming_url: str,  tranfer_type: TransferType, rank: int, args: ServingArgs) -> None:
        LocalWorker.__init__(self, rank, args)

        state_manager = self._engine._state_manager
        MigrationWorker.__init__(self, state_manager, instance_id, worker_addr, migration_config,
                                 naming_url, tranfer_type, rank, args)
        
class MigrationRemoteWorker(RemoteWorker, MigrationWorker):
    def __init__(self, instance_id: int, worker_addr: str, migration_config: MigrationConfig,
                naming_url: str,  tranfer_type: TransferType, rank: int, args: ServingArgs) -> None:
        RemoteWorker.__init__(self, rank, args)

        state_manager = self._engine._state_manager
        MigrationWorker.__init__(self, state_manager, instance_id, worker_addr, migration_config,
                                 naming_url, tranfer_type, rank, args)
