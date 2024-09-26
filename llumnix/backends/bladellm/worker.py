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

from blade_llm.service.args import ServingArgs
from blade_llm.service.workers.remote_worker import RemoteWorker
from blade_llm.service.workers.local_worker import LocalWorker

from llumnix.internal_config import MigrationConfig
from llumnix.backends.bladellm.migration_worker import MigrationWorker

class MigrationLocalWorker(LocalWorker, MigrationWorker):
    def __init__(self, rank: int, serving_args: ServingArgs,
                 instance_id: int, migration_config: MigrationConfig,) -> None:
        LocalWorker.__init__(self, rank, serving_args)

        state_manager = self._engine._state_manager
        MigrationWorker.__init__(self, state_manager, instance_id, migration_config,
                                 rank, serving_args)

class MigrationRemoteWorker(RemoteWorker, MigrationWorker):
    def __init__(self, rank: int, serving_args: ServingArgs,
                 instance_id: int, migration_config: MigrationConfig,) -> None:
        RemoteWorker.__init__(self, rank, serving_args)

        state_manager = self._engine._state_manager
        MigrationWorker.__init__(self, state_manager, instance_id, migration_config,
                                 rank, serving_args)
