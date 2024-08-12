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

from typing import Optional
from llumnix.llumlet.migrating_request import MigratingRequest
from llumnix.backends.backend_interface import BackendInterface

class LocalMigrationScheduler:
    def __init__(self, request_migration_policy: str, backend_engine: BackendInterface) -> None:
        self.request_migration_policy = request_migration_policy
        self.backend_engine = backend_engine

    def get_migrate_out_request(self) -> Optional[MigratingRequest]:
        # TODO(s5u13b): remove the if-else codes
        migrate_out_request: MigratingRequest = None
        if self.request_migration_policy == 'LCFS':
            migrate_out_request = self.backend_engine.get_last_running_request()
        elif self.request_migration_policy in ['SJF', 'LJF']:
            if self.request_migration_policy == 'LJF':
                migrate_out_request = self.backend_engine.get_longest_running_request()
            elif self.request_migration_policy == 'SJF':
                migrate_out_request = self.backend_engine.get_shortest_running_request()
        return migrate_out_request
