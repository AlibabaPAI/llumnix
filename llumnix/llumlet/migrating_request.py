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

from typing import Any

class MigratingRequest:
    def __init__(
            self,
            request_id: int,
            backend_request: Any,
        ) -> None:
        self.request_id = request_id
        self.backend_request = backend_request
        self.stage_timestamps = []
        self.stage_num_blocks_list = []
        self.server_info = None
