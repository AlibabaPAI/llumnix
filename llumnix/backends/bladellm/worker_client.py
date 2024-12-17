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

from typing import Callable, Optional, List

from blade_llm.service.workers.worker_client import LocalWorkerClient
from blade_llm.service.proto.bladellm_pb2 import WorkerMetaRequest, WorkerStepRequest


class LocalWorkerClientLlumnix(LocalWorkerClient):
    def barrier(self, request_group_ids: List[int], callback: Optional[Callable] = None):
        request = WorkerMetaRequest(method="barrier", step=WorkerStepRequest(decode=request_group_ids)).SerializeToString()
        for i in range(len(self.reader)):
            self.rpc_call(request, i)
        tasks = [self.rpc_response(i) for i in range(len(self.reader))]
        self.futures.put_nowait((tasks, callback))
