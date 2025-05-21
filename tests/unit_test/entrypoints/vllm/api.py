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

from fastapi.responses import JSONResponse, Response
import ray

import llumnix.entrypoints.vllm.api_server

llumnix_client = llumnix.entrypoints.vllm.api_server.llumnix_client
manager = None
instance = None
app = llumnix.entrypoints.vllm.api_server.app


@app.get("/stats")
def stats() -> Response:
    """Get the statistics of the engine."""
    return JSONResponse(ray.get(instance.testing_stats.remote()))
