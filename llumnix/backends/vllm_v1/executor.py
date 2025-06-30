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

import time

from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.v1.outputs import ModelRunnerOutput

from llumnix.internal_config import MigrationConfig
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


class LlumnixRayDistributedExecutor(RayDistributedExecutor):
    instance_id: str = None
    migration_config: MigrationConfig = None
    last_inference_latency: int = 0
    
    async def execute_model_async(
        self,
        scheduler_output,
    ) -> "ModelRunnerOutput":
        """Executes the model asynchronously on the Ray workers.

        Args:
            scheduler_output: The scheduler output to execute.

        Returns:
            The model runner output.
        """
        t0 = time.time()
        # Build the compiled DAG for the first time.
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=True)

        refs = await self.forward_dag.execute_async(scheduler_output)  # type: ignore

        t1 = time.time()
        self.last_inference_latency = (t1 - t0) * 1000
        # The returned refs is a list of futures.
        return await refs[0]