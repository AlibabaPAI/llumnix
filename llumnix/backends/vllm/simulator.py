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

import os

from typing import List

from vllm.utils import Counter
from vllm.engine.arg_utils import EngineArgs

from llumnix.logger import init_logger
from llumnix.config import MigrationConfig
from llumnix.backends.vllm.scheduler import SchedulerLlumnix
from llumnix.backends.vllm.llm_engine import LLMEngineLlumnix, BackendVLLM
from llumnix.backends.profiling import ProfilingDatabase, LatencyMemData, ProfilingResult, SimParallelConfig


logger = init_logger(__name__)

class BackendSimVLLM(BackendVLLM):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        instance_id: int,
        migration_config: MigrationConfig,
        profiling_result_file_path: str,
        gpu_type: str,
        engine_args: EngineArgs,
    ) -> None:
        # load database
        profiling_database = ProfilingDatabase(profiling_result_file_path)
        engine_config = engine_args.create_engine_config()
        model_config = engine_config.model_config
        parallel_config = engine_config.parallel_config
        model_name = model_config.model
        # get model_name from model path
        if model_name.endswith('/'):
            model_name = model_name[:-1]
        model_name = os.path.basename(model_name)
        # get latency mem
        profiling_result: ProfilingResult = profiling_database.get(model_name)
        sim_parallel_config = SimParallelConfig(gpu_type, parallel_config.tensor_parallel_size,
                                                parallel_config.pipeline_parallel_size)
        assert sim_parallel_config in profiling_result.para_dict.keys(), "sim parallel config not in database"
        latency_mem: LatencyMemData = profiling_result.para_dict[sim_parallel_config]

        self.engine: LLMEngineLlumnix = LLMEngineLlumnix.from_engine_args(migration_config=migration_config,
                                                                          latency_mem=latency_mem, engine_args=engine_args)
        self.engine.scheduler = SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config)
        self.engine.output_processor.scheduler = self.engine.scheduler
        # multi-instance args
        self.migration_config = migration_config
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.scaling_down = False
        self.request_server_info = {}

    def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]) -> None:
        self.engine.model_executor.send_blocks(len(src_blocks))
