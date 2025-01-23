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
import asyncio
from typing import List

from ray.util.placement_group import PlacementGroup
from vllm.engine.arg_utils import EngineArgs

from llumnix.logging.logger import init_logger
from llumnix.internal_config import MigrationConfig
from llumnix.backends.vllm.scheduler import SchedulerLlumnix
from llumnix.backends.vllm.llm_engine import LLMEngineLlumnix, BackendVLLM, EngineState
from llumnix.backends.profiling import ProfilingDatabase, LatencyMemData, ProfilingResult, SimParallelConfig
from llumnix.queue.queue_type import QueueType

logger = init_logger(__name__)


class BackendSimVLLM(BackendVLLM):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: EngineArgs,
        profiling_result_file_path: str
    ) -> None:
        # multi-instance args
        latency_mem = self._get_lantecy_mem(profiling_result_file_path, engine_args)
        self.engine: LLMEngineLlumnix = LLMEngineLlumnix.from_engine_args(instance_id,
                                                                          placement_group,
                                                                          request_output_queue_type,
                                                                          migration_config,
                                                                          engine_args,
                                                                          latency_mem)
        self.engine.scheduler = SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config)
        self.engine.scheduler.add_update_instance_info_callback(self.engine.update_instance_info)
        self.engine.output_processor.scheduler = self.engine.scheduler
        self.instance_id = instance_id

        self.state = EngineState.INIT
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))

        self._stop_event = asyncio.Event()
        asyncio.create_task(self._start_engine_step_loop())

    def _get_lantecy_mem(self, profiling_result_file_path: str, engine_args: EngineArgs) -> LatencyMemData:
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
        assert profiling_result is not None, f"can't find {model_name} in profiling database"
        sim_parallel_config = SimParallelConfig(parallel_config.tensor_parallel_size,
                                                parallel_config.pipeline_parallel_size)
        assert sim_parallel_config in profiling_result.para_dict.keys(), "sim parallel config not in database"
        latency_mem: LatencyMemData = profiling_result.para_dict[sim_parallel_config]
        return latency_mem

    # pylint: disable=unused-argument
    async def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]) -> None:
        await self.engine.model_executor.send_blocks(len(src_blocks))
