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

import asyncio
import math
from multiprocessing import shared_memory
import ray
import pytest
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import json
from blade_llm.service.args import ServingArgs
from blade_llm.utils.load_model_options import LoadModelOptions
from blade_llm.protocol import (
    GenerateStreamResponse,
    SamplingParams,
    ServerRequest,
    StoppingCriteria,
)
from blade_llm.protocol import GenerateStreamResponse
from blade_llm.service.scheduler_types import (
    SchedulerAsyncUpdateOutput,
    SchedulerStepOutput,
)

from llumnix.llumlet.llumlet import Llumlet
from llumnix.backends.bladellm.sequence import ServerRequestLlumnix, GenerateStreamResponseLlumnix
from llumnix.server_info import ServerInfo
from llumnix.internal_config import MigrationConfig
from llumnix.arg_utils import EngineManagerArgs
from llumnix.queue.queue_type import QueueType
from llumnix.backends.bladellm.llm_engine import LLMEngineLlumnix
from llumnix.queue.ray_queue_server import RayQueueServer
from llumnix.backends.backend_interface import BackendInterface, BackendType

from tests.conftest import setup_ray_env

@pytest.mark.asyncio
async def test_llumlet(setup_ray_env):
    migration_config: MigrationConfig = EngineManagerArgs(migration_backend='grpc').create_migration_config()
    engine_class = ray.remote(num_cpus=1, num_gpus=1, name="blade_llumlet", namespace='llumnix', max_concurrency=4)(Llumlet).\
        options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False))
    engine_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/self-hosted/model/Qwen-7B'))

    llumlet0 = engine_class.remote("0", QueueType.RAYQUEUE, BackendType.BLADELLM, migration_config, engine_args, 
                                   None, ray.get_runtime_context().get_node_id())
    await llumlet0.is_ready.remote()

    request_output_queue = RayQueueServer()
    server_info = ServerInfo("my_server", QueueType.RAYQUEUE, request_output_queue, None, None)
    engine_request = ServerRequest(
        id=11,
        prompt="hello",
        prompt_tokens=[10001,10002,10003],
        sampling_params=SamplingParams(top_p=0.9),
        stopping_criterial=StoppingCriteria(max_new_tokens=2),
    )
    await llumlet0.generate.remote("0", server_info, math.inf, engine_request.model_dump_json())
    
    finish = False
    while not finish:
        request_outputs = await request_output_queue.get()
        for request_output in request_outputs:
            print(" -------- from_test -----  ", type(request_output), json.loads(request_output))
            if json.loads(request_output)['is_finished']:
                finish = True

    await llumlet0.execute_engine_method.remote("stop")