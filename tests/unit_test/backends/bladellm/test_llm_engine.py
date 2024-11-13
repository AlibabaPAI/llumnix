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

import math
import torch
import asyncio
import queue
from unittest.mock import MagicMock
from typing import Dict, Set, Tuple, Optional

import pytest
import ray


from llumnix.backends.bladellm.llm_engine import LLMEngineLlumnix
# from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor, SimGPUExecutor
from llumnix.backends.profiling import LatencyMemData
from llumnix.backends.bladellm.sequence import GenerationGroupStateLlumnix
from llumnix.queue.queue_type import QueueType
from llumnix.server_info import ServerInfo
from llumnix.backends.bladellm.sequence import ServerRequestLlumnix
from llumnix.backends.bladellm.worker import _WorkerProcessesLlumnix, _RemoteWorkerProcessesLlumnix
from llumnix.backends.bladellm.worker_client import LlumnixPipelineWorkerClient, LlumnixLocalWorkerClient
from llumnix.internal_config import MigrationConfig
from llumnix.arg_utils import EngineManagerArgs
from llumnix.backends.bladellm.scheduler import PagedSchedulerLlumnix


from blade_llm.model.config_utils import load_config
from blade_llm.service.proto import bladellm_pb2 as pb
from blade_llm.service.scheduler_types import SchedulerAsyncUpdateOutput
from blade_llm.protocol import GenerateStreamResponse
from blade_llm.service.args import ServingArgs
from blade_llm.utils.load_model_options import LoadModelOptions
from blade_llm.protocol import (
    GenerateStreamResponse,
    SamplingParams,
    ServerRequest,
    StoppingCriteria,
)
from blade_llm.service.clients import GeneralLLMClient
from blade_llm.service.engine import AsyncLLMEngineClient

from .utils import create_dummy_request, initialize_scheduler
# from tests.unit_test.backends.bladellm.utils import initialize_scheduler, create_dummy_request

class MockEngine(LLMEngineLlumnix):
    def __init__(self, args: Optional[ServingArgs] = None):
        self._args = args
        self._scheduler = initialize_scheduler()
        self.step_request_queue = asyncio.Queue()
        self._req_buffer = asyncio.Queue()
        self._dropped_req: Set[int] = set()
        self._back_queue: Dict[int, asyncio.Queue] = {}
        self.put_queue_args_queue = queue.Queue()
        self.instance_info = None

    def set_client(self):
        client = AsyncLLMEngineClient(False, self._req_buffer, self._dropped_req,  self._back_queue, self._scheduler,)
        self._model_conf = load_config(self._args.load_model_options.model)
        self._client = GeneralLLMClient(self._args, client, self._model_conf)

@pytest.mark.asyncio
async def test_llm_engine_process_model_outputs():
    # process model outputs
    llm_engine = MockEngine()
    prompt_length = 7
    server_request_0 = create_dummy_request(0, prompt="0" * prompt_length)
    server_request_1 = create_dummy_request(1, prompt="1" * prompt_length)

    llm_engine._scheduler.add_request(server_request_0)
    llm_engine._scheduler.add_request(server_request_1)

    _ = llm_engine._scheduler.step()

    update_output = SchedulerAsyncUpdateOutput(
                    response={
                        0: GenerateStreamResponse(is_finished=False),
                        1: GenerateStreamResponse(is_finished=False),
                    },
                )
    # normal case, all requests be processed
    llm_engine.process_model_outputs(update_output)
    request_outputs, _ = llm_engine.put_queue_args_queue.get_nowait()
    assert len(request_outputs) == 2
    _ = llm_engine._scheduler.step()
    llm_engine._scheduler.remove_running_request(0)
    # migration case , requests stopping during last stage migration, stop process
    llm_engine.process_model_outputs(update_output)
    request_outputs, _ = llm_engine.put_queue_args_queue.get_nowait()
    assert len(request_outputs) == 1

def test_llm_engine_init():
    engine_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/dataset/opt-125m'))
    migration_config = EngineManagerArgs().create_migration_config()
    llm_engine = LLMEngineLlumnix("0", QueueType.RAYQUEUE, migration_config, None, ray.get_runtime_context().get_node_id(),
                                    engine_args)
    assert isinstance(llm_engine._worker_processes, _WorkerProcessesLlumnix)
    loop = asyncio.new_event_loop()
    llm_engine.start(loop)
    assert isinstance(llm_engine._workers, LlumnixLocalWorkerClient)
    assert isinstance(llm_engine._scheduler, PagedSchedulerLlumnix)
    llm_engine.stop()
    loop.close()

    engine_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/dataset/opt-125m'), enable_remote_worker=True, device=0, server_ip="127.0.0.1",
            rank=0)
    migration_config = EngineManagerArgs().create_migration_config()
    llm_engine = LLMEngineLlumnix("0", QueueType.RAYQUEUE, migration_config, None, ray.get_runtime_context().get_node_id(),
                                    engine_args)
    assert isinstance(llm_engine._worker_processes, _RemoteWorkerProcessesLlumnix)
    # need to start worker
    # llm_engine.start(asyncio.new_event_loop())
    # assert isinstance(llm_engine._workers, LlumnixPipelineWorkerClient)
    # assert isinstance(llm_engine._scheduler, PagedSchedulerLlumnix)
    # llm_engine.stop()

@pytest.mark.asyncio
async def test_llm_engine_add_requset():
    engine_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/dataset/opt-125m'))
    llm_engine = MockEngine(engine_args)
    llm_engine.set_client()
    llm_engine._scheduler.add_update_instance_info_callback(llm_engine.update_instance_info)
    server_info = ServerInfo(None, None, None, None, None)
    engine_request = ServerRequest(
                        id=11,
                        prompt="hello",
                        sampling_params=SamplingParams(top_p=0.9),
                        stopping_criterial=StoppingCriteria(max_new_tokens=10),
                    )
    server_request = ServerRequestLlumnix(engine_request, "0", server_info, math.inf)
    await llm_engine.add_request(server_request)
    llm_engine._handle_new_requests()
    assert len(llm_engine._scheduler.waiting) == 1
    assert llm_engine._scheduler.waiting[-1].request_id == "0"
    assert llm_engine._scheduler.waiting[-1].expected_steps == math.inf
    assert isinstance(llm_engine._scheduler.waiting[-1], GenerationGroupStateLlumnix)
    # llm_engine.stop()

@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for unit test")
@pytest.mark.asyncio
async def test_llm_engine_step():
    engine_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/dataset/Qwen--Qwen1.5-7B-Chat'), tensor_parallel_size=4)
    llm_engine = LLMEngineLlumnix("0", QueueType.RAYQUEUE, None, None, ray.get_runtime_context().get_node_id(),
                                    engine_args)
    await llm_engine._init()
    llm_engine._scheduler.add_update_instance_info_callback(llm_engine.update_instance_info)
    server_info = ServerInfo(None, None, None, None, None)
    engine_request = ServerRequest(
                        id=11,
                        prompt="hello",
                        prompt_tokens=[1] * 5,
                        sampling_params=SamplingParams(top_p=0.9),
                        stopping_criterial=StoppingCriteria(max_new_tokens=10),
                    )
    server_request = ServerRequestLlumnix(engine_request, "0", server_info, math.inf)
    llm_engine._back_queue[server_request.id] = asyncio.Queue()
    llm_engine._req_buffer.put_nowait(server_request)
    await llm_engine.step()
    assert list(llm_engine._back_queue.keys()) == [11]
    assert [request_group.request_group_id for request_group in llm_engine._scheduler.running] == [11]
    llm_engine.stop()