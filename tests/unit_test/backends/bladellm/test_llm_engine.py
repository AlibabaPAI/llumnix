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
import asyncio
from unittest.mock import MagicMock
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



from blade_llm.service.proto import bladellm_pb2 as pb
from blade_llm.service.scheduler_types import SchedulerAsyncUpdateOutput
from blade_llm.protocol import GenerateStreamResponse
from blade_llm.service.args import ServingArgs
from blade_llm.utils.load_model_options import LoadModelOptions
from blade_llm.protocol import ServerRequest



from .utils import create_dummy_request, initialize_scheduler
# from tests.unit_test.backends.bladellm.utils import initialize_scheduler, create_dummy_request



class MockEngine(LLMEngineLlumnix):
    def __init__(self, *args, executor_class=None, **kwargs):
        self._scheduler = initialize_scheduler()
        self.step_request_queue = asyncio.Queue()
#         detokenizer = MagicMock(spec=Detokenizer)
#         stop_checker = MagicMock(spec=StopChecker)
#         self.seq_counter = Counter()
#         self.instance_info = None
#         self.executor_class = executor_class
#         self.scheduler.add_update_instance_info_callback(self.update_instance_info)
#         self.output_processor = SingleStepOutputProcessor(self.scheduler.scheduler_config,detokenizer, self.scheduler, self.seq_counter, stop_checker)

#     def update_instance_info(self, instance_info):
#         pass

# @pytest.mark.asyncio

# async def test_llm_engine_process_model_outputs():
#     # process model outputs
#     llm_engine = MockEngine()
#     prompt_length = 7
#     server_request_0 = create_dummy_request(0, prompt="0" * prompt_length)
#     server_request_1 = create_dummy_request(1, prompt="1" * prompt_length)

#     llm_engine._scheduler.add_request(server_request_0)
#     llm_engine._scheduler.add_request(server_request_1)

#     _ = llm_engine._scheduler.step()

#     update_output = SchedulerAsyncUpdateOutput(
#                     response={
#                         0: GenerateStreamResponse(is_finished=False),
#                         1: GenerateStreamResponse(is_finished=False),
#                     },
#                 )
#     # normal case, all requests be processed
#     llm_engine.process_model_outputs(update_output)
#     request_outputs, _ = llm_engine.step_request_queue.get_nowait()
#     assert len(request_outputs) == 2
#     _ = llm_engine._scheduler.step()
#     llm_engine._scheduler.remove_running_request(0)
#     # migration case , requests stopping during last stage migration, stop process
#     llm_engine.process_model_outputs(update_output)
#     request_outputs, _ = llm_engine.step_request_queue.get_nowait()
#     assert len(request_outputs) == 1

# def test_llm_engine_init():
#     engine_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/dataset/opt-125m'))
#     migration_config = EngineManagerArgs().create_migration_config()
#     llm_engine = LLMEngineLlumnix("0", QueueType.RAYQUEUE, migration_config, None, ray.get_runtime_context().get_node_id(),
#                                     engine_args)
#     assert isinstance(llm_engine._worker_processes, _WorkerProcessesLlumnix)
#     loop = asyncio.new_event_loop()
#     llm_engine.start(loop)
#     assert isinstance(llm_engine._workers, LlumnixLocalWorkerClient)
#     assert isinstance(llm_engine._scheduler, PagedSchedulerLlumnix)
#     llm_engine.stop()
#     loop.close()

#     engine_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/dataset/opt-125m'), enable_remote_worker=True, device=0, server_ip="127.0.0.1",
#             rank=0)
#     migration_config = EngineManagerArgs().create_migration_config()
#     llm_engine = LLMEngineLlumnix("0", QueueType.RAYQUEUE, migration_config, None, ray.get_runtime_context().get_node_id(),
#                                     engine_args)
#     assert isinstance(llm_engine._worker_processes, _RemoteWorkerProcessesLlumnix)
#     # need to start worker
#     # llm_engine.start(asyncio.new_event_loop())
#     # assert isinstance(llm_engine._workers, LlumnixPipelineWorkerClient)
#     # assert isinstance(llm_engine._scheduler, PagedSchedulerLlumnix)
#     # llm_engine.stop()

@pytest.mark.asyncio
async def test_llm_engine_add_requset():
    engine_args = ServingArgs(load_model_options=LoadModelOptions(model='/mnt/dataset/opt-125m'))
    llm_engine = LLMEngineLlumnix("0", QueueType.RAYQUEUE, None, None, ray.get_runtime_context().get_node_id(),
                                    engine_args)
    await llm_engine._init()
    server_info = ServerInfo(None, None, None, None, None)
    engine_request = ServerRequest(id=11, prompt="prompt")
    server_request = ServerRequestLlumnix.from_server_request(engine_request, "0", server_info, math.inf)
    await llm_engine.add_request(server_request)
    await llm_engine.step()
    assert len(llm_engine._scheduler.waiting) == 1
    assert llm_engine._scheduler.waiting[-1].request_id == "0"
    assert llm_engine._scheduler.waiting[-1].expected_steps == math.inf
    assert isinstance(llm_engine._scheduler.waiting[-1], GenerationGroupStateLlumnix)
    llm_engine.stop()