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

from typing import Optional, Tuple

from blade_llm.model.config_base import ConfigBase
from blade_llm.service.args import ServingArgs
from blade_llm.service.clients import GeneralLLMClient
from blade_llm.service.schedulers import (
    ContinuousBatchingScheduler,
    DynamicBatchingScheduler,
    _SCHEDULER_MAP
)
from blade_llm.service.clients import BaseLLMClient, LLMResponse
from blade_llm.protocol import ServerRequest

from utils import manager_generate, manager_abort
from api_server import llumnix_context

class DummyAsyncLLMEngineClient():
    async def add_request(self, request: ServerRequest) -> LLMResponse:
        resp_stream = await manager_generate(request, request.id, llumnix_context)
        return resp_stream
    
    async def add_request(self, request: ServerRequest) -> LLMResponse:
        resp_stream = await manager_generate(request, request.id, llumnix_context)
        return resp_stream
    
    async def drop_request(self, request_id: int):
        await manager_abort(request_id, llumnix_context)

class GeneralLLMClientLlumnix(GeneralLLMClient):
    def __init__(self, args: ServingArgs, client: BaseLLMClient, model_conf: Optional[ConfigBase] = None):
        super().__init__(args, client, model_conf)
        self.scheduler = _SCHEDULER_MAP[args.decode_algo if args.use_lookahead else args.load_model_options.attn_cls]

    def support_beam_search(self):
        if self.args.pipeline_parallel_size > 1 or not self.scheduler == ContinuousBatchingScheduler:
            return (
                False,
                "beam_search can only used with continuous_batching scheduler and pipeline disabled.",
            )
        else:
            return True, ""
    
    def support_chat_stream(self) -> Tuple[bool, str]:
        if self.scheduler == DynamicBatchingScheduler:
            return False, "DynamicBatchingScheduler not support chat_stream"
        else:
            return True, ""
