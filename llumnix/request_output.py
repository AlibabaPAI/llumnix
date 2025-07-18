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

from typing import Any, Dict
import copy

from llumnix.request_processing_context import RequestProcessingContext
from llumnix.utils import RequestIDType


class LlumnixRequestOuput:
    def __init__(self, request_id: RequestIDType, instance_id: str,
                 engine_output: Any, request_processing_context: RequestProcessingContext = None):
        self.request_id = request_id
        self.instance_id = instance_id
        self.engine_output = engine_output
        self.request_processing_context: RequestProcessingContext = request_processing_context
        if request_processing_context is not None and request_processing_context.request_output_queue is not None:
            self.request_processing_context = copy.copy(request_processing_context)
            # should set queue to None, otherwise it will OOM when use rayqueue
            self.request_processing_context.request_output_queue = None

    def get_engine_output(self):
        return self.engine_output

class LlumnixRequestOutputs:
    """Wrapper of vLLM v1 EngineCoreOutputs"""
    def __init__(self, instance_id: str, engine_outputs: Any,
                 request_processing_context_dict: Dict[RequestIDType, RequestProcessingContext] = None):
        self.instance_id = instance_id
        self.engine_outputs = engine_outputs
        self.request_processing_context_dict: Dict[RequestIDType, RequestProcessingContext] = request_processing_context_dict
        for request_id, context in self.request_processing_context_dict.items():
            if context.request_output_queue is not None:
                new_context = copy.copy(context)
                new_context.request_output_queue = None
                self.request_processing_context_dict[request_id] = new_context
