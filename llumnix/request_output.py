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

from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.utils import RequestIDType


class LlumnixRequestOuput:
    def __init__(self, request_id: RequestIDType, instance_id: str,
                 engine_output: Any, request_timestamps: RequestTimestamps = None):
        self.request_id = request_id
        self.instance_id = instance_id
        self.engine_output = engine_output
        self.request_timestamps = request_timestamps

    def get_engine_output(self):
        return self.engine_output
    
class LlumnixRequestOutputs:
    """Wrapper of vLLM v1 EngineCoreOutputs"""
    def __init__(self, instance_id: str, engine_outputs: Any, 
                 current_completion_tokens_dict: Dict[RequestIDType, int],
                 request_timestamps_dict: Dict[RequestIDType, RequestTimestamps] = None):
        self.instance_id = instance_id
        self.engine_outputs = engine_outputs
        self.current_completion_tokens_dict = current_completion_tokens_dict
        self.request_timestamps_dict = request_timestamps_dict