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

from blade_llm.service.communications.engine_client import MultiProcessingLLMClient
from blade_llm.service.communications.protocol import Stats
from blade_llm.service.communications.response import LLMResponse
from blade_llm.service.args import ServingArgs
from blade_llm.protocol import ServerRequest, GenerateStreamResponse

from llumnix.server_info import RequestTimestamps
from llumnix.entrypoints.utils import LlumnixEntrypointsContext
from llumnix.backends.bladellm.sequence import GenerateStreamResponseLlumnix
from llumnix.entrypoints.utils import (
    init_per_token_latency_breakdown_dict,
    record_per_token_latency_breakdown,
)
from llumnix.logger import init_logger

logger = init_logger(__name__)

class MeasureEntrypoint:
    def __init__(self, request_id, start_time, expected_resp_len):
        self.request_id = request_id
        self.final_output = None
        self.per_token_latency = []
        self.generation_text = []
        self.per_token_latency_breakdown_dict = init_per_token_latency_breakdown_dict()
        self.start_time = start_time
        self.expected_resp_len = expected_resp_len
        self.final_output = None
    
    @property
    def generation(self) -> bool:
        return f"{''.join(self.generation_text)}"

def measure_single_resp(resp: GenerateStreamResponse, measure: MeasureEntrypoint):
        now = time.time()
        measure.per_token_latency.append([now, (now - measure.start_time)*1000])
        measure.start_time = now
        measure.generation_text.extend([t.text for t in resp.tokens])
        measure.final_output = resp
        if hasattr(resp, 'request_timestamps'):
            resp.request_timestamps.api_server_generate_benchmark_timestamp_end = now
            record_per_token_latency_breakdown(measure.per_token_latency_breakdown_dict, resp.request_timestamps)

def measure_resp(measure_handle: MeasureEntrypoint):
    final_output = measure_handle.final_output
    assert final_output is not None
    from llumnix.entrypoints.bladellm.api_server import llumnix_context
    if llumnix_context.log_requests:
        llumnix_context.num_finished_requests += 1
        logger.info("Finished request {}.".format(measure_handle.request_id))
        logger.info("num_finished_requests {}.".format(llumnix_context.num_finished_requests))

    num_output_tokens = len(final_output.usage.completion_tokens)
    num_input_tokens = len(final_output.usage.prompt_tokens)
    if not max(measure_handle.expected_resp_len, 1) == max(num_output_tokens, 1):
        "request_id={}, expected_resp_len={}, num_output_tokens={}, num_input_tokens={}".format(
            measure_handle.request_id, measure_handle.expected_resp_len, num_output_tokens, num_input_tokens)
    ret = {
        'request_id': measure_handle.request_id,
        'generated_text': measure_handle.generation,
        'num_output_tokens_cf': num_output_tokens,
        'per_token_latency': measure_handle.per_token_latency,
        'per_token_latency_breakdown_dict': measure_handle.per_token_latency_breakdown_dict
    }

