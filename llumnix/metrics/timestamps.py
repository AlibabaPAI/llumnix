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
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List
from llumnix.constants import REQUEST_TIMESTAMPS_ATTR_STR
from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


def set_timestamp(obj: Any, timestamp_attr: str, timestamp: float):
    if not isinstance(obj, Iterable):
        _set_timestamp_single_obj(obj, timestamp_attr, timestamp)
        return
    objs = list(obj)
    for item in objs:
        _set_timestamp_single_obj(item, timestamp_attr, timestamp)

def _set_timestamp_single_obj(item: Any, timestamp_attr: str, timestamp: float):
    if hasattr(item, REQUEST_TIMESTAMPS_ATTR_STR):
        if hasattr(item.request_timestamps, timestamp_attr):
            setattr(item.request_timestamps, timestamp_attr, timestamp)
    elif hasattr(item, timestamp_attr):
        setattr(item, timestamp_attr, timestamp)

def get_timestamp(item: Any, timestamp_attr: str):
    if hasattr(item, "request_timestamps") and hasattr(
        item.request_timestamps, timestamp_attr
    ):
        return getattr(item.request_timestamps, timestamp_attr)
    if hasattr(item, timestamp_attr):
        return getattr(item, timestamp_attr)
    return False

@dataclass
class DecodeTimestamps:
    
    

@dataclass
class RequestTimestamps:
    api_server_generate_timestamp: float = 0.0
    manager_generate_timestamp: float = 0.0
    llumlet_generate_timestamp: float = 0.0
    engine_add_request_timestamp: float = 0.0
    engine_process_model_outputs_timestamp_begin: float = 0.0
    engine_process_model_outputs_timestamp_end: float = 0.0
    engine_step_timestamp_begin: float = 0.0
    engine_step_timestamp_end: float = 0.0
    engine_step_postprocess_timestamp_end: float = 0.0
    engine_put_queue_timestamp: float = 0.0
    engine_thread_put_queue_timestamp: float = 0.0
    engine_actor_put_queue_timestamp: float = 0.0
    queue_client_send_timestamp: float = 0.0
    queue_server_receive_timestamp: float = 0.0
    api_server_get_queue_timestamp: float = 0.0
    api_server_generate_timestamp_end: float = 0.0
    
    decode_engine_step_timestamp_begin: List[float] = []
    decode_engine_step_timestamp_end: List[float] = []
    decode_engine_process_model_outputs_timestamp_begin: List[float] = []
    decode_engine_process_model_outputs_timestamp_end: List[float] = []
    decode_engine_put_queue_timestamp: List[float] = []
    decode_engine_thread_put_queue_timestamp: List[float] = []
    decode_engine_actor_put_queue_timestamp: List[float] = []
    decode_queue_client_send_timestamp: List[float] = []
    decode_queue_server_receive_timestamp: List[float] = []
    
    def to_latency_breakdown_dict(self) -> Dict[str, float]:
        latency_dict = {
            "across_manager_latency": (self.manager_generate_timestamp - self.api_server_generate_timestamp) * 1000,
            "across_llumlet_latency": (self.llumlet_generate_timestamp - self.manager_generate_timestamp) * 1000,
            "across_engine_latency": (self.engine_add_request_timestamp - self.llumlet_generate_timestamp) * 1000,
            "process_model_outputs_latency":
                (self.engine_process_model_outputs_timestamp_end - self.engine_process_model_outputs_timestamp_begin) * 1000,
            "engine_step_latency":
                (self.engine_step_timestamp_end - self.engine_step_timestamp_begin) * 1000,
            "step_postprocess_latency":
                (self.engine_step_postprocess_timestamp_end - self.engine_step_timestamp_end) * 1000,
            "across_async_put_queue_thread_latency":
                (self.engine_thread_put_queue_timestamp - self.engine_put_queue_timestamp) * 1000,
            "across_async_put_queue_actor_latency":
                (self.engine_actor_put_queue_timestamp - self.engine_thread_put_queue_timestamp) * 1000,
            "across_queue_client_latency":
                (self.queue_client_send_timestamp - self.engine_actor_put_queue_timestamp) * 1000,
            "queue_rpc_latency":
                (self.queue_server_receive_timestamp - self.queue_client_send_timestamp) * 1000,
            "api_server_get_queue_latency":
                (self.api_server_get_queue_timestamp - self.queue_server_receive_timestamp) * 1000,
            "across_request_streams_latency":
                (self.api_server_generate_timestamp_end - self.api_server_get_queue_timestamp) * 1000,
        }
        return latency_dict

    def set_timestamp(self, timestamp_attr: str, timestamp: float = None):
        if not hasattr(self, timestamp_attr):
            logger.warning("Timestamp attribute {} not found in RequestTimestamps.".format(timestamp_attr))
        if timestamp is None:
            timestamp = time.perf_counter()
        setattr(self, timestamp_attr, timestamp)
        
    def set_decode_timestamp(self, timestamp_attr: str, timestamp: float = None):
        if not hasattr(self, timestamp_attr):
            logger.warning("Timestamp attribute {} not found in RequestTimestamps.".format(timestamp_attr))
        if timestamp is None:
            timestamp = time.perf_counter()
        getattr(self, timestamp_attr).append(timestamp)
