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

from dataclasses import dataclass, field
from typing import List, Union, Dict, Any, Iterable

def set_timestamp(obj: Any, timestamp_attr: str, timestamp: float):
    if not isinstance(obj, Iterable):
        obj = [obj,]
    objs = list(obj)
    for item in objs:
        if hasattr(item, "request_timestamps"):
            if hasattr(item.request_timestamps, timestamp_attr):
                setattr(item.request_timestamps, timestamp_attr, timestamp)


@dataclass
class RequestTimestamps:
    api_server_generate_timestamp: float = field(default=0.0)
    manager_generate_timestamp: float = field(default=0.0)
    llumlet_generate_timestamp: float = field(default=0.0)
    engine_add_request_timestamp: float = field(default=0.0)
    engine_process_model_outputs_timestamp_begin: float = field(default=0.0)
    engine_process_model_outputs_timestamp_end: float = field(default=0.0)
    engine_step_timestamp_begin: float = field(default=0.0)
    engine_step_timestamp_end: float = field(default=0.0)
    engine_step_postprocess_timestamp_end: float = field(default=0.0)
    engine_put_queue_timestamp: float = field(default=0.0)
    engine_thread_put_queue_timestamp: float = field(default=0.0)
    engine_actor_put_queue_timestamp: float = field(default=0.0)
    queue_client_send_timestamp: float = field(default=0.0)
    queue_server_receive_timestamp: float = field(default=0.0)
    api_server_get_queue_timestamp: float = field(default=0.0)
    api_server_generate_timestamp_end: float = field(default=0.0)

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
