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

from llumnix.queue.ray_queue_server import RayQueueServer
from llumnix.queue.queue_type import QueueType

class RequestTimestamps:
    def __init__(self):
        self.api_server_manager_generate_timestamp = -1.0
        self.manager_generate_timestamp = -1.0
        self.llumlet_generate_timestamp = -1.0
        self.engine_add_request_timestamp = -1.0
        self.engine_process_model_outputs_timestamp_begin = -1.0
        self.engine_process_model_outputs_timestamp_end = -1.0
        self.engine_step_timestamp_begin = -1.0
        self.engine_step_timestamp_end = -1.0
        self.engine_step_postprocess_timestamp_end = -1.0
        self.engine_thread_put_queue_timestamp = -1.0
        self.engine_actor_put_queue_timestamp = -1.0
        self.queue_client_send_timestamp = -1.0
        self.queue_server_receive_timestamp = -1.0
        self.api_server_background_process_get_queue_timestamp = -1.0
        self.api_server_generate_benchmark_timestamp_end = -1.0

    @property
    def process_model_outputs_latency(self):
        return (self.engine_process_model_outputs_timestamp_end - self.engine_process_model_outputs_timestamp_begin)*1000

    @property
    def step_latency_engine(self):
        return (self.engine_step_timestamp_end - self.engine_step_timestamp_begin)*1000

    @property
    def step_postprocess_latency(self):
        return (self.engine_step_postprocess_timestamp_end - self.engine_step_timestamp_end)*1000

    @property
    def across_async_put_queue_thread_latency(self):
        return (self.engine_thread_put_queue_timestamp - self.engine_step_timestamp_end)*1000

    @property
    def across_async_put_queue_actor_latency(self):
        return (self.engine_actor_put_queue_timestamp - self.engine_thread_put_queue_timestamp)*1000

    @property
    def zmq_rpc_latency(self):
        return (self.queue_server_receive_timestamp - self.queue_client_send_timestamp)*1000

    @property
    def background_process_get_queue_latency(self):
        return (self.api_server_background_process_get_queue_timestamp - self.queue_server_receive_timestamp)*1000

    @property
    def generate_benchmark_return_output_latency(self):
        return (self.api_server_generate_benchmark_timestamp_end - self.api_server_background_process_get_queue_timestamp)*1000

class ServerInfo:
    def __init__(self,
                 server_id: str,
                 output_queue_type: QueueType,
                 request_output_queue: RayQueueServer,
                 request_output_queue_ip: str,
                 request_output_queue_port: int) -> None:
        self.server_id = server_id
        self.output_queue_type = output_queue_type
        if output_queue_type == QueueType.RAYQUEUE:
            assert request_output_queue is not None
        self.request_output_queue = request_output_queue.queue if output_queue_type == QueueType.RAYQUEUE else None
        self.request_output_queue_ip = request_output_queue_ip
        self.request_output_queue_port = request_output_queue_port
