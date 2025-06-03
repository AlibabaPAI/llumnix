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


from llumnix.metrics.base_metrics import BaseMetrics
from llumnix.metrics.metrics_types import Registery, Summary
from llumnix.logging.logger import init_logger
from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.metrics.utils import enable_any_metrics

logger = init_logger(__name__)


class LlumnixClientMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.register = Registery()

        self.across_manager_latency = Summary(
            "across_manager_latency", registry=self.register
        )
        self.across_llumlet_latency = Summary(
            "across_llumlet_latency", registry=self.register
        )
        self.across_engine_latency = Summary(
            "across_engine_latency", registry=self.register
        )
        self.process_model_outputs_latency = Summary(
            "process_model_outputs_latency", registry=self.register
        )
        self.engine_step_latency = Summary(
            "engine_step_latency", registry=self.register
        )
        self.step_postprocess_latency = Summary(
            "step_postprocess_latency", registry=self.register
        )
        self.across_async_put_queue_thread_latency = Summary(
            "across_async_put_queue_thread_latency", registry=self.register
        )
        self.across_async_put_queue_actor_latency = Summary(
            "across_async_put_queue_actor_latency", registry=self.register
        )
        self.across_queue_client_latency = Summary(
            "across_queue_client_latency", registry=self.register
        )
        self.queue_rpc_latency = Summary("queue_rpc_latency", registry=self.register)
        self.api_server_get_queue_latency = Summary(
            "api_server_get_queue_latency", registry=self.register
        )
        self.across_request_streams_latency = Summary(
            "across_request_streams_latency", registry=self.register
        )

        self.enable_metrics = enable_any_metrics()
        if self.enable_metrics:
            self.start_metrics_export_loop()

    def record_latency_if_exist(
        self,
        timestamp_begin: float,
        timestamp_end: float,
        metrics_name: str,
        labels: dict = None,
    ) -> bool:
        if timestamp_begin > 0.0 and timestamp_end > 0.0:
            self.register.get(metrics_name).observe(
                value=(timestamp_end - timestamp_begin) * 1000,
                labels=labels,
            )

    async def record_request_timestamps(
        self,
        request_timestamp: RequestTimestamps,
        server_id: str,
        instance_id: str = "unknown",
    ):
        # some manager metrics or engine metrics may passed by timestamps,
        # so we record them here
        if enable_any_metrics() and request_timestamp:
            self.record_latency_if_exist(
                timestamp_end=request_timestamp.manager_generate_timestamp,
                timestamp_begin=request_timestamp.api_server_generate_timestamp,
                metrics_name="across_manager_latency",
                labels={"server_id": server_id},
            )
            self.record_latency_if_exist(
                timestamp_end=request_timestamp.llumlet_generate_timestamp,
                timestamp_begin=request_timestamp.manager_generate_timestamp,
                metrics_name="across_llumlet_latency",
                labels={"instance_id": instance_id},
            )
            self.record_latency_if_exist(
                timestamp_end=request_timestamp.engine_add_request_timestamp,
                timestamp_begin=request_timestamp.llumlet_generate_timestamp,
                metrics_name="across_engine_latency",
                labels={"instance_id": instance_id},
            )

            self.record_latency_if_exist(
                timestamp_end=request_timestamp.engine_process_model_outputs_timestamp_end,
                timestamp_begin=request_timestamp.engine_process_model_outputs_timestamp_begin,
                metrics_name="process_model_outputs_latency",
                labels={"instance_id": instance_id},
            )

            self.record_latency_if_exist(
                timestamp_end=request_timestamp.engine_step_timestamp_end,
                timestamp_begin=request_timestamp.engine_step_timestamp_begin,
                metrics_name="engine_step_latency",
                labels={"instance_id": instance_id},
            )

            self.record_latency_if_exist(
                timestamp_end=request_timestamp.engine_step_postprocess_timestamp_end,
                timestamp_begin=request_timestamp.engine_step_timestamp_end,
                metrics_name="step_postprocess_latency",
                labels={"instance_id": instance_id},
            )
            self.record_latency_if_exist(
                timestamp_end=request_timestamp.engine_thread_put_queue_timestamp,
                timestamp_begin=request_timestamp.engine_put_queue_timestamp,
                metrics_name="across_async_put_queue_thread_latency",
                labels={"instance_id": instance_id},
            )
            self.record_latency_if_exist(
                timestamp_end=request_timestamp.engine_actor_put_queue_timestamp,
                timestamp_begin=request_timestamp.engine_thread_put_queue_timestamp,
                metrics_name="across_async_put_queue_actor_latency",
                labels={"instance_id": instance_id},
            )
            self.record_latency_if_exist(
                timestamp_end=request_timestamp.queue_client_send_timestamp,
                timestamp_begin=request_timestamp.engine_actor_put_queue_timestamp,
                metrics_name="across_queue_client_latency",
                labels={"instance_id": instance_id},
            )

            self.record_latency_if_exist(
                timestamp_end=request_timestamp.queue_server_receive_timestamp,
                timestamp_begin=request_timestamp.queue_client_send_timestamp,
                metrics_name="queue_rpc_latency",
                labels={"instance_id": instance_id},
            )
            self.record_latency_if_exist(
                timestamp_end=request_timestamp.api_server_get_queue_timestamp,
                timestamp_begin=request_timestamp.queue_server_receive_timestamp,
                metrics_name="api_server_get_queue_latency",
                labels={"instance_id": instance_id},
            )
            self.record_latency_if_exist(
                timestamp_end=request_timestamp.api_server_generate_timestamp_end,
                timestamp_begin=request_timestamp.api_server_get_queue_timestamp,
                metrics_name="across_request_streams_latency",
                labels={"instance_id": instance_id},
            )
