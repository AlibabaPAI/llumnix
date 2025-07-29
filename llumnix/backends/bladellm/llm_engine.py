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

# pylint: disable=protected-access

import sys
import time
from functools import partial
from typing import List, Optional, Tuple, Union, Iterable, Dict, Any, Coroutine
from collections import defaultdict
import asyncio
import queue
import os

import msgspec
import ray
import ray.actor
import grpc
from ray.util.placement_group import PlacementGroup
from loguru import logger as loguru_logger
from google.protobuf import empty_pb2

from blade_llm.utils.constants import LOGGER_FORMAT
from blade_llm.service.engine import AsyncLLMEngine
from blade_llm.service.args import ServingArgs
from blade_llm.protocol_msgspec import ServerRequest, GenerateStreamResponse
from blade_llm.utils.disagg_utils import InstanceRole
from blade_llm.service.disagg_pd_engine import PrefillAsyncLLMEngine, DecodeAsyncLLMEngine
from blade_llm.service.communications.engine_msg_server import EngineMsgServer
from blade_llm.service.engine_args import CommunicationArgs
from blade_llm.service.metric import init_metric, InMemExporter
from blade_llm.module.parallel import setup_dist_environ, master_node_in_distributed_inference
from blade_llm.utils.constants import NCCL_PORT
from blade_llm.module.parallel import is_distributed_inference
from blade_llm.service.communications import AsyncLLMEngineClient
from blade_llm.service.communications.protocol_msgspec import Stats
from blade_llm.utils.hardware_util import get_cpu_number

from llumnix.arg_utils import InstanceArgs
from llumnix.utils import (
    get_ip_address,
    asyncio_wait_for_ray_remote_call_with_timeout,
    get_free_port,
    wait_port_free,
    run_coroutine_in_new_thread,
    MigrationResponse,
)
from llumnix.internal_config import MigrationConfig
from llumnix.request_processing_context import RequestProcessingContext
from llumnix.backends.utils import EngineState
from llumnix.backends.output_forwarder import OutputForwarder, RequestOutputForwardingMode
from llumnix.llumlet.request import LlumnixRequest, RequestStatus, RequestInferenceType
from llumnix.instance_info import InstanceInfo
from llumnix.queue.queue_type import QueueType
from llumnix.logging.logger import init_logger
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc
from llumnix.backends.bladellm.proto.migration_worker_pb2 import RecvCacheRequest, WorkerInfo
from llumnix.backends.bladellm.sequence import GenerationGroupStateLlumnix
from llumnix.backends.bladellm.worker import WorkerProcessesRay
from llumnix.constants import RAY_RPC_TIMEOUT
from llumnix.utils import BackendType, InstanceContext
from llumnix.request_output import LlumnixRequestOuput
from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.arg_utils import LlumnixEngineArgs
from llumnix.utils import RequestIDType, exception_wrapper_async
from llumnix.backends.backend_interface import BackendInterface

logger = init_logger(__name__)


class RequestBarrier:
    def __init__(self, request: GenerationGroupStateLlumnix):
        self.request = request
        self.wait_event = asyncio.Event()

    def notify(self):
        self.wait_event.set()

    async def wait(self):
        await self.wait_event.wait()


class AsyncBackQueueWrapper:
    def __init__(self,
                 placement_group: PlacementGroup,
                 instance_id: str,
                 request_output_queue_type: QueueType,
                 resp_queue: asyncio.Queue,
                 request_metrics_queue_dict: Dict[str, asyncio.Queue],
                 metrics_queue: asyncio.Queue,
                 backend_type: BackendType,
                 request_output_forwarding_mode: RequestOutputForwardingMode,
                 drop_request_callback: Coroutine) -> None:
        self.instance_id = instance_id
        self.msg_encoder = msgspec.msgpack.Encoder()
        self.output_forwarder = OutputForwarder(
            instance_id,
            request_output_queue_type,
            request_output_forwarding_mode,
            drop_request_callback,
            placement_group,
            backend_type,
        )
        self.resp_queue: asyncio.Queue = resp_queue
        self.metrics_queue: asyncio.Queue = metrics_queue
        self.request_metrics_queue_dict: Dict[str, asyncio.Queue] = request_metrics_queue_dict

        self.request_processing_context_map = {}
        # asyncio.queue is not thread-safe, just create a asyncio.task
        asyncio.create_task(self._put_request_outputs_loop())

        self.dangling_request_server_info: Dict[int, int] = {} # req_id, expired_step
        self.backup_dangling_request_server_info: Dict[int, int] = {}
        self.get_current_step_counter_queue: asyncio.Queue = asyncio.Queue()
        asyncio.create_task(self._clear_request_server_info_loop())

    # Due to Bladellm returning tokens to Llumnix asynchronously, Llumnix cannot directly delete the
    # server-related information for a request in the request_processing_context_map during handle_abort, handle_drop,
    # or migration free_src_request. Instead, the related information can only be safely removed after
    # the corresponding step's output token has been actually processed in the AsyncBackQueueWrapper.
    async def _clear_request_server_info_loop(self):
        while True:
            cur_step_idx: int = await self.get_current_step_counter_queue.get()
            self.backup_dangling_request_server_info.update(self.dangling_request_server_info)
            self.dangling_request_server_info = {}

            expired_req_ids = []
            for req_id, expired_step_idx in self.backup_dangling_request_server_info.items():
                if cur_step_idx >= expired_step_idx:
                    expired_req_ids.append(req_id)
                    self.request_processing_context_map.pop(req_id, None)
                    self.request_metrics_queue_dict.pop(req_id, None)

            for req_id in expired_req_ids:
                self.backup_dangling_request_server_info.pop(req_id, None)

    def _set_step_metrics(self):
        try:
            self.current_step_metrics = self.metrics_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def _put_request_outputs_loop(self):
        async def get_single_response() -> Tuple[GenerateStreamResponse, RequestProcessingContext]:
            resp: Union[GenerateStreamResponse, int] = await self.resp_queue.get()
            while isinstance(resp, int):
                self.get_current_step_counter_queue.put_nowait(resp)
                self._set_step_metrics()
                resp: Union[GenerateStreamResponse, int] = await self.resp_queue.get()
            request_processing_context: RequestProcessingContext = self.request_processing_context_map[resp.req_id]
            if resp.is_finished:
                logger.info("Engine finished request {}.".format(resp.req_id))
                self.request_processing_context_map.pop(resp.req_id, None)
            return resp, request_processing_context

        def get_nowait_responses(output_size: int) -> Tuple[GenerateStreamResponse, RequestProcessingContext]:
            resps, request_processing_contexts = [], []

            while output_size > 0:
                resp: Union[GenerateStreamResponse, int] = self.resp_queue.get_nowait()
                output_size -= 1

                while isinstance(resp, int) and output_size > 0:
                    self.get_current_step_counter_queue.put_nowait(resp)
                    self._set_step_metrics()
                    resp: Union[GenerateStreamResponse, int] = self.resp_queue.get_nowait()
                    output_size -= 1

                if isinstance(resp, int):
                    continue

                request_processing_context: RequestProcessingContext = self.request_processing_context_map[resp.req_id]
                if resp.is_finished:
                    logger.info("Engine finished request {}.".format(resp.req_id))
                    self.request_processing_context_map.pop(resp.req_id, None)

                resps.append(resp)
                request_processing_contexts.append(request_processing_context)

            return resps, request_processing_contexts

        self.current_step_metrics: RequestTimestamps = None
        while True:
            request_outputs, request_processing_context_outputs = [], []
            resp, request_processing_context = await get_single_response()
            request_outputs.append(resp)
            request_processing_context_outputs.append(request_processing_context)

            output_size = self.resp_queue.qsize()
            if output_size > 0:
                resps, request_processing_contexts = get_nowait_responses(output_size)
                request_outputs.extend(resps)
                request_processing_context_outputs.extend(request_processing_contexts)

            server_request_outputs, server_info_dict = self._gen_server_request_outputs(request_outputs, request_processing_context_outputs)
            if server_request_outputs:
                await self.output_forwarder.put_request_outputs_to_server(server_request_outputs, server_info_dict)

    def _gen_server_request_outputs(self,
                                    request_outputs: List[GenerateStreamResponse],
                                    request_processing_contexts: List[RequestProcessingContext]) -> None:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}
        # Reorganize data in order to put request output to queue in batch at one time.
        for request_output, request_processing_context in zip(request_outputs, request_processing_contexts):
            req_id = request_output.req_id
            if req_id in self.request_metrics_queue_dict:
                request_processing_context.add_trace_timeline("engine_process_model_outputs_timestamp_end")
                engine_put_queue_timestamp = time.perf_counter()
                request_processing_context.add_trace_timeline("engine_put_queue_timestamp",engine_put_queue_timestamp)
                request_processing_context.add_trace_timeline("engine_thread_put_queue_timestamp",engine_put_queue_timestamp)
                current_step_metrics = self.request_metrics_queue_dict[req_id].get_nowait()
                request_processing_context.add_trace_timeline("engine_step_timestamp_end",current_step_metrics.engine_step_timestamp_end)
                request_processing_context.add_trace_timeline(
                    "engine_step_postprocess_timestamp_end",
                    current_step_metrics.engine_step_postprocess_timestamp_end,
                )
                request_processing_context.add_trace_timeline(
                    "engine_process_model_outputs_timestamp_begin",
                    current_step_metrics.engine_process_model_outputs_timestamp_begin,
                )

            llumnix_request_output = LlumnixRequestOuput(
                req_id, self.instance_id, self.msg_encoder.encode(request_output), request_processing_context
            )
            server_id = request_processing_context.server_id
            server_request_outputs[server_id].append(llumnix_request_output)
            if server_id not in server_info_dict:
                server_info_dict[server_id] = request_processing_context.get_server_info()

        return server_request_outputs, server_info_dict

    def stop(self):
        self.output_forwarder.stop()

    def remove_request_server_info(self, request_id: int, expired_step: int) -> None:
        self.dangling_request_server_info[request_id] = expired_step
        logger.debug("Trans_wrapper {} is going to remove request {} at step {}.".format(
            self.instance_id, request_id, expired_step))

    def add_request(self, request_id: int, request_processing_context: RequestProcessingContext) -> bool:
        if request_id in self.request_processing_context_map:
            logger.error("Request {} already exists in Trans_wrapper.".format(request_id))
            return False
        self.request_processing_context_map[request_id] = request_processing_context
        logger.debug("Trans_wrapper {} add_request {} from server {}.".format(
            self.instance_id, request_id, request_processing_context.server_id))
        return True

    def clear(self):
        self.request_processing_context_map = {}
        self.request_metrics_queue_dict = {}
        logger.info("Trans_wrapper reset.")


def setup_dist_options(serving_args: ServingArgs):
    # It means that dist_init_addr can be None when enabling distributed inference.
    if serving_args.dist_inference_options.dist_init_addr is not None:
        master_port = int(serving_args.dist_inference_options.dist_init_addr.split(":")[1])
    else:
        master_port = NCCL_PORT
    # The IP of engine and worker 0 will be same due to our sorting of workers,
    # so directly set the dist_init_addr to IP of engine is correct.
    serving_args.dist_inference_options.dist_init_addr = f"{serving_args.host}:{master_port}"
    # TODO(s5u13b): New BladeLLM will not use this environment variables, update it after rebase BladeLLM.
    os.environ["MASTER_ADDR"] = serving_args.host

def setup_dist(serving_args: ServingArgs):
    if is_distributed_inference():
        setup_dist_options(serving_args)
    setup_dist_environ(serving_args.dist_inference_options.nnodes, serving_args.dist_inference_options.node_rank)


class AsyncLLMEngineLlumnixMixin:
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 request_barriers: queue.Queue,
                 backend_type: BackendType,
                 request_output_forwarding_mode: RequestOutputForwardingMode,
                 ) -> None:
        self.instance_id = instance_id
        self.state = EngineState.INIT
        self.migration_config = migration_config
        logger.info("Engine {} current state: {}.".format(self.instance_id, self.state))

        self.placement_group = placement_group
        self.request_output_queue_type = request_output_queue_type
        self.request_output_forwarding_mode = request_output_forwarding_mode
        self._worker_processes = WorkerProcessesRay(placement_group, self._args, instance_id, migration_config)

        self.src_workers_migration_ip_addr_list: List[str] = None
        self._migration_semaphore = asyncio.Semaphore(0)
        self.request_barriers: queue.Queue = request_barriers
        self.migrated_request = set()
        self.resp_queue = asyncio.Queue()
        self.metrics_queue = asyncio.Queue()
        self.request_metrics_queue_dict: Dict[str, asyncio.Queue] = {}

        self.backend_type = backend_type
        self.step_counter: int = 0

    @property
    def instance_info(self) -> InstanceInfo:
        return self._scheduler.llumnix_metrics.to_instance_info()

    async def step(self):
        self.step_counter += 1
        self.handle_request_barriers()
        await super().step()

    async def async_start(self, loop: asyncio.AbstractEventLoop):
        await super().async_start(loop)
        self._client: AsyncLLMEngineClient = self.init_client_from_engine()
        self.trans_wrapper = AsyncBackQueueWrapper(
            self.placement_group,
            self.instance_id,
            self.request_output_queue_type,
            self.resp_queue,
            self.request_metrics_queue_dict,
            self.metrics_queue,
            self.backend_type,
            self.request_output_forwarding_mode,
            self.drop_request,
        )
        self._scheduler.trans_wrapper = self.trans_wrapper
        self._scheduler.llumnix_metrics.engine_init_metrics(self)

        if self.migration_config.enable_migration:
            migration_worker_port = self._worker_processes.get_all_workers_grpc_migration_server_port()
            ip = get_ip_address()
            self.src_workers_migration_ip_addr_list = [ip + ":" + str(port) for port in migration_worker_port]
            logger.info("Engine {} set grpc migration server address for all workers: {}.".format(
                        self.instance_id, self.src_workers_migration_ip_addr_list))
            self.worker_migration_channels = [grpc.aio.insecure_channel(worker) for worker in self.src_workers_migration_ip_addr_list]
            self.worker_migration_stubs = [migration_worker_pb2_grpc.MigrationWorkerStub(channel) for channel in self.worker_migration_channels]

    def handle_request_barriers(self):
        # pylint: disable=unused-argument
        async def finish_callback(resp_list, request_barriers: List[RequestBarrier]):
            for request_barrier in request_barriers:
                request_barrier.notify()

        barrier_size = self.request_barriers.qsize()
        if barrier_size > 0:
            all_request_barriers = []
            running_request_barriers = []
            running_filter_request_ids = set()

            for _ in range(barrier_size):
                request_barrier: RequestBarrier = self.request_barriers.get()
                all_request_barriers.append(request_barrier)
                if not request_barrier.request.should_abort_migration():
                    running_request_barriers.append(request_barrier)
                    running_filter_request_ids.add(request_barrier.request.request_id)
            self._scheduler.running_filter_request_ids.update(running_filter_request_ids)

            if len(running_request_barriers) > 0:
                self._workers.barrier(
                    request_group_ids=list(running_filter_request_ids),
                    callback=partial(finish_callback, request_barriers=running_request_barriers))
            else:
                asyncio.create_task(finish_callback(None, request_barriers=running_request_barriers))

    # TODO(KuilongCui): As barrier is always used when a request is determined to be migrated, a request
    # can be identified as decode at the time it is scheduled by the scheduler, without having to wait until
    # the update_callback.
    def _update_request_inference_type(self, resp_list):
        request_groups = resp_list[0].generation_groups.generation_group
        for gen_group in request_groups:
            request_group_id = gen_group.request_group_id
            if request_group_id in self.scheduler.id2group:
                num_out_token = gen_group.generations[0].num_out_token
                self._scheduler.id2group[request_group_id]._inference_type = \
                    RequestInferenceType.DECODE if num_out_token > 0 else RequestInferenceType.PREFILL

    async def update_callback(self, resp_list, *args, **kwargs):
        for resp in resp_list:
            request_groups = resp.generation_groups.generation_group
            step_timestamps = RequestTimestamps()
            for gen_group in request_groups:
                request_group_id = gen_group.request_group_id
                if request_group_id in self.request_metrics_queue_dict and request_group_id in self._back_queue:
                    worker_metrics = resp_list[0].worker_step_metrics
                    worker_forward_time = (
                        worker_metrics.prepare_step_ms
                        + worker_metrics.model_forward_ms
                        + worker_metrics.post_step_ms
                    )
                    step_timestamps.set_timestamp('engine_step_timestamp_end', worker_forward_time / 1000)
                    step_timestamps.set_timestamp('engine_step_postprocess_timestamp_end', worker_forward_time / 1000)
                    step_timestamps.set_timestamp('engine_process_model_outputs_timestamp_begin', time.perf_counter())
                    self.request_metrics_queue_dict[request_group_id].put_nowait(step_timestamps)

        self.resp_queue.put_nowait(self.step_counter)
        await super().update_callback(resp_list, *args, **kwargs)
        self._update_request_inference_type(resp_list)
        self.scheduler.llumnix_metrics.engine_step_metrics(self.scheduler)

    async def _loop(self):
        previous_state = self.state
        self.state = EngineState.RUNNING
        logger.info("engine {} change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        try:
            await super()._loop()
        # pylint: disable=broad-except
        except Exception:
            logger.exception("Error in engine _loop")
            self.stop()
            previous_state = self.state
            self.state = EngineState.CRASHED
            logger.info("Engine {} change state: {} -> {}.".format(self.instance_id, previous_state, self.state))

        if self.state == EngineState.RUNNING:
            self.stop()
            self.state = EngineState.STOPPED
            logger.info("Engine {} change state: {} -> {}.".format(self.instance_id, EngineState.RUNNING, self.state))

    def stop(self, *args, **kwargs):
        if self.migration_config.enable_migration:
            run_coroutine_in_new_thread(self.close_migration(), blocking=True)
        self.trans_wrapper.stop()
        super().stop(*args, **kwargs)

    # Close migraion grpc client and delete server.
    async def close_migration(self):
        # delete grpc server
        try:
            await self.run_workers_async("close_migration", empty_pb2.Empty())
        # pylint: disable=broad-except
        except Exception:
            logger.exception("Error in engine close migration for workers")
        # close grpc client
        tasks = [channel.close() for channel in self.worker_migration_channels]
        rets = await asyncio.gather(*tasks, return_exceptions=True)
        for rank, ret in enumerate(rets):
            if isinstance(ret, Exception):
                logger.exception("Error in engine close migration channel for worker (rank: {})".format(rank))

    async def _handle_abort(self, abort: Optional[List[Tuple[int, int, str]]] = None):
        await super()._handle_abort(abort)
        if abort is not None and len(abort) > 0:
            for req_id, _, _ in abort:
                self._scheduler.id2group[req_id].set_llumnix_status(RequestStatus.FINISHED)
                self.trans_wrapper.remove_request_server_info(
                    req_id, self.step_counter + 1
                )

    async def _handle_reset(self):
        await super()._handle_reset()
        self.trans_wrapper.clear()

    async def add_request_wrapper(self, request_processing_context: RequestProcessingContext, server_request: ServerRequest):
        logger.debug("Engine {} add request {}:{}.".format(self.instance_id, server_request.id, server_request.external_id))
        if request_processing_context.enable_trace:
            request_processing_context.add_trace_timeline('engine_add_request_timestamp')
            self.request_metrics_queue_dict[server_request.id] = asyncio.Queue()
        if self.trans_wrapper.add_request(server_request.id, request_processing_context):
            # pylint: disable=protected-access
            await self._client._add_request(server_request, self.resp_queue)

    async def drop_request(self, req_id: int):
        logger.debug("Engine {} drop request {}.".format(self.instance_id, req_id))
        await self._client.drop_request(req_id)

    # Only be used in migration method.
    async def run_workers_async(self, worker_method: str, *args, **kwargs) -> List[Any]:
        coros = []
        for stub in self.worker_migration_stubs:
            method = getattr(stub, worker_method)
            coros.append(method(*args, **kwargs))
        result = await asyncio.gather(*coros)
        return result

    async def wake_engine(self):
        completed, pending = await asyncio.wait(
            [
                self._req_buffer.get(),
                self._stop_event.wait(),
                self._migration_semaphore.acquire(),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        return completed, pending

    # metrics
    def get_num_trans_wrapper_cached_request(self):
        return len(self.trans_wrapper.request_processing_context_map)

    def get_num_wait_update_request_ids(self) -> int:
        return self.request_barriers.qsize()

    async def get_metrics(self) -> str:
        return InMemExporter.METRICS_TXT


class AsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, AsyncLLMEngine):
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 request_barriers: queue.Queue,
                 backend_type: BackendType,
                 request_output_forwarding_mode: RequestOutputForwardingMode,
                 serving_args: ServingArgs,
                 *args, **kwargs,
                 ) -> None:
        AsyncLLMEngine.__init__(self, serving_args, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(
            self,
            instance_id,
            placement_group,
            request_output_queue_type,
            migration_config,
            request_barriers,
            backend_type,
            request_output_forwarding_mode,
        )


class PrefillAsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, PrefillAsyncLLMEngine):
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 request_barriers: queue.Queue,
                 backend_type: BackendType,
                 request_output_forwarding_mode: RequestOutputForwardingMode,
                 serving_args: ServingArgs,
                 *args, **kwargs,
                ) -> None:
        # Prefill Instances typically do not need CUDA Graph to be enabled
        serving_args.load_model_options.disable_cuda_graph = True

        PrefillAsyncLLMEngine.__init__(self, serving_args, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(
            self,
            instance_id,
            placement_group,
            request_output_queue_type,
            migration_config,
            request_barriers,
            backend_type,
            request_output_forwarding_mode,
        )


class DecodeAsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, DecodeAsyncLLMEngine):
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 request_barriers: queue.Queue,
                 backend_type: BackendType,
                 request_output_forwarding_mode: RequestOutputForwardingMode,
                 serving_args: ServingArgs,
                 *args, **kwargs,
                ) -> None:
        DecodeAsyncLLMEngine.__init__(self, serving_args, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(
            self,
            instance_id,
            placement_group,
            request_output_queue_type,
            migration_config,
            request_barriers,
            backend_type,
            request_output_forwarding_mode,
        )


class BackendBladeLLM(BackendInterface):
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 instance_args: InstanceArgs,
                 llumnix_engine_args: LlumnixEngineArgs
                ) -> None:
        self.instance_id = instance_id
        self.placement_group = placement_group
        self.request_output_queue_type = request_output_queue_type
        self.instance_args = instance_args
        self.migration_config: MigrationConfig = instance_args.create_migration_config()
        self.llumnix_engine_args = llumnix_engine_args
        self.src_workers_migration_ip_addr_list = []
        self.request_barriers: queue.Queue = queue.Queue()
        self._engine_ready_event = asyncio.Event()

        self._load_and_reconfig_engine_args()
        engine_cls = self._get_engine_cls()
        self.engine: AsyncLLMEngineLlumnixMixin = engine_cls(
            self.instance_id,
            self.placement_group,
            self.request_output_queue_type,
            self.migration_config,
            self.request_barriers,
            BackendType.BLADELLM,
            instance_args.request_output_forwarding_mode,
            self.engine_args,
        )

        asyncio.create_task(self._start_engine())

    def _load_and_reconfig_engine_args(self):
        self.engine_disagg_inst_id: str = (
            os.environ.get(self.instance_args.engine_disagg_inst_id_env_var)
            if self.instance_args.engine_disagg_inst_id_env_var
            else self.instance_id
        )
        if self.instance_args.enable_engine_pd_disagg:
            self.llumnix_engine_args.update_arg("engine_disagg_inst_id", self.engine_disagg_inst_id)
        elif self.instance_args.enable_engine_semi_pd_disagg:
            self.llumnix_engine_args.update_arg("semi_pd_inst_id", self.engine_disagg_inst_id)

        self.engine_args: ServingArgs = self.llumnix_engine_args.load_engine_args()
        self.engine_args.decoding_parallelism = min(max(get_cpu_number() // 2, 1), 2)

        # add instance_id to avoid path conflict when multi-engine running in a single pod
        # use instance_id[:5] to avoid the length of worker_socket_path exceeding the OS limit
        # Note that there is still a small probability that worker_socket_path will be repeated
        self.engine_args.worker_socket_path = self.engine_args.worker_socket_path + "_" + \
            str(self.instance_id)[:5]

        init_metric(
            self.engine_args.serving_metric_options.metric_export_interval_sec,
            *self.engine_args.metric_exporters,
            observability_options=self.engine_args.serving_observability_options,
        )
        if self.engine_args.host not in ("127.0.0.1", "0.0.0.0"):
            self.engine_args.host = get_ip_address()
        self._config_inner_engine_logger(self.engine_args)

        if master_node_in_distributed_inference():
            wait_port_free(self.engine_args.multi_node_hb_port())

        setup_dist(self.engine_args)

        if self.engine_args.enable_disagg:
            self.engine_args.disagg_options.token_port = get_free_port()

    def _config_inner_engine_logger(self, engine_args: ServingArgs):
        loguru_logger.remove()
        loguru_logger.add(
            sys.stderr,
            level=engine_args.log_level,
            format=LOGGER_FORMAT,
        )

    async def _start_engine(self):
        if self.engine_args.enable_disagg:
            communication_args = CommunicationArgs(
                instance_name=self.engine_args.disagg_options.inst_id,
                pull_port=self.engine_args.disagg_options.token_port,
                resp_port=self.engine_args.disagg_options.token_port + 1,
                zmq_timeout=self.engine_args.serving_multi_processing_options.zmq_timeout,
            )
            msg_server = EngineMsgServer(engine=self.engine, args=communication_args)
            await msg_server.async_start(asyncio.get_event_loop(), disable_frontend_multiprocessing=True,
                                         enable_disagg_pd=True)
        await self.engine.async_start(asyncio.get_event_loop())

        if self.migration_config.enable_migration:
            self.worker_infos = []
            self.kv_transfer_instance_id = self.instance_id
            if self.engine_args.enable_disagg and self.engine_args.disagg_options is not None:
                self.kv_transfer_instance_id = self.engine_args.disagg_options.inst_id
            for index, ip_addr in enumerate(self.engine.src_workers_migration_ip_addr_list):
                self.worker_infos.append(
                    WorkerInfo(ip_address=ip_addr, instance_id=self.instance_id,
                            kv_transfer_instance_id=self.kv_transfer_instance_id, worker_id=index))

        self.msg_decoder = msgspec.msgpack.Decoder()
        self._engine_ready_event.set()

    def stop(self):
        self.engine.stop()
        logger.info("Engine stops (instance_id: {}).".format(self.instance_id))

    async def is_ready(self) -> bool:
        await self._engine_ready_event.wait()
        return True

    @property
    def _stop_event(self):
        # pylint: disable=protected-access
        return self.engine._stop_event

    @property
    def state(self):
        return self.engine.state

    def _get_engine_cls(self):
        engine_cls = None
        if not self.engine_args.enable_disagg:
            engine_cls = AsyncLLMEngineLlumnix
        else:
            if self.engine_args.disagg_options.inst_role == InstanceRole.PREFILL:
                engine_cls = PrefillAsyncLLMEngineLlumnix
            else:
                engine_cls = DecodeAsyncLLMEngineLlumnix
        return engine_cls

    # -------------- dispatch related method --------------

    async def add_request(self, request_id: int, request_processing_context: RequestProcessingContext, expected_steps: int, *args, **kwargs) -> None:
        assert "server_request" in kwargs and kwargs["server_request"]
        server_request = msgspec.convert(self.msg_decoder.decode(kwargs["server_request"]), type=ServerRequest)
        # The instance ID of the decode instance. If provided, engine will skip dispatch decode instance after prefilling.
        if self.engine_args.enable_disagg:
            decode_instance_id = kwargs.get("decode_instance_id", "")
            if decode_instance_id:
                server_request.decode_instances = [decode_instance_id]

        if self.engine_args.enable_semi_pd_mode:
            server_request.kvt_meta_info = {}

            server_request.kvt_meta_info['semi_p_inst_id'] = kwargs.get("semi_p_inst_id", None)
            server_request.kvt_meta_info['semi_d_inst_id'] = kwargs.get("semi_d_inst_id", None)
            assert server_request.kvt_meta_info['semi_d_inst_id'] == self.engine_disagg_inst_id

            # If the decode instance is the same as the prefill instance, ignore pdd.
            if server_request.kvt_meta_info['semi_p_inst_id'] == server_request.kvt_meta_info['semi_d_inst_id']:
                server_request.kvt_meta_info['semi_d_inst_id'] = ""
                server_request.kvt_meta_info['semi_p_inst_id'] = ""

        if self.engine_args.enable_semi_pd_mode:
            # TODO(KuilongCui): add exception handle
            add_request_exception_wrapper = exception_wrapper_async(self.engine.add_request_wrapper)
            asyncio.create_task(add_request_exception_wrapper(request_processing_context, server_request))
        else:
            await self.engine.add_request_wrapper(request_processing_context, server_request)

    async def abort_request(self, request_id: Union[int, Iterable[int]]) -> None:
        if isinstance(request_id, int):
            request_id = (request_id,)
        request_ids = set(request_id)
        for req_id in request_ids:
            await self.engine.drop_request(req_id)

    # -------------- migration related method --------------

    async def _run_workers_async(self, *args, timeout=RAY_RPC_TIMEOUT, **kwargs) -> List[Any]:
        return await self.engine.run_workers_async(*args, timeout=timeout, **kwargs)

    def get_running_queue(self):
        return self.engine.scheduler.get_running_queue()

    def get_waiting_queue(self):
        return self.engine.scheduler.get_waiting_queue()

    async def get_request_incremental_blocks(self, backend_request: GenerationGroupStateLlumnix,
                                             pre_stage_num_blocks: int) -> List[int]:
        if backend_request.should_abort_migration():
            return [], [], False

        incremental_blocks = self.engine.scheduler.get_request_incremental_blocks(backend_request, pre_stage_num_blocks)
        is_last_stage = (len(incremental_blocks) <= self.migration_config.migration_last_stage_max_blocks) \
            or backend_request.blocking_migration
        if is_last_stage:
            request_barrier = RequestBarrier(backend_request)
            self.request_barriers.put_nowait(request_barrier)
            if self.engine._migration_semaphore.locked():
                self.engine._migration_semaphore.release()
            await request_barrier.wait()

            if backend_request.should_abort_migration() or self.engine.scheduler.is_hunger():
                # If there are currently no requests that can be scheduled, and the request migration process is aborted,
                # wake the engine to start the next round of scheduling.
                if self.engine.scheduler.is_hunger():
                    self.engine.scheduler.running_filter_request_ids.remove(backend_request.request_id)
                    if self.engine._migration_semaphore.locked():
                        self.engine._migration_semaphore.release()
                return [], [], False

            if not backend_request.should_abort_migration():
                incremental_blocks = self.engine.scheduler.get_request_incremental_blocks(backend_request, pre_stage_num_blocks)
                backend_request.detokenizer_migration_state = self.engine.scheduler._detokenizer.get_state(backend_request.request_id)
                backend_request.req_tracker_migration_state = self.engine._req_tracker.get_req_metrics(backend_request.request_id)
            else:
                self.engine.scheduler.running_filter_request_ids.remove(backend_request.request_id)

        return incremental_blocks, [], is_last_stage

    def remove_waiting_request(self, *args, **kwargs) -> bool:
        return self.engine.scheduler.remove_waiting_request(*args, **kwargs)

    def add_waiting_request(self, *args, **kwargs) -> None:
        self.engine.scheduler.add_waiting_request(*args, **kwargs)

    async def remove_running_request(self, request_id: int) -> bool:
        ret = self.engine.scheduler.remove_running_request(request_id)
        self.engine.scheduler.running_filter_request_ids.remove(request_id)
        self.engine._req_tracker.remove_span(request_id)
        return ret

    def add_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler.add_migrating_out_request_last_stage(*args, **kwargs)

    def pop_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler.pop_migrating_out_request_last_stage(*args, **kwargs)

    def pre_alloc_cache(self, *args, **kwargs) -> MigrationResponse:
        return self.engine.scheduler.pre_alloc_cache(*args, **kwargs)

    async def add_running_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        self.engine.trans_wrapper.add_request(backend_request.request_id, backend_request.request_processing_context)
        self.engine._req_tracker.req_metrics_map[backend_request.request_id] = backend_request.req_metrics
        return self.engine.scheduler.add_running_request(backend_request)

    def free_pre_alloc_cache(self, *args, **kwargs) -> None:
        return self.engine.scheduler.free_pre_alloc_cache(*args, **kwargs)

    def free_src_request(self, backend_request: LlumnixRequest) -> None:
        expired_step = self.engine.step_counter + 1
        self.engine.trans_wrapper.remove_request_server_info(backend_request.request_id, expired_step)
        return self.engine.scheduler.free_src_request(backend_request)

    async def send_cache(self,
                         dst_instance_actor: ray.actor.ActorHandle,
                         src_blocks: List[int],
                         dst_blocks: List[int],
                         request_id: int,
                         is_last_stage: bool) -> MigrationResponse:
        return await asyncio_wait_for_ray_remote_call_with_timeout(
            dst_instance_actor.execute_migration_method_async,
            "recv_cache",
            request_id=request_id,
            src_worker_handle_list=self.worker_infos,
            dst_blocks=dst_blocks,
            src_blocks=src_blocks,
            is_last_stage=is_last_stage
        )

    async def recv_cache(self,
                         request_id: RequestIDType,
                         src_worker_handle_list: List[Any],
                         src_blocks: List[int],
                         dst_blocks: List[int],
                         is_last_stage: bool) -> MigrationResponse:
        request = RecvCacheRequest(
            src_worker_handle_list=src_worker_handle_list,
            request_id=request_id,
            is_last_stage=is_last_stage,
            src_blocks=src_blocks,
            dst_blocks=dst_blocks,
        )
        responses = await self._run_workers_async("recv_cache", request)
        is_ok_list = [response.is_ok for response in responses]
        return MigrationResponse(success=all(is_ok_list), return_value=None)

    async def commit_dst_request(self,
                                 request_id: RequestIDType,
                                 backend_request: GenerationGroupStateLlumnix) -> MigrationResponse:
        assert len(backend_request.paged_reqs) == 1, "Currently llumnix doesn't support multi-paged request migration."

        seq = backend_request.paged_reqs[0]
        seq.block_table_id = next(self.engine.scheduler.block_manager.block_table_counter)
        pre_alloc_blocks = self.engine.scheduler.pre_alloc_cache_dict.pop(request_id)
        self.engine.scheduler.add_block_table(pre_alloc_blocks, seq.block_table_id)

        backend_request.reset_migration_states_dst()
        self.engine._back_queue[request_id] = self.engine.resp_queue
        self.engine._req_tracker.req_metrics_map[request_id] = backend_request.req_tracker_migration_state
        await self.add_running_request(backend_request)

        self.engine.scheduler.llumnix_metrics.scheduler_step_metrics(self.engine.scheduler)
        if self.engine._migration_semaphore.locked():
            self.engine._migration_semaphore.release()

        return MigrationResponse(success=True, return_value=None)

    def get_stats(self) -> Stats:
        return self.engine.get_stats()

    async def get_metrics(self) -> str:
        return await self.engine.get_metrics()

    async def start_profiler(self):
        return await self.engine.start_profiler()

    async def stop_profiler(self):
        return await self.engine.stop_profiler()

    def get_instance_info(self):
        return self.engine.instance_info

    def get_engine_context(self):
        return InstanceContext(local_engine_id=self.engine_disagg_inst_id)
