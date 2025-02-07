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

from functools import partial
import json
import traceback
from typing import List, Optional, Tuple, Union, Iterable, Any
from collections import defaultdict
import threading
import asyncio
import queue

import ray
import grpc
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from blade_llm.service.engine import AsyncLLMEngine
from blade_llm.service.args import ServingArgs
from blade_llm.protocol import GenerateStreamResponse, ServerRequest
from blade_llm.service.proto.bladellm_pb2 import WorkerStepResponse
from blade_llm.service.communications.engine_wrapper import APIWrapper
from blade_llm.utils.disagg_utils import InstanceRole
from blade_llm.service.disagg_pd_engine import PrefillAsyncLLMEngine, DecodeAsyncLLMEngine
from blade_llm.service.worker import launch_worker

from llumnix.entrypoints.setup import get_ip_address
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.internal_config import MigrationConfig
from llumnix.server_info import ServerInfo
from llumnix.backends.utils import AsyncPutQueueActor
from llumnix.llumlet.request import LlumnixRequest, RequestStatus, RequestInferenceType
from llumnix.instance_info import InstanceInfo
from llumnix.queue.queue_type import QueueType
from llumnix.logging.logger import init_logger
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc
from llumnix.backends.bladellm.proto.migration_worker_pb2 import MigrateCacheRequest, WorkerInfo
from llumnix.backends.bladellm.sequence import GenerationGroupStateLlumnix

logger = init_logger(__name__)


class RequestBarrier:
    def __init__(self, request_id: int):
        self.request_id = request_id
        self.wait_event = asyncio.Event()

    def notify(self):
        self.wait_event.set()

    async def wait(self):
        await self.wait_event.wait()

class AsyncBackQueueWrapper(APIWrapper):
    def __init__(self, placement_group, instance_id, request_output_queue_type) -> None:
        super().__init__(args=None, resp_queue=None)
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )
        self.put_queue_args_queue = queue.Queue()
        self.put_queue_loop_thread = threading.Thread(
            target=self._put_request_outputs_loop, args=(), daemon=True, name="put_queue_loop"
        )
        self.async_put_queue_actor = ray.remote(
            num_cpus=1,
            scheduling_strategy=scheduling_strategy
        )(AsyncPutQueueActor).remote(instance_id, request_output_queue_type)
        self.put_queue_loop_thread.start()

        self.request_server_map = {}

    def _put_request_outputs_loop(self):
        while True:
            request_outputs, req_id_outputs, server_info_outputs = [], [], []

            resp, req_id, server_info = self.put_queue_args_queue.get()
            request_outputs.append(resp)
            req_id_outputs.append(req_id)
            server_info_outputs.append(server_info)

            if self.put_queue_args_queue.qsize() > 0:
                request_size = self.put_queue_args_queue.qsize()
                for _ in range(request_size):
                    resp, req_id, server_info = self.put_queue_args_queue.get()
                    request_outputs.append(resp)
                    req_id_outputs.append(req_id)
                    server_info_outputs.append(server_info)

            self._put_request_outputs_to_server(request_outputs, req_id_outputs, server_info_outputs)

    def _put_request_outputs_to_server(self, request_outputs: List[GenerateStreamResponse],
                                       req_ids: List[str], server_infos: List[ServerInfo]) -> None:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}
        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output, req_id, server_info in zip(request_outputs, req_ids, server_infos):
            server_id = server_info.server_id
            server_request_outputs[server_id].append((req_id, request_output.model_dump_json()))
            if server_id not in server_info_dict:
                server_info_dict[server_id] = server_info
        logger.debug("server_request_outputs: {}".format(server_request_outputs))
        self.async_put_queue_actor.put_nowait_to_servers.remote(server_request_outputs, server_info_dict)

    # pylint: disable=unused-argument
    async def send(self, req_id, msg, reset=False):
        if req_id not in self.request_server_map:
            logger.warning("req_id {} not in request_server_map, maybe already be migrated.".format(req_id))
            return

        self.put_queue_args_queue.put_nowait((msg, str(req_id), self.request_server_map[req_id]))
        if msg.is_finished:
            logger.debug("trans_wrapper finish_request {}".format(req_id))
            self.request_server_map.pop(req_id, None)

    async def recv(self):
        return None

    def drop_request(self, request_id: int) -> None:
        self.request_server_map.pop(request_id, None)
        logger.debug("trans_wrapper drop_request {}".format(request_id))

    def add_request(self, request_id: int, server_info: ServerInfo) -> None:
        self.request_server_map[request_id] = server_info
        logger.debug("trans_wrapper add_request {} {}".format(request_id, server_info))

    def clear(self):
        self.request_server_map = {}
        logger.debug("trans_wrapper reset")

    def stop(self):
        pass

class AsyncLLMEngineLlumnixMixin:
    # pylint: disable=unused-argument
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 src_worker_ip_address: List[str],
                 request_barriers: queue.Queue,
                 ) -> None:
        self.instance_id = instance_id

        self.state = EngineState.INIT
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))

        self.placement_group = placement_group
        self.request_output_queue_type = request_output_queue_type

        self._worker_processes = launch_worker(self._args, instance_id, migration_config)
        self.src_worker_ip_address = src_worker_ip_address
        self._migration_semaphore = asyncio.Semaphore(0)
        self.request_barriers: queue.Queue = request_barriers

    @property
    def instance_info(self) -> InstanceInfo:
        return self._scheduler.llumnix_metrics.to_instance_info()

    async def start(self, loop: asyncio.AbstractEventLoop):
        await super().start(loop)
        self._client = self.init_client_from_engine()
        self.trans_wrapper: AsyncBackQueueWrapper = AsyncBackQueueWrapper(self.placement_group,
                                                                          self.instance_id,
                                                                          self.request_output_queue_type)
        self._scheduler.request_server_map = self.trans_wrapper.request_server_map
        self._scheduler.llumnix_metrics.engine_init_metrics(self)

    def inject_request_barriers(self):
        async def finish_callback(resp_list, request_barriers: List[RequestBarrier], running_filter_request_ids: List[int]):
            for request_barrier in request_barriers:
                request_barrier.notify()
                self._scheduler.remove_running_request(request_barrier.request_id)
            self._scheduler.running_filter_request_ids = self._scheduler.running_filter_request_ids - running_filter_request_ids

        barrier_size = self.request_barriers.qsize()
        if barrier_size > 0:
            all_request_barriers = []
            running_filter_request_ids = set()
            for _ in range(barrier_size):
                request_barrier = self.request_barriers.get()
                all_request_barriers.append(request_barrier)
                running_filter_request_ids.add(request_barrier.request_id)
            self._scheduler.running_filter_request_ids.update(running_filter_request_ids)
            self._workers.barrier(callback=partial(finish_callback, request_barriers=all_request_barriers,
                                                   running_filter_request_ids=running_filter_request_ids))

    def _update_request_inference_type(self, resp_list: List[WorkerStepResponse]):
        request_groups = resp_list[0].generation_groups.generation_group
        for gen_group in request_groups:
            request_group_id = gen_group.request_group_id
            if request_group_id in self.scheduler.id2group:
                num_out_token = gen_group.generations[0].num_out_token
                self._scheduler.id2group[request_group_id]._inference_type = \
                    RequestInferenceType.DECODE if num_out_token > 0 else RequestInferenceType.PREFILL

    async def update_callback(self, resp_list, step_requests):
        logger.debug("update_callback {} {}".format(resp_list, step_requests))
        await super().update_callback(resp_list, step_requests)
        self._update_request_inference_type(resp_list)
        self.scheduler.llumnix_metrics.engine_step_metrics(self.scheduler)

    async def _loop(self):
        previous_state = self.state
        self.state = EngineState.RUNNING
        logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        try:
            await super()._loop()
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))

            previous_state = self.state
            self.state = EngineState.CRASHED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        if self.state == EngineState.RUNNING:
            self.state = EngineState.STOPPED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, EngineState.RUNNING, self.state))

    async def _handle_dropped_request(self):
        for req_id in self._dropped_req:
            self._scheduler.id2group[req_id]._status = RequestStatus.FINISHED
            self.trans_wrapper.drop_request(req_id)
        await super()._handle_dropped_request()

    async def _handle_abort(self, abort: Optional[List[Tuple[int, int, str]]] = None):
        await super()._handle_abort(abort)
        if abort is not None and len(abort) > 0:
            for req_id, _, _ in abort:
                self._scheduler.id2group[req_id]._status = RequestStatus.FINISHED
                self.trans_wrapper.drop_request(req_id)

    async def _handle_reset(self):
        await super()._handle_reset()
        self.trans_wrapper.clear()

    async def add_request(self, server_info: ServerInfo, server_request: ServerRequest):
        logger.debug("engine {} add request {}".format(self.instance_id, server_request))
        self.trans_wrapper.add_request(server_request.id, server_info)
        # pylint: disable=protected-access
        await self._client._add_request(server_request)

    async def drop_request(self, req_id: int):
        logger.debug("engine {} drop request {}".format(self.instance_id, req_id))
        await self._client.drop_request(req_id)

    async def run_workers(self, worker_method, *args, **kwargs):
        coros = []
        channels = [grpc.aio.insecure_channel(worker) for worker in self.src_worker_ip_address]
        for channel in channels:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)
            method = getattr(stub, worker_method)
            coros.append(method(*args, **kwargs))
        result = await asyncio.gather(*coros, return_exceptions=True)
        for channel in channels:
            await channel.close()
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
        return len(self.trans_wrapper.request_server_map)

    def get_all_request_ids(self) -> List[int]:
        return list(self._scheduler.get_all_request_ids())

    def get_num_wait_update_request_ids(self) -> int:
        return self.request_barriers.qsize()

class AsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, AsyncLLMEngine):
    def __init__(self,
                instance_id: str,
                placement_group: PlacementGroup,
                request_output_queue_type: QueueType,
                migration_config: MigrationConfig,
                src_worker_ip_address: List[str],
                request_barriers: queue.Queue,
                *args, **kwargs,
                ) -> None:
        AsyncLLMEngine.__init__(self, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(self, instance_id, placement_group, request_output_queue_type,
                                            migration_config, src_worker_ip_address, request_barriers)

class PrefillAsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, PrefillAsyncLLMEngine):
    def __init__(self,
            instance_id: str,
            placement_group: PlacementGroup,
            request_output_queue_type: QueueType,
            migration_config: MigrationConfig,
            src_worker_ip_address: List[str],
            request_barriers: queue.Queue,
            *args, **kwargs,
            ) -> None:
        PrefillAsyncLLMEngine.__init__(self, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(self, instance_id, placement_group, request_output_queue_type,
                                            migration_config, src_worker_ip_address, request_barriers)

class DecodeAsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, DecodeAsyncLLMEngine):
    def __init__(self,
            instance_id: str,
            placement_group: PlacementGroup,
            request_output_queue_type: QueueType,
            migration_config: MigrationConfig,
            src_worker_ip_address: List[str],
            request_barriers: queue.Queue,
            *args, **kwargs,
            ) -> None:
        DecodeAsyncLLMEngine.__init__(self, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(self, instance_id, placement_group, request_output_queue_type,
                                            migration_config, src_worker_ip_address, request_barriers)

class BackendBladeLLM(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: ServingArgs
    ) -> None:
        self.instance_id = instance_id
        self.engine_args = engine_args
        engine_cls = self._get_engine_cls()
        self.engine = engine_cls(instance_id,
                                 placement_group,
                                 request_output_queue_type,
                                 migration_config,
                                 engine_args)
        self.migration_config = migration_config

        ip_addr = get_ip_address()
        world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
        src_worker_start_port = self.migration_config.migration_backend_server_port
        src_ports = range(src_worker_start_port, src_worker_start_port+world_size)
        self.src_worker_ip_address = [ip_addr+":"+str(port) for port in src_ports]

        self.request_barriers: queue.Queue = queue.Queue()
        engine_cls = self._get_engine_cls()
        self.engine = engine_cls(instance_id, output_queue_type, migration_config, placement_group,
                                node_id, self.src_worker_ip_address, self.request_barriers, engine_args)

        self._engine_ready_event = asyncio.Event()
        asyncio.create_task(self._start_engine())

    async def _start_engine(self):
        await self.engine.start(asyncio.get_event_loop())
        self._engine_ready_event.set()

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

    async def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        assert "server_request" in kwargs and kwargs["server_request"]
        server_request = ServerRequest(**json.loads(kwargs["server_request"]))
        await self.engine.add_request(server_info, server_request)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for req_id in request_ids:
            self.engine.drop_request(req_id)

    def get_all_request_ids(self) -> List[int]:
        return self.engine.scheduler.get_all_request_ids()

    # -------------- migration related method --------------

    async def _run_workers(self, *args, **kwargs):
        return await self.engine.run_workers(*args, **kwargs)

    def get_running_queue(self):
        return self.engine.scheduler.get_running_queue()

    def get_waiting_queue(self):
        return self.engine.scheduler.get_waiting_queue()

    async def get_request_incremental_blocks(self, backend_request: GenerationGroupStateLlumnix,
                                             pre_stage_num_blocks: int) -> List[int]:
        if backend_request.request_id not in self.engine.scheduler.id2group:
            return [], False

        incremental_blocks = self.engine.scheduler.get_request_incremental_blocks(backend_request, pre_stage_num_blocks)
        is_last_stage = (len(incremental_blocks) <= self.migration_config.last_stage_max_blocks) or backend_request.blocking_migration
        if is_last_stage:
            request_barrier = RequestBarrier(backend_request.request_id)
            self.request_barriers.put_nowait(request_barrier)
            self.engine.inject_request_barriers()
            await request_barrier.wait()
            incremental_blocks = self.engine.scheduler.get_request_incremental_blocks(backend_request, pre_stage_num_blocks)

        return incremental_blocks, is_last_stage

    def remove_waiting_request(self, *args, **kwargs) -> bool:
        return self.engine.scheduler.remove_waiting_request(*args, **kwargs)

    def add_waiting_request(self, *args, **kwargs) -> None:
        self.engine.scheduler.add_waiting_request(*args, **kwargs)

    # In last stage migration, request has been removed in get_request_incremental_blocks barrier from running
    def remove_running_request(self, request_id: int) -> None:
        return request_id in self.engine.scheduler.id2group

    def add_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler.add_migrating_out_request_last_stage(*args, **kwargs)

    def remove_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler.remove_migrating_out_request_last_stage(*args, **kwargs)

    def pop_migrating_out_requests_last_stage(self, *args, **kwargs) -> List[Any]:
        return self.engine.scheduler.pop_migrating_out_requests_last_stage(*args, **kwargs)

    def pre_alloc(self, *args, **kwargs) -> List[int]:
        return self.engine.scheduler.pre_alloc(*args, **kwargs)

    def add_running_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        self.engine.trans_wrapper.add_request(backend_request.request_id, backend_request.server_info,
                                      backend_request.expected_steps)
        return self.engine.scheduler.add_running_request(backend_request)

    def free_dst_pre_alloc_cache(self, *args, **kwargs) -> None:
        return self.engine.scheduler.free_dst_pre_alloc_cache(*args, **kwargs)

    def free_src_request(self, backend_request: LlumnixRequest) -> None:
        self.engine.trans_wrapper.drop_request(backend_request.request_id)
        return self.engine.scheduler.free_src_request(backend_request)

    async def send_blocks(self, dst_ray_actor: ray.actor.ActorHandle, request_id: int,
                          src_blocks: List[int], dst_blocks: List[int], has_more: bool):
        worker_infos = []
        for index, ip_addr in enumerate(self.src_worker_ip_address):
            worker_infos.append(WorkerInfo(ip_address=ip_addr, instance_id=self.instance_id, worker_id=index))
        request = MigrateCacheRequest(
            src_handlers=worker_infos,
            request_id=request_id,
            has_more=has_more,
            src_blocks=src_blocks,
            dst_blocks=dst_blocks,
        )
        await dst_ray_actor.execute_async_engine_method.remote("_run_workers", "migrate_cache", request)

    def commit_dst_request(self, backend_request: LlumnixRequest) -> None:
        assert len(backend_request.paged_reqs) == 1, "currently llumnix doesn't support multi-paged-req migration."

        seq = backend_request.paged_reqs[0]
        seq.block_table_id = next(self.engine.scheduler.block_manager.block_table_counter)
        pre_alloc_blocks = self.engine.scheduler.pre_alloc_cache_dict.pop(backend_request.request_id)
        self.engine.scheduler.add_block_table(pre_alloc_blocks, seq.block_table_id)

        backend_request.reset_migration_args_dst()
        self.engine.trans_wrapper.add_request(backend_request.request_id, backend_request.server_info)
        self.engine._back_queue[backend_request.request_id] = None

        self.engine.scheduler._detokenizer.add_new_request(backend_request.paged_reqs[0].req_proto)
        self.engine.scheduler.add_running_request(backend_request)

        self.engine.scheduler.llumnix_metrics.scheduler_step_metrics(self.engine.scheduler)
        if self.engine._migration_semaphore.locked():
            self.engine._migration_semaphore.release()
        logger.debug("commit dst request {}".format(backend_request.request_id))
