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
import traceback
from typing import Any, List, Optional, Dict, Union, Iterable
from collections import defaultdict
import threading
import asyncio
import queue

from aiohttp import web
import ray
import grpc
import pickle
import zmq
from loguru import logger
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy

from blade_llm.service.engine import AsyncLLMEngine
from blade_llm.service.args import ServingArgs
from blade_llm.utils.counter import Counter
from blade_llm.protocol import GenerateStreamResponse, RemoteGenerateStreamResponse, ServerRequest
from blade_llm.service.worker import launch_worker
from blade_llm.service.communications.engine_wrapper import APIWrapper
from blade_llm.protocol import GenerateStreamResponse
from blade_llm.utils.disagg_utils import InstanceRole
from blade_llm.service.disagg_decode_server import DecodeEntrypoint
from blade_llm.service.disagg_pd_engine import PrefillAsyncLLMEngine, DecodeAsyncLLMEngine

from llumnix.backends.bladellm.queue import AsyncPutQueueActor
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.backends.bladellm.sequence import ServerRequestLlumnix, GenerationGroupStateLlumnix, RemoteGenerateStreamResponseLlumnix
from llumnix.llumlet.request import LlumnixRequest
from llumnix.internal_config import MigrationConfig
from llumnix.server_info import ServerInfo
from llumnix.backends.bladellm.proto.migration_worker_pb2 import MigrateRequests, MigrateResGroupRequests
from llumnix.queue.utils import QueueType
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2

class AsyncBackQueue(APIWrapper):
    def __init__(self, args, resp_queue, placement_group, node_id, instance_id, output_queue_type) -> None:
        super().__init__(args, resp_queue=resp_queue)
        if placement_group:
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
            )
        elif node_id:
            scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
        else: # When use simulator, placement_group and node_id are both None.
            scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )
        self.put_queue_args_queue = queue.Queue()
        self.put_queue_loop_thread = threading.Thread(
            target=self._start_put_queue_loop, args=(), daemon=True, name="put_queue_loop"
        )
        self.async_put_queue_actor = ray.remote(
            num_cpus=1,
            scheduling_strategy=scheduling_strategy
        )(AsyncPutQueueActor).remote(instance_id, output_queue_type)
        self.put_queue_loop_thread.start()

    def _start_put_queue_loop(self):
        while True:
            resps = [self.put_queue_args_queue.get()]
            while self.put_queue_args_queue.qsize() > 0:
                for _ in range(self.put_queue_args_queue.qsize()):
                    resps.append(self.put_queue_args_queue.get())

            request_outputs, server_infos = [], []
            # TODO(KuilongCui): check the migrated request for last stage
            for resp in resps:
                server_infos.append(resp.server_info)
                request_outputs.append(resp)
                self._put_request_outputs_to_server(request_outputs, server_infos)

    def _put_request_outputs_to_server(self, request_outputs: List[GenerateStreamResponse], server_infos: List[ServerInfo]) -> None:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}
        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output, server_info in zip(request_outputs, server_infos):
            server_id = server_info.server_id
            server_request_outputs[server_id].append(
                request_output.model_dump_json(exclude={"server_info": True}))
            if server_id not in server_info_dict:
                server_info_dict[server_id] = server_info
        # TODO(s5u13b): Reduce the across-actor overhead.
        logger.debug("_put_request_outputs_to_server, {}", server_request_outputs)
        self.async_put_queue_actor.put_nowait_to_servers.remote(server_request_outputs, server_info_dict)

    async def send(self, req_id, msg, reset=False):
        self.put_queue_args_queue.put_nowait(msg)

    async def recv(self):
        return None

    def stop(self):
        pass

class LLMEngineLlumnixMixin:
    def __init__(self,
                 instance_id: str,
                 output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 placement_group: Optional[PlacementGroup],
                 node_id: Optional[str],
                 ) -> None:
        self._worker_processes = launch_worker(self._args, instance_id, migration_config)
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.state = EngineState.INIT
        self.placement_group = placement_group
        self.output_queue_type = output_queue_type
        self.node_id = node_id
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))
        self.instance_info = None

        # TODO(KuilongCui): "After Blade unify engine implementation, delete this
        self.trans_warpper = self.init_trans_wrapper(self, self._args)

    def start(self, loop: asyncio.AbstractEventLoop):
        super().start(loop)
        self._client = self.init_client_from_engine()

    async def _init_scheduler(self) -> None:
        await super()._init_scheduler()
        self._scheduler.set_update_instance_info_callback(self.update_instance_info)

    def init_trans_wrapper(self, engine, serving_args: ServingArgs, *args, **kwargs) -> APIWrapper:
        return AsyncBackQueue(serving_args, None, self.placement_group,
                              self.node_id, self.instance_id, self.output_queue_type)

    async def _loop(self):
        previous_state = self.state
        self.state = EngineState.RUNNING
        logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        try:
            await super()._loop()
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
            # TODO[xinyi, kuilong]: add function methods in worker for llumnix
            # self._workers.shutdown()

            previous_state = self.state
            self.state = EngineState.CRASHED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        if self.state == EngineState.RUNNING:
            self.state = EngineState.STOPPED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, EngineState.RUNNING, self.state))

    async def step(self):
        self.instance_info = self.instance_info if self.instance_info else InstanceInfo()
        self.instance_info.instance_id = self.instance_id
        self.instance_info.step_id = next(self.step_counter)
        self.instance_info.timestamp = time.time()
        await super().step()
        # TODO(KuilongCui): update instance_info.profiling_data

    async def update_callback(self, resp_list, step_requests):
        request_groups_map = self._scheduler.get_request_groups_map()
        for resp in resp_list:
            resp_gen_groups = resp.generation_groups.generation_group
            for req_state in resp_gen_groups:
                req_id = req_state.request_group_id
                if req_id in request_groups_map:
                    request_groups_map[req_id].update_num_computed_tokens(request_groups_map[req_id].token_chunk_size)

        if len(resp_list) > 0 and len(resp_list[-1].generation_groups.generation_group) > 0 and self.instance_info:
            last_running_gen_group_id = resp_list[-1].generation_groups.generation_group[-1].request_group_id
            if last_running_gen_group_id in request_groups_map:
                last_running_gen_group = request_groups_map[last_running_gen_group_id]
                tot_blocks = []
                for req_state in last_running_gen_group.paged_reqs:
                    blocks = self._scheduler.block_manager.get_block_table(req_state)
                    tot_blocks.extend(blocks)
                tot_blocks = set(tot_blocks)
                self.instance_info.num_blocks_last_running_request = len(tot_blocks)

        await super().update_callback(resp_list, step_requests)

    def update_instance_info(self, instance_info: InstanceInfo) -> None:
        if self.instance_info is not None:
            instance_info.instance_id = self.instance_info.instance_id
            instance_info.step_id = self.instance_info.step_id
            instance_info.timestamp = self.instance_info.timestamp
            instance_info.profiling_data = self.instance_info.profiling_data
            instance_info.num_blocks_last_running_request = self.instance_info.num_blocks_last_running_request
        self.instance_info = instance_info
        logger.info("update_instance_info {} {} {}".format(self.instance_info.num_used_gpu_blocks, self.instance_info.inference_type, self.instance_info.num_running_requests))

    async def add_request(self, *args, **kwargs):
        await self._client._add_request(*args, **kwargs)

    async def _run_workers(self, worker_method, *args, **kwargs):
        worker_address = self._worker_processes.migration_config.migration_backend_server_address.split(",")
        coros = []
        logger.info("_run_workers_worker_address {}".format(worker_address))

        for worker in worker_address:
            with grpc.insecure_channel(worker) as channel:
                stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)
                method = getattr(stub, worker_method)
                coros.append(method(*args, **kwargs))
        all_outputs = await asyncio.gather(*coros)

        return all_outputs

class AsyncLLMEngineLlumnix(LLMEngineLlumnixMixin, AsyncLLMEngine):
    def __init__(self,
                instance_id: str,
                output_queue_type: QueueType,
                migration_config: MigrationConfig,
                placement_group: Optional[PlacementGroup],
                node_id: Optional[str],
                *args, **kwargs,
                ) -> None:
        AsyncLLMEngine.__init__(self, *args, **kwargs)
        LLMEngineLlumnixMixin.__init__(self, instance_id, output_queue_type, migration_config, placement_group, node_id)

class PrefillAsyncLLMEngineLlumnix(LLMEngineLlumnixMixin, PrefillAsyncLLMEngine):
    def __init__(self,
            instance_id: str,
            output_queue_type: QueueType,
            migration_config: MigrationConfig,
            placement_group: Optional[PlacementGroup],
            node_id: Optional[str],
            *args, **kwargs,
            ) -> None:
        PrefillAsyncLLMEngine.__init__(self, *args, **kwargs)
        LLMEngineLlumnixMixin.__init__(self, instance_id, output_queue_type, migration_config, placement_group, node_id)

    async def _fetch_post(self, url, data, headers, inst_id):
        logger.info("fetch_post {} {} {}", url, inst_id, data)
        decode_actor = ray.get_actor("instance_"+inst_id, namespace="llumnix")
        response = await decode_actor.exec_entrypoint_method.remote("inner_"+url.split("/")[-1], data)
        logger.info("response {}", response)
        import json
        return json.loads(response)

    def server_requests_to_dict(self, reqs: List[ServerRequestLlumnix], prefill_inst_id: str, src_naming_info: str):
        all_reqs = super().server_requests_to_dict(reqs, prefill_inst_id, src_naming_info)
        for index, single_req in enumerate(all_reqs["reqs"]):
            single_req["server_info"] = str(pickle.dumps(reqs[index].server_info))
        return all_reqs

    async def post_scheduler_update(self, resp, update_output):
        self._scheduler.remove_request_from_hanging(list(resp.kv_transfer_done_ids))
        await self._handle_generated_results(update_output)

        if update_output.response is not None:
            for req_id, _ in update_output.response.items():
                if req_id in self._single_out_token_reqs:
                    self._single_out_token_reqs.remove(req_id)
        
        if update_output.reset:
            await self._handle_reset()
        elif update_output.response is not None:
            for req_id, l_resp in update_output.response.items():
                if req_id in self._back_queue:
                    # self._back_queue[req_id].put_nowait(l_resp)
                    await self.trans_warpper.send(req_id, l_resp)
                    if l_resp.is_finished:
                        del self._back_queue[req_id]

    async def _pull_tokens_stream(self):
        await self.pd_disagg_initialized.wait()
        while not self._stop_event.is_set():
            socks = dict(await self.poller.poll(0))
            if self._recv_from_decode in socks and socks[self._recv_from_decode] & zmq.POLLIN:
                prefill_recv_obj: RemoteGenerateStreamResponseLlumnix = await self._recv_from_decode.recv_pyobj()
                if prefill_recv_obj.external_id not in self.external_to_reqid:
                    logger.warning("pd_prefill request {} not found in prefill instance", prefill_recv_obj.external_id)
                    continue
                req_id = self.external_to_reqid[prefill_recv_obj.external_id]
                if req_id not in self._back_queue:
                    logger.warning("pd_prefill request {} not found in back_queue", req_id)
                    continue
                # self._back_queue[req_id].put_nowait(prefill_recv_obj)

                prefill_recv_obj.server_info = pickle.loads(eval(prefill_recv_obj.server_info))
                prefill_recv_obj.request_id = req_id

                await self.trans_warpper.send(req_id, prefill_recv_obj)

                if prefill_recv_obj.is_finished:
                    self._remove_request_state(req_id, prefill_recv_obj.external_id)
                    del self._back_queue[req_id]
            await asyncio.sleep(0)
        logger.info('stop event is set exit pull token loop')

class DecodeAsyncLLMEngineLlumnix(LLMEngineLlumnixMixin, DecodeAsyncLLMEngine):
    def __init__(self,
            instance_id: str,
            output_queue_type: QueueType,
            migration_config: MigrationConfig,
            placement_group: Optional[PlacementGroup],
            node_id: Optional[str],
            *args, **kwargs,
            ) -> None:
        DecodeAsyncLLMEngine.__init__(self, *args, **kwargs)
        LLMEngineLlumnixMixin.__init__(self, instance_id, output_queue_type, migration_config, placement_group, node_id)

    def start(self, loop: asyncio.AbstractEventLoop):
        LLMEngineLlumnixMixin.start(self, loop)
        self.entrypoint = DecodeEntrypoint(self._client, self._args)

    def _make_generation_group(self, server_req: ServerRequest):
        gen_group = super()._make_generation_group(server_req)
        llumnix_gen_group = GenerationGroupStateLlumnix(gen_group, server_req.external_id, server_req.server_info, -1)
        return llumnix_gen_group
    
    def _generate_remote_stream_responce(self, req_id, req_external_id, l_resp) -> RemoteGenerateStreamResponse:
        response = super()._generate_remote_stream_responce(req_id, req_external_id, l_resp)
        llumnix_responce = RemoteGenerateStreamResponseLlumnix(response, l_resp.request_id, l_resp.server_info)
        return llumnix_responce

class BackendBladeLLM(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: ServingArgs,
        placement_group: PlacementGroup = None,
        node_id: str = None,
        *args,
        **kwargs
    ) -> None:        
        self.instance_id = instance_id
        self.engine_args = engine_args
        self.worker_addrs_list = [
            migration_worker_pb2.WorkerInfo(
                ip_address=addr,
                instance_id=instance_id,
                worker_id=idx
            )
            for idx, addr in enumerate(migration_config.migration_backend_server_address.split(","))
        ]
        engine_cls = self._get_engine_cls()
        self.engine = engine_cls(instance_id, output_queue_type, migration_config,
                                                            placement_group, node_id, engine_args)
        self._loop = asyncio.new_event_loop()
        self._engine_ready = threading.Event()
        self._thread = threading.Thread(target=self._start_loop, args=(self._loop,), daemon=True, name="async_engine")
        self._thread.start()
        self._engine_ready.wait()

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

    def _start_loop(self, loop):
        asyncio.set_event_loop(loop)
        self.engine.start(loop)
        self._engine_ready.set()
        loop.run_forever()

    # TODO(KuilongCui): change add_request to async
    def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, server_request: str) -> None:
        server_request = ServerRequestLlumnix(server_request, request_id, server_info, expected_steps)
        logger.debug("engine {} add request {}", self.instance_id, server_request)
        asyncio.run_coroutine_threadsafe(self.engine.add_request(server_request), self._loop)

    def exec_entrypoint_method(self, method, *args, **kwargs):
        executor = getattr(self.engine.entrypoint, method)
        return asyncio.run_coroutine_threadsafe(executor(*args, **kwargs), self._loop).result()

    async def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]) -> None:
        await dst_ray_actor.execute_engine_method.remote("_run_workers",
                                                           "migrate_cache",
                                                            MigrateRequests(
                                                                dst_blocks=dst_blocks,
                                                                src_blocks=src_blocks,
                                                                src_handlers=self.worker_addrs_list),
                                                           )

    async def send_request_group(self, dst_ray_actor: "ray.actor.ActorHandle", request_id: int):
        await dst_ray_actor.execute_engine_method.remote("_run_workers",
                                                           "migrate_request_group",
                                                            MigrateResGroupRequests(
                                                                id=request_id,
                                                                src_handlers=self.worker_addrs_list),
                                                           )

    def _run_workers(self, worker_method, *args, **kwargs):
        pass
        # asyncio.run_coroutine_threadsafe(self.engine._run_workers(worker_method, *args, **kwargs), self._loop)
    
    def commit_dst_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        seq = backend_request.paged_reqs[0]
        seq.block_table_id = next(self.engine._scheduler.block_manager.block_table_counter)
        pre_alloc_blocks = self.engine._scheduler.pre_alloc_cache_dict.pop(backend_request.request_id)
        logger.info("add seq {} to block table {}".format(seq.request_id, pre_alloc_blocks))
        self.engine._scheduler.block_manager.add_block_table(pre_alloc_blocks, seq.block_table_id)
        backend_request.reset_migration_args()
        self.engine._scheduler.add_request_to_running(backend_request)
        self.engine._migrate_event.set()

    def is_ready(self):
        return True
    
    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for req_id in request_ids:
            self.engine.drop_request(req_id)
    
    def get_running_queue(self ) -> List[ServerRequestLlumnix]:
        return self.engine._scheduler.get_running_queue()

    def get_request_incremental_blocks(self, *args, **kwargs) -> List[int]:
        return self.engine._scheduler.get_request_incremental_blocks(*args, **kwargs)

    def remove_running_request(self, *args, **kwargs) -> None:
        return self.engine._scheduler.remove_running_request(*args, **kwargs)

    def add_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine._scheduler.add_migrating_out_request_last_stage(*args, **kwargs)

    def remove_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine._scheduler.remove_migrating_out_request_last_stage(*args, **kwargs)

    def pop_migrating_out_requests_last_stage(self, *args, **kwargs) -> List[Any]:
        return self.engine._scheduler.pop_migrating_out_requests_last_stage(*args, **kwargs)

    def pre_alloc(self, *args, **kwargs) -> List[int]:
        return self.engine._scheduler.pre_alloc(*args, **kwargs)

    def add_running_request(self, *args, **kwargs) -> None:
        return self.engine._scheduler.add_running_request(*args, **kwargs)

    def is_request_running(self, *args, **kwargs) -> bool:
        return self.engine._scheduler.is_request_running(*args, **kwargs)

    def free_dst_pre_alloc_cache(self, *args, **kwargs) -> None:
        return self.engine._scheduler.free_dst_pre_alloc_cache(*args, **kwargs)

    def free_src_request(self, backend_request: LlumnixRequest) -> None:
        return self.engine._scheduler.free_src_request(backend_request)

    def get_all_request_ids(self) -> List[str]:
        return self.engine._scheduler.get_all_request_ids()
