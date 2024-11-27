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
import ray
import json
import grpc
from google.protobuf.empty_pb2 import Empty
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from llumnix.backends.bladellm.proto.migration_worker_pb2 import MigrateRequests, MigrateResGroupRequests
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from blade_llm.service.engine import AsyncLLMEngine
from blade_llm.service.args import ServingArgs
from blade_llm.utils.counter import Counter
from blade_llm.protocol import GenerateStreamResponse
from blade_llm.protocol import ServerRequest
from blade_llm.service.worker import launch_worker

from blade_llm.service.communications.engine_wrapper import APIWrapper
from blade_llm.protocol import (
    GenerateStreamResponse,
    ServerRequest,
)
from llumnix.backends.bladellm.queue import AsyncPutQueueActor
from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.backends.bladellm.sequence import ServerRequestLlumnix, GenerationGroupStateLlumnix, GenerateStreamResponseLlumnix
from llumnix.llumlet.request import LlumnixRequest
from llumnix.internal_config import MigrationConfig
from llumnix.server_info import ServerInfo
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.queue.utils import QueueType, init_output_queue_client
from llumnix.backends.bladellm.utils import string_to_int, get_model_conf
from llumnix.backends.bladellm.worker import launch_worker
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.queue.utils import QueueType
from llumnix.backends.bladellm.scheduler import PagedSchedulerLlumnix

logger = init_logger(__name__)
from loguru import logger as my_logger
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
        my_logger.info(server_request_outputs)
        self.async_put_queue_actor.put_nowait_to_servers.remote(server_request_outputs, server_info_dict)

    async def send(self, req_id, msg, reset=False):
        self.put_queue_args_queue.put_nowait(msg)

    async def recv(self):
        return None

    def stop(self):
        pass

class LLMEngineLlumnix(AsyncLLMEngine):
    def __init__(self,
                 instance_id: str,
                 output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 placement_group: Optional[PlacementGroup],
                 node_id: Optional[str],
                 *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self._worker_processes = launch_worker(self._args, instance_id, migration_config)

        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        self.state = EngineState.INIT
        self.placement_group = placement_group
        self.output_queue_type = output_queue_type
        self.node_id = node_id
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))

    def start(self, loop: asyncio.AbstractEventLoop):
        super().start(loop)
        self._client = self.init_client_from_engine()

    async def _init_scheduler(self) -> None:
        await super()._init_scheduler()
        self._scheduler.add_update_instance_info_callback(self.update_instance_info)

    def init_trans_wrapper(self, engine, serving_args: ServingArgs, resp_queue, *args, **kwargs) -> APIWrapper:
        return AsyncBackQueue(serving_args, resp_queue, self.placement_group,
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
        # TODO[xinyi]: async step may lead to put info not in order for both put_queue_args_queue and step_request_queue
        await super().step()

        self.instance_info.instance_id = self.instance_id
        self.instance_info.step_id = next(self.step_counter)
        self.instance_info.timestamp = time.time()
        # TODO(KuilongCui): update instance_info.profiling_data
        
        gen_groups = self._scheduler.running
        my_logger.info(gen_groups)
        if gen_groups:
            tot_blocks = []
            for req_state in gen_groups[-1].paged_reqs:
                blocks = self._scheduler.block_manager.get_block_table(req_state)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            self.instance_info.num_blocks_last_running_request = len(tot_blocks)

    async def update_callback(self, resp_list, step_requests):
        my_logger.info(resp_list)
        my_logger.info(step_requests)

        for resp in resp_list:
            my_logger.info(resp)
            request_groups = resp.generation_groups.generation_group
            request_groups_map = self._scheduler.get_request_groups_map()
            for req_state in request_groups:
                req_id = req_state.request_group_id
                if req_id in request_groups_map:
                    request_groups_map[req_id].update_num_computed_tokens(request_groups_map[req_id].token_chunk_size)

        await super().update_callback(resp_list, step_requests)

    #TODO[xinyi]: the same to the function in vllm, maybe put into utils.py
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

    def stop(self):
        self.engine.stop()

class BackendBladeLLM(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: ServingArgs,
        placement_group: PlacementGroup = None,
        node_id: str = None
    ) -> None:

        self.worker_addrs_list = [
            migration_worker_pb2.WorkerInfo(
                ip_address=addr,
                instance_id=int(instance_id),
                worker_id=idx
            )
            for idx, addr in enumerate(migration_config.migration_backend_server_address.split(","))
        ]

        self.engine: LLMEngineLlumnix = LLMEngineLlumnix(instance_id, output_queue_type, migration_config,
                                                            placement_group, node_id, engine_args)
        self.instance_id = instance_id
        self._loop = asyncio.new_event_loop()
        self._engine_ready = threading.Event()
        self._thread = threading.Thread(target=self._start_loop, args=(self._loop,), daemon=True, name="async_engine")
        self._thread.start()
        self._engine_ready.wait()
    
    def _start_loop(self, loop):
        asyncio.set_event_loop(loop)
        self.engine.start(loop)
        self._engine_ready.set()
        loop.run_forever()

    # TODO(KuilongCui): change add_request to async
    def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, server_request: str) -> None:
        server_request = ServerRequestLlumnix(server_request, request_id, server_info, expected_steps)
        asyncio.run_coroutine_threadsafe(self.engine.add_request(server_request), self._loop)

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
        future= asyncio.run_coroutine_threadsafe(self.engine._run_workers(worker_method, *args, **kwargs), self._loop)
        # future.result()
    
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
