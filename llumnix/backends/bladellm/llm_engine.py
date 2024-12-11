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

import json
import time
import traceback
from typing import Any, List, Optional, Dict, Tuple, Union, Iterable
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
from llumnix.internal_config import MigrationConfig
from llumnix.server_info import ServerInfo
from llumnix.queue.utils import QueueType

class AsyncBackQueue(APIWrapper):
    def __init__(self, placement_group, node_id, instance_id, output_queue_type) -> None:
        super().__init__(args=None, resp_queue=None)
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

        self.request_client_map = {}

    async def _start_put_queue_loop(self):
        while True:
            request_outputs, server_info_outputs = [], []

            resp, server_info = self.put_queue_args_queue.get()
            request_outputs.append(resp)
            server_info_outputs.append(server_info)

            if self.put_queue_args_queue.qsize() > 0:
                request_size = self.put_queue_args_queue.qsize()
                for _ in range(request_size):
                    resp, server_info = self.put_queue_args_queue.get()
                    request_outputs.append(resp)
                    server_info_outputs.append(server_info)

            self._put_request_outputs_to_server(request_outputs, server_info_outputs)
            asyncio.sleep(0)

    def _put_request_outputs_to_server(self, request_outputs: List[GenerateStreamResponse], server_infos: List[ServerInfo]) -> None:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}
        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output, server_info in zip(request_outputs, server_infos):
            server_id = server_info.server_id
            server_request_outputs[server_id].append(request_output.model_dump_json())
            if server_id not in server_info_dict:
                server_info_dict[server_id] = server_info
        logger.debug("_put_request_outputs_to_server, {}", server_request_outputs)
        self.async_put_queue_actor.put_nowait_to_servers.remote(server_request_outputs, server_info_dict)

    async def send(self, req_id, msg, reset=False):
        self.put_queue_args_queue.put_nowait(msg, self.request_client_map[req_id])
        if msg.is_finished:
            self.request_client_map.pop(req_id)

    async def recv(self):
        return None

    def drop_request(self, request_id: int) -> None:
        self.request_client_map.pop(request_id)

    def add_request(self, request_id: str, server_info: ServerInfo) -> None:
        self.request_client_map[request_id] = server_info

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
        self.instance_id = instance_id

        self.state = EngineState.INIT
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))

        self.placement_group = placement_group
        self.output_queue_type = output_queue_type
        self.node_id = node_id

    @property
    def instance_info(self):
        pass
        # self._instance_info.num_used_gpu_blocks = self._scheduler.
        # num_blocks_all_waiting_requests
        # num_running_requests
        # num_waiting_requests
        # num_available_gpu_blocks

    def start(self, loop: asyncio.AbstractEventLoop):
        super().start(loop)
        self._client = self.init_client_from_engine()
        self.trans_warpper: AsyncBackQueue = AsyncBackQueue(self.placement_group,
                              self.node_id, self.instance_id, self.output_queue_type)
        self._scheduler.engine_init_metrics(self)

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

    async def _handle_dropped_request(self):
        if self._dropped_req:
            for req_id in self._dropped_req:
                self.trans_warpper.drop_request(req_id)
        await super().drop_req()

    async def _handle_abort(self, abort: Optional[List[Tuple[int, int, str]]] = None):
        if abort is not None and len(abort) > 0:
            for req_id, _, _ in abort:
                self.trans_warpper.drop_request(req_id)
        await super().abort(abort)

    async def add_request(self, server_request: ServerRequest, server_info: ServerInfo):
        self.trans_warpper.add_request(server_request.id, server_info)
        await self._client._add_request(server_request)

    async def drop_request(self, req_id: int):
        await self._client.drop_request(req_id)

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
        decode_actor = ray.get_actor("instance_"+inst_id, namespace="llumnix")
        response = await decode_actor.exec_entrypoint_method.remote("inner_"+url.split("/")[-1], data)
        return json.loads(response)

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
        engine_cls = self._get_engine_cls()
        self.engine = engine_cls(instance_id, output_queue_type,migration_config, placement_group, node_id, engine_args)
        
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

    def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, server_request: str) -> None:
        logger.debug("engine {} add request {} {}", self.instance_id, request_id, server_request)
        server_request = ServerRequest(**json.loads(server_request))
        asyncio.run_coroutine_threadsafe(
            self.engine.add_request(server_request, request_id, server_info, expected_steps), self._loop)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for req_id in request_ids:
            self.engine.drop_request(int(req_id))

    def exec_entrypoint_method(self, method, *args, **kwargs):
        executor = getattr(self.engine.entrypoint, method)
        return asyncio.run_coroutine_threadsafe(executor(*args, **kwargs), self._loop).result()
