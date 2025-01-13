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
import traceback
from typing import List, Optional, Tuple, Union, Iterable, Deque
from collections import defaultdict
import threading
import asyncio
import queue

import ray
from loguru import logger
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from blade_llm.service.engine import AsyncLLMEngine
from blade_llm.service.args import ServingArgs
from blade_llm.protocol import GenerateStreamResponse, ServerRequest
from blade_llm.service.communications.engine_wrapper import APIWrapper
from blade_llm.utils.disagg_utils import InstanceRole
from blade_llm.service.disagg_pd_engine import PrefillAsyncLLMEngine, DecodeAsyncLLMEngine

from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.internal_config import MigrationConfig
from llumnix.server_info import ServerInfo
from llumnix.backends.utils import AsyncPutQueueActor
from llumnix.llumlet.request import LlumnixRequest, RequestStatus
from llumnix.instance_info import InstanceInfo
from llumnix.queue.queue_type import QueueType

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
        logger.debug("_put_request_outputs_to_server: {}", server_request_outputs)
        self.async_put_queue_actor.put_nowait_to_servers.remote(server_request_outputs, server_info_dict)

    # pylint: disable=unused-argument
    async def send(self, req_id, msg, reset=False):
        self.put_queue_args_queue.put_nowait((msg, str(req_id), self.request_server_map[req_id]))
        if msg.is_finished:
            self.request_server_map.pop(req_id)

    async def recv(self):
        return None

    def drop_request(self, request_id: int) -> None:
        self.request_server_map.pop(request_id)

    def add_request(self, request_id: str, server_info: ServerInfo) -> None:
        self.request_server_map[request_id] = server_info

    def stop(self):
        pass

class AsyncLLMEngineLlumnixMixin:
    # pylint: disable=unused-argument
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 ) -> None:
        self.instance_id = instance_id

        self.state = EngineState.INIT
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))

        self.placement_group = placement_group
        self.request_output_queue_type = request_output_queue_type

    @property
    def instance_info(self) -> InstanceInfo:
        return self._scheduler.llumnix_metrics.to_instance_info()

    def start(self, loop: asyncio.AbstractEventLoop):
        super().start(loop)
        self._client = self.init_client_from_engine()
        self.trans_wrapper: AsyncBackQueueWrapper = AsyncBackQueueWrapper(self.placement_group,
                                                                          self.instance_id,
                                                                          self.request_output_queue_type)
        self._scheduler.llumnix_metrics.engine_init_metrics(self)

    async def update_callback(self, resp_list, step_requests):
        await super().update_callback(resp_list, step_requests)
        self._scheduler.llumnix_metrics.engine_step_metrics(self._scheduler)

    async def _loop(self):
        previous_state = self.state
        self.state = EngineState.RUNNING
        logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        try:
            await super()._loop()
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))

            previous_state = self.state
            self.state = EngineState.CRASHED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        if self.state == EngineState.RUNNING:
            self.state = EngineState.STOPPED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, EngineState.RUNNING, self.state))

    async def _handle_dropped_request(self):
        if self._dropped_req:
            for req_id in self._dropped_req:
                self.trans_wrapper.drop_request(req_id)
        await super()._handle_dropped_request()

    async def _handle_abort(self, abort: Optional[List[Tuple[int, int, str]]] = None):
        if abort is not None and len(abort) > 0:
            for req_id, _, _ in abort:
                self.trans_wrapper.drop_request(req_id)
        await super()._handle_abort(abort)

    async def add_request(self, server_info: ServerInfo, server_request: ServerRequest):
        logger.debug("engine {} add request {}".format(self.instance_id, server_request))
        self.trans_wrapper.add_request(server_request.id, server_info)
        # pylint: disable=protected-access
        await self._client._add_request(server_request)

    async def drop_request(self, req_id: int):
        await self._client.drop_request(req_id)

class AsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, AsyncLLMEngine):
    def __init__(self,
                instance_id: str,
                placement_group: PlacementGroup,
                request_output_queue_type: QueueType,
                migration_config: MigrationConfig,
                *args, **kwargs,
                ) -> None:
        AsyncLLMEngine.__init__(self, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(self, instance_id, placement_group, request_output_queue_type, migration_config)

class PrefillAsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, PrefillAsyncLLMEngine):
    def __init__(self,
            instance_id: str,
            placement_group: PlacementGroup,
            request_output_queue_type: QueueType,
            migration_config: MigrationConfig,
            *args, **kwargs,
            ) -> None:
        PrefillAsyncLLMEngine.__init__(self, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(self, instance_id, placement_group, request_output_queue_type, migration_config)

class DecodeAsyncLLMEngineLlumnix(AsyncLLMEngineLlumnixMixin, DecodeAsyncLLMEngine):
    def __init__(self,
            instance_id: str,
            placement_group: PlacementGroup,
            request_output_queue_type: QueueType,
            migration_config: MigrationConfig,
            *args, **kwargs,
            ) -> None:
        DecodeAsyncLLMEngine.__init__(self, *args, **kwargs)
        AsyncLLMEngineLlumnixMixin.__init__(self, instance_id, placement_group, request_output_queue_type, migration_config)

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

        self._loop = asyncio.new_event_loop()
        self._engine_ready = threading.Event()
        self._thread = threading.Thread(target=self._start_loop, args=(self._loop,), daemon=True, name="async_engine")
        self._thread.start()
        self._engine_ready.wait()

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

    def _start_loop(self, loop):
        asyncio.set_event_loop(loop)
        self.engine.start(loop)
        self._engine_ready.set()
        loop.run_forever()

    def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        assert "server_request" in kwargs and kwargs["server_request"]
        server_request = ServerRequest(**json.loads(kwargs["server_request"]))
        asyncio.run_coroutine_threadsafe(self.engine.add_request(server_info, server_request), self._loop)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for req_id in request_ids:
            self.engine.drop_request(int(req_id))

    def get_request_incremental_blocks(self, backend_request: LlumnixRequest, pre_stage_num_blocks: int) -> List[int]:
        pass

    def get_running_queue(self) -> Deque[LlumnixRequest]:
        pass

    def get_waiting_queue(self) -> Deque[LlumnixRequest]:
        pass

    def remove_running_request(self, request_id: str) -> bool:
        pass

    def remove_waiting_request(self, request_id: str) -> bool:
        pass

    def add_migrating_out_request_last_stage(self, backend_request: LlumnixRequest) -> None:
        pass

    def remove_migrating_out_request_last_stage(self, backend_request: LlumnixRequest) -> None:
        pass

    def pop_migrating_out_requests_last_stage(self) -> List[LlumnixRequest]:
        pass

    def pre_alloc(self,
                  request_id: str,
                  request_status: RequestStatus,
                  request_arrival_time: float,
                  block_num: int) -> List[int]:
        pass

    def add_running_request(self, backend_request: LlumnixRequest) -> None:
        pass

    def add_waiting_request(self, backend_request: LlumnixRequest) -> None:
        pass

    def free_dst_pre_alloc_cache(self, request_id: str = None) -> None:
        pass

    def free_src_request(self, backend_request: LlumnixRequest) -> None:
        pass

    async def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]):
        pass

    def commit_dst_request(self, backend_request: LlumnixRequest) -> None:
        pass

    def get_all_request_ids(self) -> List[str]:
        pass
