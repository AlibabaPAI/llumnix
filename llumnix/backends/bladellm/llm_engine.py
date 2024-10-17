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
from typing import Any, List, Optional, Dict, Union, Iterable, Tuple
from collections import defaultdict
import threading
import asyncio
import queue
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy

from blade_llm.service.engine import AsyncLLMEngine
from blade_llm.service.args import ServingArgs
from blade_llm.utils.counter import Counter
from blade_llm.protocol import GenerateStreamResponse, LogitsProcessorParams


from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.backends.bladellm.scheduler import SchedulerLlumnix
from llumnix.backends.bladellm.sequence import GenerationGroupStateLlumnix, ServerRequestLlumnix
from llumnix.llumlet.request import LlumnixRequest
from llumnix.internal_config import MigrationConfig
from llumnix.server_info import ServerInfo
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.queue.utils import get_output_queue_client, QueueType

logger = init_logger(__name__)

#TODO[xinyi]: the same to the function in vllm, maybe put into utils/common.py ?
class AsyncPutQueueActor:
    def __init__(self, instance_id, output_queue_type: QueueType):
        self.instance_id = instance_id
        self.output_queue_type = output_queue_type
        self.request_output_queue_client: QueueClientBase = get_output_queue_client(output_queue_type)
        self.engine_actor_handle = None

    async def put_nowait_to_servers(self,
                                    server_request_outputs: Dict[str, List[GenerateStreamResponse]],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
        try:
            if self.engine_actor_handle is None:
                self.engine_actor_handle = ray.get_actor("instance_{}".format(self.instance_id), namespace="llumnix")
            tasks = []
            for server_id, req_outputs in server_request_outputs.items():
                server_info = server_info_dict[server_id]
                for req_output in req_outputs:
                    if hasattr(req_output, 'request_timestamps'):
                        req_output.request_timestamps.engine_actor_put_queue_timestamp = time.time()
                tasks.append(asyncio.create_task(self.request_output_queue_client.put_nowait(req_outputs, server_info)))
            rets = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, ret in enumerate(rets):
                if isinstance(ret, (TimeoutError, ray.exceptions.RayActorError)):
                    server_id = list(server_request_outputs.keys())[idx]
                    server_info = server_info_dict[server_id]
                    logger.info("Server {} is dead".format(server_id))
                    if self.output_queue_type == QueueType.ZMQ:
                        logger.info("request output queue ip: {}, port: {}".format(server_info.request_output_queue_ip,
                                                                                server_info.request_output_queue_port))
                    req_outputs = list(server_request_outputs.values())[idx]
                    request_ids = [req_output.request_id for req_output in req_outputs]
                    self.engine_actor_handle.abort_request.remote(request_ids)
        # pylint: disable=W0703
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))

class LLMEngineLlumnix(AsyncLLMEngine):
    def __init__(self, instance_id: str, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        # TODO(s5u13b): Reduce the overhead.
        self.async_put_queue_thread = AsyncPutQueueThread(instance_id)
        self.async_put_queue_thread.start()
        self.request_server_info: Dict[str, ServerInfo] = {}
        self.put_queue_args_queue = queue.Queue()
        self.put_queue_loop_thread = threading.Thread(
            target=self._start_put_queue_loop, args=(), daemon=True, name="put_queue_loop"
        )
        self.async_put_queue_actor = ray.remote(
            num_cpus=0,
            scheduling_strategy=scheduling_strategy
        )(AsyncPutQueueActor).remote(instance_id, output_queue_type)
        self.put_queue_loop_thread.start()

    
    async def _init(self):
        self._worker_processes.start()

        self._stop_event = asyncio.Event()
        self._req_buffer = asyncio.Queue()
        client_args = (
            (self._args, self._worker_processes.worker_addrs)
            if self._args.enable_remote_worker
            else (self._args, [], self._inst_id)
        )
        logger.info("client_args {}".format(client_args))
        # pylint: disable=protected-access
        self._workers = LlumnixAioWorkerClient(*client_args)
        await self._workers.wait_backend_ready()
        token_capacity, block_size, model_max_len, cpu_blocks = (
            await self._workers.estimate_token_capacity()
        )
        logger.info(
            "Workers estimate token capacity to: {}, cpu_blocks: {}, block_size: {}".format(
                token_capacity,
                cpu_blocks,
                block_size,
            )
        )
        self._scheduler = SchedulerLlumnix(
            self._args,
            SchedulerInitInfo(
                token_capacity=token_capacity,
                block_size=block_size,
                model_max_len=model_max_len,
                cpu_blocks=cpu_blocks,
            ),
        )

        # profiling-related fields
        self._sch_status_helper = scheduler_status_helper(self._scheduler)
        # this will take control of exporters if step-wise tracing will trace from the beginning
        self.engine_pre_step_metrics()
    
    async def _loop(self):
        previous_state = self.state
        self.state = EngineState.RUNNING
        logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        try:
            await super()._loop()
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
            self._run_workers("shutdown")

            previous_state = self.state
            self.state = EngineState.CRASHED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        if self.state == EngineState.RUNNING:
            self.state = EngineState.STOPPED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, EngineState.RUNNING, self.state))

    async def step(self):
        # TODO[xinyi]: add step time metrics
        step_begin_time = time.time()
        await super().step()

        instance_info: InstanceInfo = self.instance_info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.profiling_data=(instance_info.inference_type.value,
                                      instance_info.num_seqs,
                                      sum(instance_info.running_seq_lens),
                                      self.model_executor.last_inference_latency)
        gen_groups = self.scheduler.running
        if gen_groups:
            tot_blocks = []
            for req_state in gen_groups[-1].paged_reqs:
                blocks = self.scheduler.block_manager.get_block_table(req_state)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_blocks_last_running_request = len(tot_blocks)
        self.instance_info=instance_info

    async def update_callback(self, resp):
        server_infos = []
        request_outputs = []

        # metrics
        worker_timer, step_request = self.worker_timer_queue.get_nowait()
        worker_timer.done()

        update_output = self._scheduler.update(resp)
        # handle update ouput
        if update_output.reset:
            await self._handle_reset()
        elif update_output.response is not None:
            for req_id, l_resp in update_output.response.items():
                if req_id in self._scheduler.request_groups_map and l_resp:
                    request_outputs.append(l_resp)
                    server_infos.append(self._scheduler.request_groups_map[req_id].server_info)
                    if l_resp.is_finished:
                        del self._back_queue[req_id]
        self.put_queue_args_queue.put_nowait((request_outputs, server_infos))
        self.worker_post_step_metrics(step_request, resp)
        self.semaphore.release()

    #TODO[xinyi]: the same to the function in vllm, maybe put into utils.py
    def update_instance_info(self, instance_info: InstanceInfo) -> None:
        # These fields are updated after step.
        if self.instance_info is not None:
            instance_info.instance_id = self.instance_info.instance_id
            instance_info.step_id = self.instance_info.step_id
            instance_info.timestamp = self.instance_info.timestamp
            instance_info.profiling_data = self.instance_info.profiling_data
            instance_info.num_blocks_last_running_request = self.instance_info.num_blocks_last_running_request
        self.instance_info = instance_info
    
    async def add_request(self, *args, **kwargs):
        # TODO[xinyi]: next step split webserver
        await self._client.add_request(*args, **kwargs)
    
    #TODO[xinyi]: the same to the function in vllm, maybe put into utils/common.py
    def _start_put_queue_loop(self):
        while True:
            args = self.put_queue_args_queue.get()
            request_outputs, server_infos = args
            for request_output in request_outputs:
                if hasattr(request_output, 'request_timestamps'):
                    request_output.request_timestamps.engine_thread_put_queue_timestamp = time.time()
            self._put_request_outputs_to_server(request_outputs, server_infos)
    
    def _put_request_outputs_to_server(self, request_outputs: List[GenerateStreamResponse], server_infos: List[ServerInfo]) -> None:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}
        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output, server_info in zip(request_outputs, server_infos):
            server_id = server_info.server_id
            server_request_outputs[server_id].append(request_output)
            if server_id not in server_info_dict:
                server_info_dict[server_id] = server_info
        # TODO(s5u13b): Reduce the across-actor overhead.
        self.async_put_queue_actor.put_nowait_to_servers.remote(server_request_outputs, server_info_dict)

class BackendBladeLLM(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        migration_config: MigrationConfig,
        engine_args: ServingArgs,
        placement_group: PlacementGroup = None,
        node_id: str = None
    ) -> None:
        self.engine: LLMEngineLlumnix = LLMEngineLlumnix(instance_id, engine_args)
        self.engine.scheduler = SchedulerLlumnix()#self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config)
        self.engine.scheduler.add_update_instance_info_callback(self.engine.update_instance_info)

        self.instance_id = instance_id
        loop = asyncio.get_event_loop()
        self.engine.start(loop)
        self.state = EngineState.INIT
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))
        
        loop = asyncio.get_event_loop()
        self.engine.start(loop)

    
    def add_request(self,
                    request_id: str,
                    server_info: ServerInfo,
                    expected_steps: int,
                    *args,
                    **kwargs) -> None:
        # Store the server information of each request to put the request outputs back to the corresponding api server correctly.
        self.engine.add_request(ServerRequestLlumnix(request_id, server_info, expected_steps, *args, **kwargs))

    def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]) -> None:
        ray.get(dst_ray_actor.execute_engine_method.remote("_run_workers",
                                                           "migrate_cache",
                                                           dst_blocks=dst_blocks,
                                                           src_blocks=src_blocks,
                                                           src_worker_handle_list=self.worker_handle_list))

    def is_ready(self):
        return True
    
    
    
    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for req_id in request_ids:
            self.engine.drop_request(req_id) # TODO
    
    def get_running_queue(self ) -> List[SequenceGroupLlumnix]:
        return self.engine.scheduler.get_running_queue()

    def get_request_incremental_blocks(self, *args, **kwargs) -> List[int]:
        return self.engine.scheduler.get_request_incremental_blocks(*args, **kwargs)

    def remove_running_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler.remove_running_request(*args, **kwargs)

    def add_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler.add_migrating_out_request_last_stage(*args, **kwargs)

    def remove_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler.remove_migrating_out_request_last_stage(*args, **kwargs)

    def pop_migrating_out_requests_last_stage(self, *args, **kwargs) -> List[Any]:
        return self.engine.scheduler.pop_migrating_out_requests_last_stage(*args, **kwargs)

    def pre_alloc(self, *args, **kwargs) -> List[int]:
        return self.engine.scheduler.pre_alloc(*args, **kwargs)

    def should_abort_migration(self, *args, **kwargs) -> bool:
        return self.engine.scheduler.should_abort_migration(*args, **kwargs)

    def add_running_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler.add_running_request(*args, **kwargs)

    def is_request_running(self, *args, **kwargs) -> bool:
        return self.engine.scheduler.is_request_running(*args, **kwargs)

    def free_dst_pre_alloc_cache(self, *args, **kwargs) -> None:
        return self.engine.scheduler.free_dst_pre_alloc_cache(*args, **kwargs)

    def free_src_request(self, backend_request: LlumnixRequest) -> None:
        return self.engine.scheduler.free_src_request(backend_request)

    def get_all_request_ids(self) -> List[str]:
        return self.engine.scheduler.get_all_request_ids()
    
    def get_request_server_info(self, request_id: str) -> ServerInfo:
        return self.engine.request_server_info[request_id]

    