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
import gc
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from llumnix.backends.bladellm.proto.migration_worker_pb2 import MigrateRequests

from blade_llm.service.engine import AsyncLLMEngine
from blade_llm.service.args import ServingArgs
from blade_llm.utils.counter import Counter
from blade_llm.service.clients.base_client import AsyncRespStreamer
from blade_llm.protocol import GenerateStreamResponse, LogitsProcessorParams
from blade_llm.service.schedulers.base_scheduler import BaseScheduler
from llumnix.backends.bladellm.scheduler import get_scheduler
from blade_llm.service.workers.worker_client import worker_client_main
from blade_llm.model.tokenizer_utils import load_tokenizer
from blade_llm.service.scheduler_types import SchedulerInitInfo, SchedulerStepOutput
from blade_llm.service.metric import MetricMixin, scheduler_status_helper
from blade_llm.service.clients import GeneralLLMClient
from blade_llm.service.engine import AsyncLLMEngineClient
from blade_llm.protocol import ServerRequest

from blade_llm.protocol import (
    GenerateStreamResponse,
    SamplingParams,
    ServerRequest,
    StoppingCriteria,
)
from llumnix.backends.bladellm.queue import AsyncPutQueueActor
from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.backends.bladellm.scheduler import PagedSchedulerLlumnix
from llumnix.backends.bladellm.sequence import ServerRequestLlumnix, GenerationGroupStateLlumnix, GenerateStreamResponseLlumnix
from llumnix.backends.profiling import LatencyMemData
from llumnix.llumlet.request import LlumnixRequest
from llumnix.internal_config import MigrationConfig
from llumnix.server_info import ServerInfo
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.queue.utils import QueueType, init_output_queue_client
from llumnix.backends.bladellm.utils import string_to_int, get_model_conf
from llumnix.backends.bladellm.worker import launch_worker


logger = init_logger(__name__)

class LLMEngineLlumnix(AsyncLLMEngine):
    def __init__(self,
                 instance_id: str,
                 output_queue_type: QueueType,
                 migration_config: MigrationConfig,
                 placement_group: Optional[PlacementGroup],
                 node_id: Optional[str],
                 *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        engine_instance_id = string_to_int(instance_id)
        self._worker_processes = launch_worker(self._args, engine_instance_id, migration_config)
        # TODO[xinyi]: maybe inherit from Metirc Class in Bladellm or create a Metric Class for Llumnix
        # TODO[xinyi]: support PipelineLLMMixin Class for Bladellm
        self.step_request_queue = asyncio.Queue()
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        self.state = EngineState.INIT
        self.is_warmup = False
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))
        # TODO[xinyi]: repeated code in llm_engine for both bladellm and vllm. Simplify it.
        # Place the async put queue actor together with the instance.
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
        try:
            self.async_put_queue_actor = ray.remote(
                num_cpus=1,
                num_gpus=1,
                scheduling_strategy=scheduling_strategy
            )(AsyncPutQueueActor).remote(instance_id, output_queue_type)
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
        self.put_queue_loop_thread.start()
        # TODO[xinyi]: support scheduler_status_helper

    # TODO[xinyi]: delete this function, now just for test
    async def _init(self):
        print("init...")
        # NOTE(zycao): Keep Hybrid DP mode as not implemented status before all parts are ready.
        if self._args.enable_hybrid_dp:
            raise NotImplementedError("We will enable Hybrid DP for MoE models later.")
        # NOTE(junqi): Disable gc in the begin of process might affect ut
        gc.disable()
        self._worker_processes.start()
        self.semaphore = asyncio.BoundedSemaphore(2)
        self.lock = asyncio.Lock()
        self.worker_timer_queue = asyncio.Queue()

        # NB(xiafei.qiuxf): Lazy init of aio-related objects, this function MUST be call on the exact
        # thread and event loop with `step` method. Because we use default event loop for the following
        # objects.
        self._stop_event = asyncio.Event()
        self._req_buffer = asyncio.Queue()
        client_args = (
            self._args,
            self._worker_processes.worker_addrs if self._args.enable_remote_worker else None,
        )

        self._workers = worker_client_main(self.pp_enabled, self._args, client_args)
        await self._workers.wait_backend_ready()

        token_capacity, block_size, model_max_len, cpu_blocks = await self._workers.estimate_token_blocks_capacity()
        self._tokenizer = load_tokenizer(
            self._args.load_model_options.tokenizer_dir,
            self._args.load_model_options.special_token_dict,
        )
        self._scheduler: BaseScheduler = get_scheduler(
            self._args,
            self._tokenizer,
            SchedulerInitInfo(
                token_capacity=token_capacity,
                block_size=block_size,
                model_max_len=model_max_len,
                cpu_blocks=cpu_blocks,
            ),
            self._model_conf,
        )

        # profiling-related
        # 
        #  fields
        self._sch_status_helper = scheduler_status_helper(self._scheduler)
        # this will take control of exporters if step-wise tracing will trace from the beginning
        self.engine_pre_step_metrics()
        client = AsyncLLMEngineClient(
            self.pp_enabled,
            self._req_buffer,
            self._dropped_req,
            self._back_queue,
            self._scheduler,
        )
        self._client = GeneralLLMClient(self._args, client, self._model_conf)

        # TODO(xinyi): keep this line after test
        # await super()._init()
        self._scheduler.add_update_instance_info_callback(self.update_instance_info)

    # TODO(xinyi): import process_model_outpputs for warmup next step
    def _warmup_llumnix_request(self, prefill_request: ServerRequest):
        self.is_warmup = True
        server_request = ServerRequestLlumnix(prefill_request, prefill_request.id, None, 1000)
        return server_request

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

    def process_model_outputs(self, update_output):
        try:
            server_infos = []
            request_outputs = []
            request_groups_map = self._scheduler.get_request_groups_map()
            running_requests = [gen_group.request_group_id for gen_group in self._scheduler.running]
            for req_id, l_resp in update_output.response.items():
                if req_id in request_groups_map:
                    request_groups_map[req_id].update_num_computed_tokens(request_groups_map[req_id].token_chunk_size)
                    if req_id in running_requests and l_resp:
                        server_info = request_groups_map[req_id].server_info
                        server_infos.append(server_info)
                        request_outputs.append(GenerateStreamResponseLlumnix(l_resp, req_id))
                        if l_resp.is_finished:
                            # TODO[xinyi]: handle error info in back queue
                            del self._back_queue[req_id]
            # TODO[xinyi]: design metric
            for request_output in request_outputs:
                if hasattr(request_output, 'request_timestamps'):
            #         request_output.request_timestamps.engine_step_timestamp_begin = step_begin_time
            #         request_output.request_timestamps.engine_step_timestamp_end = step_end_time
                    request_output.request_timestamps.engine_step_postprocess_timestamp_end = time.time()
            if request_outputs:
                self.put_queue_args_queue.put_nowait((request_outputs, server_infos))
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))


    async def step(self):
        # TODO[xinyi]: async step may lead to put info not in order for both put_queue_args_queue and step_request_queue
        try:
            step_begin_time = time.time()
            await super().step()
            step_end_time = time.time()

            instance_info: InstanceInfo = self.instance_info
            if instance_info:
                instance_info.instance_id = self.instance_id
                instance_info.step_id = next(self.step_counter)
                instance_info.timestamp = time.time()
                instance_info.profiling_data=(instance_info.inference_type.value,
                                            instance_info.num_seqs,
                                            sum(instance_info.running_seq_lens))
                # TODO(xinyi): need this metric?
                                        #   self.model_executor.last_inference_latency) todo?
            gen_groups = self._scheduler.running
            if gen_groups:
                tot_blocks = []
                for req_state in gen_groups[-1].paged_reqs:
                    blocks = self._scheduler.block_manager.get_block_table(req_state)
                    tot_blocks.extend(blocks)
                tot_blocks = set(tot_blocks)
                instance_info.num_blocks_last_running_request = len(tot_blocks)
            self.instance_info=instance_info
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))

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
        output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: ServingArgs,
        placement_group: PlacementGroup = None,
        node_id: str = None
    ) -> None:
        try:
            self.engine: LLMEngineLlumnix = LLMEngineLlumnix(instance_id, output_queue_type, migration_config, placement_group,
                                                            node_id,
                                                            engine_args)
            self.instance_id = instance_id
            self.worker_addrs_list = [f"unix://{engine_args.worker_socket_path}.{i}" for i in range(engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size)]
            self._loop = asyncio.new_event_loop()#asyncio.new_event_loop()##
            self._engine_ready = threading.Event()

            def _start_loop(loop):
                try:
                    asyncio.set_event_loop(loop)
                    self.engine.start(loop)
                    self._engine_ready.set()  # inform readiness of async engine.
                    loop.run_forever()
                except Exception as e:
                    logger.error("Error in engine loop: {}".format(e))
                    logger.error("exception traceback: {}".format(traceback.format_exc())) 
            self._thread = threading.Thread(target=_start_loop, args=(self._loop,), daemon=True, name="async_engine")
            self._thread.start()
            self._engine_ready.wait()  # wait till wrapped async engine is ready.
            self.engine.is_warmup = False
        except Exception as e:
            logger.error("Error in engine loop: {}".format(e))
            logger.error("exception traceback: {}".format(traceback.format_exc()))
        # TODO[xinyi, kuilong]: add function methods in worker for llumnix
        # self.engine._workers.init_migration()
    
    def add_request(self,
                    request_id: str,
                    server_info: ServerInfo,
                    expected_steps: int,
                    server_request: ServerRequest) -> None:
        # Store the server information of each request to put the request outputs back to the corresponding api server correctly.
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
        
    def commit_dst_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        seq = backend_request.paged_reqs[0]
        # seq.seq_id = next(self.engine.seq_counter) # TODO(xinyi): whether it is no need to change seq_id
        logger.info("add seq {} to block table".format(seq.request_id))
        pre_alloc_blocks = self.engine._scheduler.pre_alloc_cache_dict.pop(backend_request.request_id)
        self.engine._scheduler.block_manager.add_block_table(pre_alloc_blocks, seq.request_id)
        backend_request.reset_migration_args()
        self.add_running_request(backend_request)

    def _run_workers(self, worker_method, *args, **kwargs):
        # pylint: disable=protected-access
        executor = getattr(self.engine._workers, worker_method)
        return executor(*args, **kwargs)

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
