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
from typing import Any, List, Optional, Union, Iterable, Deque, Tuple
from collections import defaultdict
import threading
import asyncio
import queue
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.outputs import RequestOutput, RequestOutputFactory, EmbeddingRequestOutput
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import Counter
from vllm.usage.usage_lib import UsageContext

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.backends.vllm.scheduler import SchedulerLlumnix
from llumnix.backends.vllm.sequence import SequenceGroupLlumnix, RequestStatus
from llumnix.backends.profiling import LatencyMemData
from llumnix.server_info import ServerInfo
from llumnix.internal_config import MigrationConfig
from llumnix.queue.utils import QueueType
from llumnix.backends.utils import AsyncPutQueueActor
from llumnix.llumlet.request import LlumnixRequest
from llumnix.utils import get_instance_name

logger = init_logger(__name__)

NO_OUTPUTS_STEP_INTERVAL = 0.01


class LlumnixRequestOutputFactory(RequestOutputFactory):
    @staticmethod
    def create(seq_group: SequenceGroupLlumnix, use_cache: bool = False):
        # Determine the type based on a condition, for example:
        if hasattr(seq_group,
                   'embeddings') and seq_group.embeddings is not None:
            return EmbeddingRequestOutput.from_seq_group(seq_group), seq_group.server_info
        if RequestStatus.is_migrating(seq_group.status):
            return None
        # pylint: disable=too-many-function-args
        return RequestOutput.from_seq_group(seq_group, use_cache), seq_group.server_info

class LLMEngineLlumnix(_AsyncLLMEngine):
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 *arg, **kwargs) -> None:
        # pylint: disable=import-outside-toplevel
        import vllm.outputs
        vllm.outputs.RequestOutputFactory.create = LlumnixRequestOutputFactory.create
        super().__init__(*arg, **kwargs)
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        # Place the async put queue actor together with the instance.
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )
        self.put_queue_args_queue = queue.Queue()
        self.put_queue_loop_thread = threading.Thread(
            target=self._start_put_queue_loop, args=(), daemon=True, name="put_queue_loop"
        )
        self.async_put_queue_actor = ray.remote(
            num_cpus=1,
            scheduling_strategy=scheduling_strategy
        )(AsyncPutQueueActor).remote(instance_id, request_output_queue_type)
        self.put_queue_loop_thread.start()

    # pylint: disable=W0221
    @classmethod
    def from_engine_args(
        cls,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: EngineArgs,
        latency_mem: Optional[LatencyMemData] = None,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT
    ) -> "LLMEngineLlumnix":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        # Hack to pass placement_group for init workers.
        engine_config.parallel_config.placement_group = placement_group
        # Initialize the cluster and specify the executor class.
        # pylint: disable=import-outside-toplevel
        if latency_mem is not None:
            from llumnix.backends.vllm.executor import SimGPUExecutor
            executor_class = SimGPUExecutor
            executor_class.latency_mem = latency_mem
        elif engine_config.parallel_config.use_ray:
            from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor
            executor_class = LlumnixRayGPUExecutor
            executor_class.migration_config = migration_config
        else:
            raise ValueError('Unsupported executor backend')
        # Create the LLM engine.
        engine = cls(
            instance_id=instance_id,
            placement_group=placement_group,
            request_output_queue_type=request_output_queue_type,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )
        return engine

    def _process_request_outputs(
            self,
            outputs: List[Tuple[RequestOutput,ServerInfo]],
            step_begin_time: float
    ) -> Tuple[List[RequestOutput], List[ServerInfo]]:
        request_outputs = []
        server_infos = []
        if outputs:
            request_outputs, server_infos = zip(*outputs)
            request_outputs = list(request_outputs)
            server_infos = list(server_infos)
        for request_output, server_info in zip(request_outputs, server_infos):
            if hasattr(server_info, 'request_timestamps'):
                request_output.request_timestamps = server_info.request_timestamps
                request_output.request_timestamps.engine_process_model_outputs_timestamp_end = time.time()
            if request_output.finished:
                logger.info("engine finished request {}".format(request_output.request_id))
        for server_info in server_infos:
            if hasattr(server_info, 'request_timestamps'):
                server_info.request_timestamps.engine_process_model_outputs_timestamp_begin = time.time()
        for request_output in request_outputs:
            if hasattr(request_output, 'request_timestamps'):
                request_output.request_timestamps.engine_step_timestamp_begin = step_begin_time
                request_output.request_timestamps.engine_step_timestamp_end = time.time()

        instance_info: InstanceInfo = self.instance_info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        # TODO(KuilongCui): add cli_args to determine whether to collect profiling data
        instance_info.profiling_data=(instance_info.inference_type.value if instance_info.inference_type else "",
                                      instance_info.num_seqs,
                                      sum(instance_info.running_seq_lens),
                                      self.model_executor.last_inference_latency)
        seq_groups = self.scheduler[0].running
        if seq_groups:
            tot_blocks = []
            for seq in seq_groups[-1].get_seqs(SequenceStatus.RUNNING):
                blocks = self.scheduler[0].block_manager.get_block_table(seq)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_blocks_last_running_request = len(tot_blocks)

        self.instance_info = instance_info

        if request_outputs:
            self.put_queue_args_queue.put_nowait((request_outputs, server_infos))
        for request_output in request_outputs:
            if hasattr(request_output, 'request_timestamps'):
                request_output.request_timestamps.engine_step_postprocess_timestamp_end = time.time()

        return request_outputs, server_infos

    async def step_async(self) -> Tuple[List[RequestOutput], List[ServerInfo]]:
        step_begin_time = time.time()
        # pylint: disable=too-many-function-args
        outputs = await super().step_async(0)
        return self._process_request_outputs(outputs, step_begin_time)

    def update_instance_info(self, instance_info: InstanceInfo) -> None:
        # These fields are updated after step.
        if self.instance_info is not None:
            instance_info.instance_id = self.instance_info.instance_id
            instance_info.step_id = self.instance_info.step_id
            instance_info.timestamp = self.instance_info.timestamp
            instance_info.profiling_data = self.instance_info.profiling_data
            instance_info.num_blocks_last_running_request = self.instance_info.num_blocks_last_running_request
        self.instance_info = instance_info

    # pylint: disable=invalid-overridden-method
    async def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs):
        super().add_request(request_id, *args, **kwargs)
        seq_group = self.scheduler[0].waiting[-1]
        if hasattr(server_info, 'request_timestamps'):
            server_info.request_timestamps.engine_add_request_timestamp = time.time()
        self.scheduler[0].waiting[-1] = SequenceGroupLlumnix(request_id, server_info, expected_steps, [seq_group.get_seqs()[0]],
                                                          seq_group.metrics.arrival_time, seq_group.sampling_params, seq_group.lora_request,
                                                          seq_group.trace_headers, seq_group.prompt_adapter_request, seq_group.encoder_seq,
                                                          seq_group.priority)

    def _start_put_queue_loop(self):
        while True:
            args = self.put_queue_args_queue.get()
            request_outputs, server_infos = args
            for request_output in request_outputs:
                if hasattr(request_output, 'request_timestamps'):
                    request_output.request_timestamps.engine_thread_put_queue_timestamp = time.time()
            self._put_request_outputs_to_server(request_outputs, server_infos)

    def _put_request_outputs_to_server(self, request_outputs: List[RequestOutput], server_infos: List[ServerInfo]) -> None:
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

class BackendVLLM(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: EngineArgs,
    ) -> None:
        self.engine: LLMEngineLlumnix = LLMEngineLlumnix.from_engine_args(engine_args=engine_args,
                                                                          request_output_queue_type=request_output_queue_type,
                                                                          migration_config=migration_config,
                                                                          instance_id=instance_id,
                                                                          placement_group=placement_group)
        self.engine.scheduler = [SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config)
                                 for _ in range(engine_args.pipeline_parallel_size)]
        for vid in range(engine_args.pipeline_parallel_size):
            self.engine.scheduler[vid].add_update_instance_info_callback(self.engine.update_instance_info)
        self.engine.output_processor.scheduler = self.engine.scheduler
        self.migration_config = migration_config
        self.instance_id = instance_id
        self.worker_handle_list = self.engine.model_executor.workers.copy()
        if len(self.worker_handle_list) + 1 == self.engine.parallel_config.world_size:
            self.worker_handle_list.insert(0, ray.get_actor(get_instance_name(self.instance_id), namespace="llumnix"))
        self._run_workers("init_migration", instance_id=instance_id,
                                            migration_config=migration_config,
                                            src_worker_handle_list=self.worker_handle_list,
                                            placement_group=placement_group)

        self.state = EngineState.INIT
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))

        self._stop_event = asyncio.Event()
        asyncio.create_task(self._start_engine_step_loop())

    async def _start_engine_step_loop(self) -> None:
        self._stop_event.clear()

        previous_state = self.state
        self.state = EngineState.RUNNING
        logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        while not self._stop_event.is_set():
            try:
                request_outputs, _ = await self.engine.step_async()
                if len(request_outputs) == 0:
                    await asyncio.sleep(NO_OUTPUTS_STEP_INTERVAL)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Error in engine loop: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))
                self._run_workers("shutdown")

                previous_state = self.state
                self.state = EngineState.CRASHED
                logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))
                break

        if self.state == EngineState.RUNNING:
            self.state = EngineState.STOPPED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, EngineState.RUNNING, self.state))

    async def is_ready(self) -> bool:
        return True

    def execute_worker_method(self, method, *args, **kwargs):
        return self.engine.model_executor.driver_worker.execute_method(method, *args, **kwargs)

    # Store the server information of each request to put the request outputs back to the corresponding api server correctly.
    async def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        await self.engine.add_request(request_id, server_info, expected_steps, *args, **kwargs)

    def commit_dst_request(self, backend_request: SequenceGroupLlumnix) -> None:
        seq = backend_request.get_seqs()[0]
        seq.seq_id = next(self.engine.seq_counter)
        logger.info("pop request {} from pre_alloc_cache_dict".format(backend_request.request_id))
        pre_alloc_blocks = self.engine.scheduler[0].pre_alloc_cache_dict.pop(backend_request.request_id)
        self.engine.scheduler[0].block_manager.add_block_table(pre_alloc_blocks, seq.seq_id)
        backend_request.reset_migration_args_dst()
        assert RequestStatus.is_migrating(backend_request.status), \
            "The status of request migrated to dst instance should be  \
             RequestStatus.WAITING_MIGRATING or RequestStatus.RUNNING_MIGRATING"
        if backend_request.status == RequestStatus.WAITING_MIGRATING:
            self.add_waiting_request(backend_request)
        else: # RUNNING_MIGRATING:
            backend_request.reset_status()
            self.add_running_request(backend_request)

    async def send_blocks(self, dst_ray_actor: ray.actor.ActorHandle, request_id: int,
                          src_blocks: List[int], dst_blocks: List[int], has_more: bool):
        await dst_ray_actor.execute_engine_method.remote("_run_workers",
                                                         "migrate_cache",
                                                         dst_blocks=dst_blocks,
                                                         src_blocks=src_blocks,
                                                         src_worker_handle_list=self.worker_handle_list)

    def _run_workers(self, *args, **kwargs):
        # pylint: disable=protected-access
        return self.engine.model_executor._run_workers(*args, **kwargs)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        return self.engine.abort_request(request_ids)

    def get_running_queue(self) -> List[SequenceGroupLlumnix]:
        return self.engine.scheduler[0].get_running_queue()

    def get_waiting_queue(self) -> Deque[SequenceGroupLlumnix]:
        return self.engine.scheduler[0].get_waiting_queue()

    async def get_request_incremental_blocks(self, backend_request: LlumnixRequest, pre_stage_num_blocks: int) -> Tuple[List[int], List[int]]:
        incremental_blocks, incremental_token_ids = \
            self.engine.scheduler[0].get_request_incremental_blocks(backend_request, pre_stage_num_blocks)
        is_last_stage = (len(incremental_blocks) <= self.migration_config.migration_last_stage_max_blocks) or backend_request.blocking_migration
        return incremental_blocks, incremental_token_ids, is_last_stage

    def remove_running_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].remove_running_request(*args, **kwargs)

    def remove_waiting_request(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].remove_waiting_request(*args, **kwargs)

    def add_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].add_migrating_out_request_last_stage(*args, **kwargs)

    def remove_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].remove_migrating_out_request_last_stage(*args, **kwargs)

    def pop_migrating_out_requests_last_stage(self, *args, **kwargs) -> List[Any]:
        return self.engine.scheduler[0].pop_migrating_out_requests_last_stage(*args, **kwargs)

    def pre_alloc(self, *args, **kwargs) -> List[int]:
        return self.engine.scheduler[0].pre_alloc(*args, **kwargs)

    def should_abort_migration(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].should_abort_migration(*args, **kwargs)

    def add_running_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].add_running_request(*args, **kwargs)

    def add_waiting_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].add_waiting_request(*args, **kwargs)

    def is_request_running(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].is_request_running(*args, **kwargs)

    def free_dst_pre_alloc_cache(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].free_dst_pre_alloc_cache(*args, **kwargs)

    def free_src_request(self, backend_request: SequenceGroup) -> None:
        return self.engine.scheduler[0].free_src_request(backend_request)

    def get_all_request_ids(self) -> List[str]:
        return self.engine.scheduler[0].get_all_request_ids()
