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
from typing import List, Optional, Union, Iterable, Deque, Tuple, Dict, Any, Coroutine
import asyncio
import queue
import gc
from collections import defaultdict

import ray
from ray.util.placement_group import PlacementGroup
import ray.actor

from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.outputs import RequestOutput, RequestOutputFactory, EmbeddingRequestOutput
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import Counter
from vllm.usage.usage_lib import UsageContext
from vllm.engine.llm_engine import SchedulerContext
from vllm import envs as vllm_envs

from llumnix.arg_utils import InstanceArgs, LlumnixEngineArgs
from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendBaseInterface, BackendMigrationInterface
from llumnix.backends.vllm.scheduler import SchedulerLlumnix
from llumnix.backends.vllm.sequence import SequenceGroupLlumnix, RequestStatus
from llumnix.backends.profiling import LatencyMemData
from llumnix.request_processing_context import RequestProcessingContext
from llumnix.internal_config import MigrationConfig
from llumnix.queue.queue_type import QueueType
from llumnix.backends.utils import EngineState
from llumnix.backends.output_forwarder import RequestOutputForwardingMode, OutputForwarder
from llumnix.ray_utils import LlumnixActor, get_llumnix_actor_handle
from llumnix.llumlet.request import LlumnixRequest
from llumnix.constants import NO_OUTPUTS_STEP_INTERVAL, RAY_RPC_TIMEOUT
from llumnix.utils import (
    make_async,
    BackendType,
    RequestIDType,
    MigrationResponse,
    asyncio_wait_for_ray_remote_call_with_timeout,
    InstanceContext,
)
from llumnix.request_output import LlumnixRequestOuput

logger = init_logger(__name__)


class LlumnixRequestOutputFactory(RequestOutputFactory):
    @staticmethod
    def create(seq_group: SequenceGroupLlumnix, use_cache: bool = False):
        # Determine the type based on a condition, for example:
        if hasattr(seq_group,
                   'embeddings') and seq_group.embeddings is not None:
            return EmbeddingRequestOutput.from_seq_group(seq_group), seq_group.request_processing_context
        # pylint: disable=too-many-function-args
        return RequestOutput.from_seq_group(seq_group, use_cache), seq_group.request_processing_context


class LLMEngineLlumnix(_AsyncLLMEngine):
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 disable_async_output_proc: bool,
                 backend_type: BackendType,
                 request_output_forwarding_mode: RequestOutputForwardingMode,
                 abort_request_callback: Coroutine,
                 *arg, **kwargs) -> None:
        # pylint: disable=import-outside-toplevel
        import vllm.outputs
        vllm.outputs.RequestOutputFactory.create = LlumnixRequestOutputFactory.create
        super().__init__(*arg, **kwargs)
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        self.output_forwarder = OutputForwarder(
            instance_id,
            request_output_queue_type,
            request_output_forwarding_mode,
            abort_request_callback,
            placement_group,
            backend_type,
        )
        self.disable_async_output_proc = disable_async_output_proc
        self.request_ids = set()

    # pylint: disable=W0221
    @classmethod
    def from_engine_args(
        cls,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: EngineArgs,
        backend_type: BackendType,
        request_output_forwarding_mode: RequestOutputForwardingMode,
        abort_request_callback: Coroutine,
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
            from llumnix.backends.vllm.sim_executor import SimGPUExecutor
            executor_class = SimGPUExecutor
            executor_class.latency_mem = latency_mem
        elif engine_config.parallel_config.use_ray:
            from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor
            executor_class = LlumnixRayGPUExecutor
            executor_class.migration_config = migration_config
            executor_class.instance_id = instance_id
        else:
            raise ValueError('Unsupported executor backend')
        # Create the LLM engine.
        engine = cls(
            instance_id=instance_id,
            placement_group=placement_group,
            request_output_queue_type=request_output_queue_type,
            disable_async_output_proc=engine_args.disable_async_output_proc,
            backend_type=backend_type,
            request_output_forwarding_mode=request_output_forwarding_mode,
            abort_request_callback=abort_request_callback,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )
        return engine

    # pylint: disable=inconsistent-return-statements
    def _process_model_outputs(self,
                               ctx: SchedulerContext,
                               request_id: Optional[str] = None) -> None:
        if len(ctx.output_queue) == 0:
            return None

        if request_id:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output, skip) = ctx.output_queue[0]
        else:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output, skip) = ctx.output_queue.popleft()

        # Filter out outputs of migrating requests.
        request_processing_contexts: List[RequestProcessingContext] = []
        if outputs:
            new_outputs = []
            new_scheduled_seq_groups = []
            new_seq_group_metadata_list = []
            for scheduled_seq_group, seq_group_meta, seq_group_output in \
                    zip(scheduler_outputs.scheduled_seq_groups, seq_group_metadata_list, outputs[0].outputs):
                seq_group = scheduled_seq_group.seq_group
                new_scheduled_seq_groups.append(scheduled_seq_group)
                new_seq_group_metadata_list.append(seq_group_meta)
                new_outputs.append(seq_group_output)
                request_processing_contexts.append(seq_group.request_processing_context)
            scheduler_outputs.scheduled_seq_groups = new_scheduled_seq_groups
            outputs[0].outputs = new_outputs
            seq_group_metadata_list = new_seq_group_metadata_list

        if request_id:
            ctx.output_queue[0] = (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
                                   is_last_step, is_first_step_output, skip)
        else:
            ctx.output_queue.appendleft((outputs, seq_group_metadata_list, scheduler_outputs, is_async,
                                         is_last_step, is_first_step_output, skip))

        now_time = time.perf_counter()
        for request_processing_context in request_processing_contexts:
            request_processing_context.add_trace_timeline('engine_process_model_outputs_timestamp_begin',now_time)

        super()._process_model_outputs(ctx, request_id)

        if ctx.request_outputs:
            _, request_processing_contexts = zip(*ctx.request_outputs)

            now_time = time.perf_counter()
            for request_processing_context in request_processing_contexts:
                request_processing_context.add_trace_timeline('engine_process_model_outputs_timestamp_end', now_time)

        if not self.disable_async_output_proc:
            while self._output_proc_done_event_queue.qsize() > 0:
                output_proc_done_event = self._output_proc_done_event_queue.get()
                output_proc_done_event.set()

    async def _process_request_outputs(
        self,
        outputs: List[Tuple[RequestOutput, RequestProcessingContext]],
    ) -> Tuple[List[RequestOutput], List[RequestProcessingContext]]:
        request_outputs = []
        request_processing_contexts: List[RequestProcessingContext] = []
        if outputs:
            request_outputs, request_processing_contexts = zip(*outputs)
            request_outputs = list(request_outputs)
            request_processing_contexts = list(request_processing_contexts)

        for context in request_processing_contexts:
            context.add_trace_timeline('engine_step_timestamp_begin', self.step_begin_time)
            context.add_trace_timeline('engine_step_timestamp_end')

        for request_output in request_outputs:
            if request_output.finished:
                logger.info("Engine finished request {}".format(request_output.request_id))
                if request_output.request_id in self.request_ids:
                    self.request_ids.remove(request_output.request_id)

        instance_info: InstanceInfo = self.instance_info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.profiling_data=(instance_info.inference_type.value if instance_info.inference_type else "",
                                      instance_info.num_seqs,
                                      sum(instance_info.running_seq_lens),
                                      self.model_executor.last_inference_latency)
        seq_groups = self.scheduler[0].running
        if seq_groups:
            tot_blocks = []
            for seq in seq_groups[-1].get_seqs(SequenceStatus.RUNNING):
                # temporary log to catch unfixed bug
                try:
                    blocks = self.scheduler[0].block_manager.get_block_table(seq)
                except KeyError:
                    logger.exception(
                        "Error in scheduler get_block_table (seq_id: {}, request_id: {})".format(
                            seq.seq_id, seq_groups[-1].request_id
                        )
                    )
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_blocks_last_running_request = len(tot_blocks)

        self.instance_info = instance_info

        for context in request_processing_contexts:
            context.add_trace_timeline("engine_put_queue_timestamp")
            context.add_trace_timeline("engine_step_postprocess_timestamp_end")

        if request_outputs:
            server_request_outputs, server_info_dict = self._gen_server_request_outputs(request_outputs, request_processing_contexts)
            if server_request_outputs:
                self.output_forwarder.put_request_outputs_to_server(server_request_outputs, server_info_dict)

        return request_outputs, request_processing_contexts

    def _gen_server_request_outputs(
        self,
        request_outputs: List[RequestOutput],
        request_processing_contexts: List[RequestProcessingContext]
    ) -> Tuple[Dict[str, List[LlumnixRequestOuput]], Dict[str, RequestProcessingContext]]:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}
        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output, request_processing_context in zip(request_outputs, request_processing_contexts):
            server_id = request_processing_context.server_id
            llumnix_resquest_output = LlumnixRequestOuput(
                request_output.request_id, self.instance_id, request_output, request_processing_context
            )
            server_request_outputs[server_id].append(llumnix_resquest_output)
            if server_id not in server_info_dict:
                server_info_dict[server_id] = request_processing_context.get_server_info()

        return server_request_outputs, server_info_dict

    async def step_async(self) -> Tuple[List[RequestOutput], List[RequestProcessingContext]]:
        self.step_begin_time = time.perf_counter()
        # pylint: disable=too-many-function-args
        outputs = await super().step_async(0)
        return await self._process_request_outputs(outputs)

    def stop(self) -> None:
        self.output_forwarder.stop()
        gc.collect()

    def update_instance_info(self, instance_info: InstanceInfo) -> None:
        # These fields are updated after step.
        if self.instance_info is not None:
            instance_info.instance_id = self.instance_info.instance_id
            instance_info.step_id = self.instance_info.step_id
            instance_info.timestamp = self.instance_info.timestamp
            instance_info.profiling_data = self.instance_info.profiling_data
            instance_info.num_blocks_last_running_request = self.instance_info.num_blocks_last_running_request
        if self.instance_info is None:
            instance_info.instance_id = self.instance_id
        self.instance_info = instance_info

    # pylint: disable=invalid-overridden-method
    async def add_request(self, request_id: str, request_processing_context: RequestProcessingContext, expected_steps: int, *args, **kwargs):
        if request_id in self.request_ids:
            logger.error('Request {} already exists in LLMEngineLlumnix.'.format(request_id))
            return
        self.request_ids.add(request_id)
        super().add_request(request_id, *args, **kwargs)
        seq_group = self.scheduler[0].waiting[-1]
        request_processing_context.add_trace_timeline('engine_add_request_timestamp')
        hit_length = kwargs.get("hit_length", 0)
        transfer_penalty = kwargs.get("transfer_penalty", 1)
        self.scheduler[0].waiting[-1] = SequenceGroupLlumnix(
            request_id,
            request_processing_context,
            expected_steps,
            hit_length,
            transfer_penalty,
            [seq_group.get_seqs()[0]],
            seq_group.metrics.arrival_time,
            seq_group.sampling_params,
            seq_group.lora_request,
            seq_group.trace_headers,
            seq_group.prompt_adapter_request,
            seq_group.encoder_seq,
            seq_group.priority
        )

    def clear_request(self, request_id) -> None:
        if request_id in self.request_ids:
            self.request_ids.remove(request_id)


class BackendVLLM(BackendBaseInterface, BackendMigrationInterface):
    def __init__(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        instance_args: InstanceArgs,
        llumnix_engine_args: LlumnixEngineArgs
    ) -> None:
        self.engine_disagg_inst_id = instance_id
        engine_args = llumnix_engine_args.load_engine_args()
        self.migration_config = instance_args.create_migration_config()
        self.engine: LLMEngineLlumnix = LLMEngineLlumnix.from_engine_args(
            engine_args=engine_args,
            request_output_queue_type=request_output_queue_type,
            migration_config=self.migration_config,
            instance_id=instance_id,
            placement_group=placement_group,
            backend_type=BackendType.VLLM,
            request_output_forwarding_mode=instance_args.request_output_forwarding_mode,
            abort_request_callback=self.abort_request,
        )
        # In order to call the verify_async_output_proc implicitly.
        engine_config = engine_args.create_engine_config()
        if not engine_config.model_config.use_async_output_proc:
            self.engine.scheduler = [SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config)
                                     for _ in range(engine_args.pipeline_parallel_size)]
        else:
            self.engine.scheduler = [
                SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config,
                                engine_args.pipeline_parallel_size, self.engine.async_callbacks[v_id])
                                for v_id in range(engine_args.pipeline_parallel_size)]
        for vid in range(engine_args.pipeline_parallel_size):
            self.engine.scheduler[vid].add_update_instance_info_callback(self.engine.update_instance_info)
        self.engine.output_processor.scheduler = self.engine.scheduler
        self.instance_id = instance_id
        self.worker_handle_list = self.engine.model_executor.workers.copy()
        if len(self.worker_handle_list) + 1 == self.engine.parallel_config.world_size:
            self.worker_handle_list.insert(0, get_llumnix_actor_handle(LlumnixActor.INSTANCE, self.instance_id))

        if self.migration_config.enable_migration:
            self._run_workers("init_migration", instance_id=instance_id,
                                                migration_config=self.migration_config,
                                                src_worker_handle_list=self.worker_handle_list,
                                                placement_group=placement_group)
        else:
            logger.info("Migration is disabled, skip migration initialization.")

        self.state = EngineState.INIT
        logger.info("engine {} current state: {}".format(self.instance_id, self.state))

        self.disable_async_output_proc = engine_args.disable_async_output_proc

        self._step_done_event_queue = queue.Queue()
        self._remove_running_request_ret: Dict[str] = {}
        if not self.disable_async_output_proc:
            self._output_proc_done_event_queue = queue.Queue()
            self.engine._output_proc_done_event_queue = self._output_proc_done_event_queue

        self.use_ray_spmd_worker = vllm_envs.VLLM_USE_RAY_SPMD_WORKER

        self._stop_event = asyncio.Event()
        asyncio.create_task(self._start_engine_step_loop())

    async def _start_engine_step_loop(self) -> None:
        self._stop_event.clear()

        previous_state = self.state
        self.state = EngineState.RUNNING
        logger.info("engine {} change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        while not self._stop_event.is_set():
            try:
                while self._step_done_event_queue.qsize() > 0:
                    request_id, step_done_event = self._step_done_event_queue.get()
                    self._remove_running_request_ret[request_id] = self._remove_running_request(request_id)
                    step_done_event.set()
                await asyncio.sleep(0.0)
                request_outputs, _ = await self.engine.step_async()
                if len(request_outputs) == 0:
                    await asyncio.sleep(NO_OUTPUTS_STEP_INTERVAL)
            # pylint: disable=broad-except
            except Exception:
                logger.exception("Error in engine loop")
                self.stop()
                previous_state = self.state
                self.state = EngineState.CRASHED
                logger.info("engine {} change state: {} -> {}".format(self.instance_id, previous_state, self.state))
                break

        if self.state == EngineState.RUNNING:
            self.stop()
            self.state = EngineState.STOPPED
            logger.info("engine {} change state: {} -> {}".format(self.instance_id, EngineState.RUNNING, self.state))

    def stop(self):
        self.engine.stop()
        logger.info("Engine stops, instance_id: {}".format(self.instance_id))

    async def execute_driver_worker_method_async(self, method, *args, **kwargs):
        return await make_async(self.engine.model_executor.driver_worker.execute_method)(method, *args, **kwargs)

    async def add_request(self, request_id: str, request_processing_context: RequestProcessingContext, expected_steps: int, *args, **kwargs) -> None:
        await self.engine.add_request(request_id, request_processing_context, expected_steps, *args, **kwargs)

    async def commit_dst_request(self,
                                 request_id: RequestIDType,
                                 backend_request: SequenceGroupLlumnix) -> MigrationResponse:
        if self.use_ray_spmd_worker and backend_request.llumnix_status == RequestStatus.RUNNING_MIGRATING:
            await self._run_workers_async("commit_seq_group_metadata", request_id)

        seq = backend_request.get_seqs()[0]
        seq.seq_id = next(self.engine.seq_counter)
        logger.info("pop request {} from pre-allocated cache".format(request_id))
        pre_alloc_blocks = self.engine.scheduler[0].pre_alloc_cache_dict.pop(request_id)
        logger.info("add seq {} to block table".format(seq.seq_id))
        self.engine.scheduler[0].block_manager.add_block_table(pre_alloc_blocks, seq.seq_id)
        backend_request.reset_migration_states_dst()
        assert RequestStatus.is_migrating(backend_request.llumnix_status), \
            "The status of request migrated to dst instance should be  \
             RequestStatus.WAITING_MIGRATING or RequestStatus.RUNNING_MIGRATING"
        if backend_request.llumnix_status == RequestStatus.RUNNING_MIGRATING:
            backend_request.reset_llumnix_status()
            await self.add_running_request(backend_request)
        else: # WAITING_MIGRATING:
            self.add_waiting_request(backend_request)

        return MigrationResponse(success=True, return_value=None)

    async def send_cache(self,
                         dst_instance_actor: ray.actor.ActorHandle,
                         src_blocks: List[int],
                         dst_blocks: List[int],
                         request_id: str,
                         is_last_stage: bool) -> MigrationResponse:
        return await asyncio_wait_for_ray_remote_call_with_timeout(
            dst_instance_actor.execute_migration_method_async,
            "recv_cache",
            request_id=request_id,
            src_worker_handle_list=self.worker_handle_list,
            src_blocks=src_blocks,
            dst_blocks=dst_blocks,
            is_last_stage=is_last_stage
        )

    async def recv_cache(self,
                         request_id: RequestIDType,
                         src_worker_handle_list: List[ray.actor.ActorHandle],
                         src_blocks: List[int],
                         dst_blocks: List[int],
                         is_last_stage: bool) -> MigrationResponse:
        success_list = await self._run_workers_async(
            "recv_cache", request_id, src_worker_handle_list, src_blocks, dst_blocks, is_last_stage)
        return MigrationResponse(success=all(success_list), return_value=None)

    def _run_workers(self, *args, timeout=RAY_RPC_TIMEOUT, **kwargs):
        # pylint: disable=protected-access
        return self.engine.model_executor._run_workers(*args, timeout=timeout, **kwargs)

    async def _run_workers_async(self, *args, timeout=RAY_RPC_TIMEOUT, **kwargs) -> List[Any]:
        # pylint: disable=protected-access
        return await make_async(self.engine.model_executor._run_workers)(*args, timeout=timeout, **kwargs)

    async def is_ready(self):
        return True

    async def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        return self.engine.abort_request(request_ids)

    def get_running_queue(self) -> Deque[SequenceGroupLlumnix]:
        return self.engine.scheduler[0].get_running_queue()

    def get_waiting_queue(self) -> Deque[SequenceGroupLlumnix]:
        return self.engine.scheduler[0].get_waiting_queue()

    async def get_request_incremental_blocks(self,
                                             backend_request: LlumnixRequest,
                                             pre_stage_num_blocks: int) -> Tuple[List[int], List[int]]:
        incremental_blocks, incremental_token_ids = \
            self.engine.scheduler[0].get_request_incremental_blocks(backend_request, pre_stage_num_blocks)
        is_last_stage = (len(incremental_blocks) <= self.migration_config.migration_last_stage_max_blocks) or \
            backend_request.blocking_migration
        return incremental_blocks, incremental_token_ids, is_last_stage

    async def remove_running_request(self, request_id: str) -> bool:
        step_done_event = asyncio.Event()
        self._step_done_event_queue.put((request_id, step_done_event))
        await step_done_event.wait()
        ret = self._remove_running_request_ret.pop(request_id)
        if not self.disable_async_output_proc:
            output_proc_done_event = asyncio.Event()
            self._output_proc_done_event_queue.put(output_proc_done_event)
            await output_proc_done_event.wait()
        return ret

    def _remove_running_request(self, request_id: str) -> bool:
        return self.engine.scheduler[0].remove_running_request(request_id)

    def remove_waiting_request(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].remove_waiting_request(*args, **kwargs)

    def add_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].add_migrating_out_request_last_stage(*args, **kwargs)

    def pop_migrating_out_request_last_stage(self, backend_request: LlumnixRequest) -> None:
        return self.engine.scheduler[0].pop_migrating_out_request_last_stage(backend_request.request_id)

    def pre_alloc_cache(self, *args, **kwargs) -> MigrationResponse:
        return self.engine.scheduler[0].pre_alloc_cache(*args, **kwargs)

    def should_abort_migration(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].should_abort_migration(*args, **kwargs)

    async def add_running_request(self, backend_request: LlumnixRequest) -> None:
        # Although add_running_request is always be called in last stage migration,
        # there exists migrating_out_seq_group_metadata in worker only when last stage do_send is executed,
        # so the request id does not always exists.
        if self.use_ray_spmd_worker and backend_request.llumnix_status == RequestStatus.RUNNING_MIGRATING:
            # pylint: disable=protected-access
            await self._run_workers_async(
                "restore_migrating_out_seq_group_metadata", backend_request.request_id
            )
        return self.engine.scheduler[0].add_running_request(backend_request)

    def add_waiting_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].add_waiting_request(*args, **kwargs)

    def is_request_running(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].is_request_running(*args, **kwargs)

    def free_pre_alloc_cache(self, request_id: str) -> None:
        # TODO(s5u13b): Only needed when migrating running request.
        if self.use_ray_spmd_worker:
            # pylint: disable=protected-access
            asyncio.create_task(self._run_workers_async("free_migrating_in_seq_group_metadata"))
        return self.engine.scheduler[0].free_pre_alloc_cache(request_id)

    def free_src_request(self, backend_request: SequenceGroup) -> None:
        # When free_src_request is called, it means that all migration operations is successful.
        # However, there exists migrating_out_seq_group_metadata in worker only when migrating running request,
        # so the request id does not always exists.
        if self.use_ray_spmd_worker and backend_request.llumnix_status == RequestStatus.RUNNING_MIGRATING:
            # pylint: disable=protected-access
            asyncio.create_task(
                self._run_workers_async(
                    "pop_migrating_out_seq_group_metadata", backend_request.request_id))
        self.engine.clear_request(backend_request.request_id)

        return self.engine.scheduler[0].free_src_request(backend_request)

    def get_instance_info(self):
        return self.engine.instance_info

    def get_engine_context(self):
        return InstanceContext(local_engine_id=self.instance_id)
