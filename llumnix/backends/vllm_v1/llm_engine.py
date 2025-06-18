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
from typing import List, Optional, Union, Iterable, Deque, Tuple, Dict, Any
from collections import defaultdict
import threading
import asyncio
import queue
import gc

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray.actor

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import VllmConfig

from vllm.v1.engine import EngineCoreRequest, EngineCoreOutputs
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.request import Request, RequestStatus

# from vllm.engine.async_llm_engine import _AsyncLLMEngine
# from vllm.outputs import RequestOutput, RequestOutputFactory, EmbeddingRequestOutput
# from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import Counter
# from vllm.usage.usage_lib import UsageContext
# from vllm.engine.llm_engine import SchedulerContext
from vllm import envs as vllm_envs

from llumnix.arg_utils import InstanceArgs, LlumnixEngineArgs
from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, RequestInferenceType
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.backends.vllm_v1.scheduler import SchedulerLlumnix
from llumnix.backends.vllm_v1.request import LlumnixRequestVLLMV1
from llumnix.backends.profiling import LatencyMemData
from llumnix.server_info import ServerInfo
from llumnix.internal_config import MigrationConfig
from llumnix.queue.utils import QueueType
from llumnix.backends.utils import AsyncPutQueueActor
from llumnix.utils import make_async, ray_get_with_timeout
from llumnix.ray_utils import get_instance_name, asyncio_wait_for_with_timeout
from llumnix.llumlet.request import LlumnixRequest
from llumnix.metrics.timestamps import set_timestamp
from llumnix.constants import NO_OUTPUTS_STEP_INTERVAL, RAY_RPC_TIMEOUT
from llumnix.backends.backend_interface import BackendType
from llumnix.request_output import LlumnixRequestOuput as LlumnixRequestOuputVLLM
from llumnix.utils import RequestIDType, MigrationResponse

logger = init_logger(__name__)


# class LlumnixRequestOutputFactory(RequestOutputFactory):
#     @staticmethod
#     def create(seq_group: SequenceGroupLlumnix, use_cache: bool = False):
#         # Determine the type based on a condition, for example:
#         if hasattr(seq_group,
#                    'embeddings') and seq_group.embeddings is not None:
#             return EmbeddingRequestOutput.from_seq_group(seq_group), seq_group.server_info
#         # pylint: disable=too-many-function-args
#         return RequestOutput.from_seq_group(seq_group, use_cache), seq_group.server_info


# class StopPutQueueSignal:
#     pass


class EngineCoreProcLlumnix(EngineCoreProc):
    def __init__(self,
                 instance_id: str,
                 vllm_config: VllmConfig,
                 *args, **kwargs) -> None:
        
        # Change EngineCore.scheduler to SchedulerLlumnix
        vllm_config.scheduler_config.scheduler_cls = SchedulerLlumnix

        super().__init__(vllm_config=vllm_config, *args, **kwargs)
        
        self.scheduler.add_update_instance_info_callback(self.update_instance_info)

        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None

        assert isinstance(self.scheduler, SchedulerLlumnix), \
            "EngineCore.scheduler failed to set to SchedulerLlumnix"

    # pylint: disable=W0221
    @classmethod
    def from_engine_args(
        cls,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: AsyncEngineArgs,
        backend_type: BackendType,
        latency_mem: Optional[LatencyMemData] = None,
    ) -> "EngineCoreProcLlumnix":
        """Creates an EngineCoreProc from the engine arguments."""
        # FIXME(zhaozhiyu): I don't where speculative_config is set, just overload it
        engine_args.speculative_config = None
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        logger.info("engine_config: {}", engine_config)
        # Hack to pass placement_group for init workers.
        engine_config.parallel_config.placement_group = placement_group
        # Initialize the cluster and specify the executor class.
        # pylint: disable=import-outside-toplevel
        if latency_mem is not None:
            raise NotImplementedError('vLLM v1 sim_executor not implemented yet')
            # from llumnix.backends.vllm_v1.sim_executor import SimGPUExecutor
            # executor_class = SimGPUExecutor
            # executor_class.latency_mem = latency_mem
        elif engine_config.parallel_config.use_ray:
            from llumnix.backends.vllm_v1.executor import LlumnixRayDistributedExecutor
            executor_class = LlumnixRayDistributedExecutor
            executor_class.migration_config = migration_config
            executor_class.instance_id = instance_id
        else:
            raise ValueError('Unsupported executor backend')
        # Create the EngineCoreProc
        # vllm_config: VllmConfig,
        # on_head_node: bool,
        # handshake_address: str,
        # executor_class: type[Executor],
        # log_stats: bool,
        # engine_index: int = 0,
        # FIXME(zhaozhiyu): pass corret args to EngineCoreProc
        engine = cls(
            instance_id=instance_id,
            vllm_config=engine_config,
            on_head_node=True,
            handshake_address="tcp://127.0.0.1:29550",
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
        )
        return engine

    # FIXME(zhaozhiyu): this method update instance info, maybe should move the logic to output_processor
    def _process_request_output(
            self,
            output: Tuple[int, EngineCoreOutputs],
    ):
        client_index, outputs = output

        # TODO(zhaozhiyu): check where this timestamp is used, determine whether to set timestamp in outputs or in each request_output
        set_timestamp(outputs, 'engine_step_timestamp_begin', self.step_begin_time)
        set_timestamp(outputs, 'engine_step_timestamp_end', self.step_end_time)

        for request_output in outputs.outputs:
            if request_output.finished:
                logger.info("Engine finished request {}".format(request_output.request_id))

        instance_info: InstanceInfo = self.instance_info # type: ignore
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.profiling_data=(instance_info.inference_type.value if instance_info.inference_type else "",
                                      instance_info.num_seqs,
                                      sum(instance_info.running_seq_lens),
                                      self.model_executor.last_inference_latency)
        reqs: List[LlumnixRequestVLLMV1] = self.scheduler.running
        if reqs:
            tot_blocks = defaultdict(list)
            for req in reqs:
                if req.status != RequestStatus.RUNNING:
                    continue
                # block_ids (List[List[int]]): A two-level list where
                # the outer list corresponds to KV cache groups
                # each inner list contains the block_ids of the blocks in that group
                block_ids: List[List[int]] = self.scheduler.kv_cache_manager.get_block_ids(req.request_id)
                for group_id, group in enumerate(block_ids):
                    tot_blocks[group_id].extend(group)
            
            num_blocks_last_running_request = 0
            for group_id, group in tot_blocks.items():
                num_blocks_last_running_request += len(set(group))
            instance_info.num_blocks_last_running_request = num_blocks_last_running_request

        self.instance_info = instance_info

        set_timestamp(outputs, 'engine_put_queue_timestamp', time.time())
        self.output_queue.put_nowait((client_index, outputs))
        # set_timestamp(outputs, 'engine_step_postprocess_timestamp_end', time.time())

    def _process_engine_step(self) -> bool:
        """Overloading EngineCore._process_engine_step() to update instance info"""

        # Step the engine core.
        self.step_begin_time = time.time()
        outputs, model_executed = self.step_fn()
        self.step_end_time = time.time()
        # Put EngineCoreOutputs into the output queue.
        for output in (outputs.items() if outputs else ()):
            self._process_request_output(output)

        return model_executed

    def stop(self) -> None:
        super().shutdown()
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
        # TODO(zhaozhiyu): super().add_request() bypass the input_queue in EngineCoreProc, should double-check whether it's properiate
        super().add_request(*args, **kwargs) # EngineCore.add_request(request: EngineCoreRequest)
        request: LlumnixRequestVLLMV1 = self.scheduler.waiting[-1]
        set_timestamp(server_info, 'engine_add_request_timestamp', time.time())
        self.scheduler.waiting[-1] = LlumnixRequestVLLMV1(
            request_id, server_info, expected_steps, 
            request.prompt_token_ids, request.mm_inputs,
            request.mm_hashes, request.mm_positions, 
            request.sampling_params, request.eos_token_id,
            request.client_index, request.lora_request,
            request.structured_output_request, request.cache_salt
        )


class BackendVLLMV1(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        instance_args: InstanceArgs,
        llumnix_engine_args: LlumnixEngineArgs
    ) -> None:
        self.engine_disagg_inst_id = instance_id
        engine_args: AsyncEngineArgs = llumnix_engine_args.load_engine_args() # type: ignore
        self.migration_config = instance_args.create_migration_config()
        # FIXME(zhaozhiyu): check args
        self.engine: EngineCoreProcLlumnix = EngineCoreProcLlumnix.from_engine_args(engine_args=engine_args,
                                                                          request_output_queue_type=request_output_queue_type,
                                                                          migration_config=self.migration_config,
                                                                          instance_id=instance_id,
                                                                          placement_group=placement_group,
                                                                          backend_type=BackendType.VLLM)
        # engine_config: VllmConfig = engine_args.create_engine_config()
        self.instance_id = instance_id
        self.worker_handle_list = self.engine.model_executor.workers.copy()
        if len(self.worker_handle_list) + 1 == self.engine.vllm_config.parallel_config.world_size:
            self.worker_handle_list.insert(0, ray.get_actor(get_instance_name(self.instance_id), namespace="llumnix"))

        if self.migration_config.enable_migration:
            self._run_workers("init_migration", instance_id=instance_id,
                                                migration_config=self.migration_config,
                                                src_worker_handle_list=self.worker_handle_list,
                                                placement_group=placement_group)
        else:
            logger.info("Migration is disabled, skip migration initialization.")

        self.state = EngineState.INIT
        logger.info("engine {} current state: {}".format(self.instance_id, self.state))

        # self.disable_async_output_proc = engine_args.disable_async_output_proc

        self._step_done_event_queue = queue.Queue()
        self._remove_running_request_ret: Dict[str] = {}
        # if not self.disable_async_output_proc:
        #     self._output_proc_done_event_queue = queue.Queue()
        #     self.engine._output_proc_done_event_queue = self._output_proc_done_event_queue

        # TODO(zhaozhiyu): determine whether this env var is still in vllm v1
        self.use_ray_spmd_worker = vllm_envs.VLLM_USE_RAY_SPMD_WORKER

        self._stop_event = asyncio.Event()
        asyncio.create_task(self._start_engine_step_loop())

    # FIXME(zhaozhiyu): This loop should be removed, running loop is inside EngineCoreProc. Need to handle self.state.
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
            except Exception as e:
                logger.exception("Error in engine loop, unexpected exception: {}".format(e))
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

    async def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        await self.engine.add_request(request_id, server_info, expected_steps, *args, **kwargs)

    async def commit_dst_request(self,
                                 request_id: RequestIDType,
                                 backend_request) -> MigrationResponse:
        raise NotImplementedError("commit_dst_request not implemented in vllm v1")
        # if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
        #     await self._run_workers_async("commit_seq_group_metadata", request_id)

        # seq = backend_request.get_seqs()[0]
        # seq.seq_id = next(self.engine.seq_counter)
        # logger.info("pop request {} from pre_alloc_cache_dict".format(request_id))
        # pre_alloc_blocks = self.engine.scheduler[0].pre_alloc_cache_dict.pop(request_id)
        # self.engine.scheduler[0].block_manager.add_block_table(pre_alloc_blocks, seq.seq_id)
        # backend_request.reset_migration_states_dst()
        # assert RequestStatus.is_migrating(backend_request.status), \
        #     "The status of request migrated to dst instance should be  \
        #      RequestStatus.WAITING_MIGRATING or RequestStatus.RUNNING_MIGRATING"
        # if backend_request.status == RequestStatus.RUNNING_MIGRATING:
        #     backend_request.reset_status()
        #     self.add_running_request(backend_request)
        # else: # WAITING_MIGRATING:
        #     self.add_waiting_request(backend_request)

        # return MigrationResponse(success=True, return_value=None)

    async def send_cache(self,
                         dst_instance_actor: ray.actor.ActorHandle,
                         src_blocks: List[int],
                         dst_blocks: List[int],
                         request_id: str,
                         is_last_stage: bool) -> MigrationResponse:
        raise NotImplementedError("send_cache not implemented in vllm v1")
        # return await asyncio_wait_for_with_timeout(
        #     dst_instance_actor.execute_migration_method_async.remote(
        #         "recv_cache",
        #         request_id=request_id,
        #         src_worker_handle_list=self.worker_handle_list,
        #         src_blocks=src_blocks,
        #         dst_blocks=dst_blocks,
        #         is_last_stage=is_last_stage
        #     )
        # )

    async def recv_cache(self,
                         request_id: RequestIDType,
                         src_worker_handle_list: List[ray.actor.ActorHandle],
                         src_blocks: List[int],
                         dst_blocks: List[int],
                         is_last_stage: bool) -> MigrationResponse:
        raise NotImplementedError("recv_cache is not implemented in vllm v1.")
        # success_list = await self._run_workers_async(
        #     "recv_cache", request_id, src_worker_handle_list, src_blocks, dst_blocks, is_last_stage)
        # return MigrationResponse(success=all(success_list), return_value=None)

    def _run_workers(self, *args, timeout=RAY_RPC_TIMEOUT, **kwargs):
        # pylint: disable=protected-access
        return self.engine.model_executor._run_workers(*args, timeout=timeout, **kwargs)

    async def _run_workers_async(self, *args, timeout=RAY_RPC_TIMEOUT, **kwargs) -> List[Any]:
        # pylint: disable=protected-access
        return await make_async(self.engine.model_executor._run_workers)(*args, timeout=timeout, **kwargs)

    # FIXME(zhaozhiyu): May need to check handshake result of EngineCoreClient and EngineCore
    async def is_ready(self):
        return True

    async def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        # EngineCore.abort_requests(request_ids: list[str])
        request_ids: List[str] = list(request_id)
        return self.engine.abort_requests(request_ids)

    def get_running_queue(self) -> List[LlumnixRequestVLLMV1]:
        return self.engine.scheduler.running

    def get_waiting_queue(self) -> Deque[LlumnixRequestVLLMV1]:
        return self.engine.scheduler.waiting

    async def get_request_incremental_blocks(self,
                                             backend_request: LlumnixRequest,
                                             pre_stage_num_blocks: int) -> Tuple[List[int], List[int]]:
        incremental_blocks, incremental_token_ids = \
            self.engine.scheduler.get_request_incremental_blocks(backend_request, pre_stage_num_blocks)
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

    def add_running_request(self, backend_request: LlumnixRequest) -> None:
        # Although add_running_request is always be called in last stage migration,
        # there exists migrating_out_seq_group_metadata in worker only when last stage do_send is executed,
        # so the request id does not always exists.
        if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
            # pylint: disable=protected-access
            asyncio.create_task(
                self._run_workers_async(
                    "restore_migrating_out_seq_group_metadata", backend_request.request_id))
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

    def free_src_request(self, backend_request) -> None:
        # When free_src_request is called, it means that all migration operations is successful.
        # However, there exists migrating_out_seq_group_metadata in worker only when migrating running request,
        # so the request id does not always exists.
        if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
            # pylint: disable=protected-access
            asyncio.create_task(
                self._run_workers_async(
                    "pop_migrating_out_seq_group_metadata", backend_request.request_id))
        return self.engine.scheduler[0].free_src_request(backend_request)
