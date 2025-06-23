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
from collections import defaultdict
import asyncio
import queue
import gc

import ray
from ray.util.placement_group import PlacementGroup
import ray.actor

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import VllmConfig
from vllm.utils import Counter
from vllm import envs as vllm_envs
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType, EngineCoreOutputs
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import get_engine_client_zmq_addr


from llumnix.arg_utils import InstanceArgs, LlumnixEngineArgs
from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, RequestInferenceType
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.backends.vllm_v1.async_core import AsyncEngineCoreProc
from llumnix.backends.vllm_v1.scheduler import SchedulerLlumnix
from llumnix.backends.vllm_v1.request import LlumnixRequestVLLMV1
from llumnix.backends.profiling import LatencyMemData
from llumnix.server_info import ServerInfo
from llumnix.internal_config import MigrationConfig
from llumnix.queue.utils import QueueType
from llumnix.backends.utils import RequestOutputForwardingMode, OutputMediator
from llumnix.utils import make_async, ray_get_with_timeout
from llumnix.ray_utils import get_instance_name
from llumnix.llumlet.request import LlumnixRequest
from llumnix.metrics.timestamps import set_timestamp
from llumnix.constants import NO_OUTPUTS_STEP_INTERVAL, RAY_RPC_TIMEOUT
from llumnix.backends.backend_interface import BackendType
from llumnix.utils import RequestIDType, MigrationResponse
from llumnix.request_output import LlumnixRequestOutputs


logger = init_logger(__name__)

class ServerInfoTable:
    """
    Manages a lookup table that maps a client_index (integer) to ServerInfo.
    Supports efficient add, delete, and lookup operations.
    """

    def __init__(self):
        self._servers: Dict[int, ServerInfo] = {}

    def add_server_info(self, client_index: int, server_info: ServerInfo):
        self._servers[client_index] = server_info

    def delete_server_info(self, client_index: int):
        if client_index in self._servers:
            del self._servers[client_index]

    def get_server_info(self, client_index: int) -> Optional[ServerInfo]:
        return self._servers.get(client_index)

    def list_all_servers(self) -> Dict[int, ServerInfo]:
        return self._servers.copy()

    def __len__(self) -> int:
        return len(self._servers)

    def __contains__(self, client_index: int) -> bool:
        return client_index in self._servers


class AsyncEngineCoreProcLlumnix(AsyncEngineCoreProc):
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 disable_async_output_proc: bool,
                 backend_type: BackendType,
                 request_output_forwarding_mode: RequestOutputForwardingMode,
                 abort_request_callback: Coroutine,
                 vllm_config: VllmConfig,
                 on_head_node: bool,
                 handshake_address: str,
                 executor_class: type[Executor],
                 log_stats: bool,
                 engine_index: int = 0) -> None:
        
        # Change EngineCore.scheduler to SchedulerLlumnix
        vllm_config.scheduler_config.scheduler_cls = SchedulerLlumnix
        super().__init__(vllm_config, on_head_node, handshake_address, executor_class, log_stats, engine_index)
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        self.output_mediator = OutputMediator(
            instance_id,
            request_output_queue_type,
            request_output_forwarding_mode,
            abort_request_callback,
            placement_group,
            backend_type,
        )
        
        self.scheduler.add_update_instance_info_callback(self.update_instance_info)
        self.disable_async_output_proc = disable_async_output_proc
        self.server_info_table = ServerInfoTable()

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
        request_output_forwarding_mode: RequestOutputForwardingMode,
        abort_request_callback: Coroutine,
        latency_mem: Optional[LatencyMemData] = None,
    ) -> "AsyncEngineCoreProcLlumnix":
        """Creates an EngineCoreProc from the engine arguments."""
        # FIXME(zhaozhiyu): don't known where speculative_config is set, just overload it
        engine_args.speculative_config = None
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        logger.info("engine_config: {}".format(engine_config))
        # Hack to pass placement_group for init workers.
        engine_config.parallel_config.placement_group = placement_group
        # Initialize the cluster and specify the executor class.
        # pylint: disable=import-outside-toplevel
        if latency_mem is not None:
            raise NotImplementedError('vLLM v1 sim_executor not implemented yet')
        elif engine_config.parallel_config.use_ray:
            from llumnix.backends.vllm_v1.executor import LlumnixRayDistributedExecutor
            executor_class = LlumnixRayDistributedExecutor
            executor_class.migration_config = migration_config
            executor_class.instance_id = instance_id
            logger.debug("executor_class set to LlumnixRayDistributedExecutor")
        else:
            raise ValueError('Unsupported executor backend')
        
        handshake_address = get_engine_client_zmq_addr(
            False, engine_config.parallel_config.data_parallel_master_ip, engine_config.parallel_config.data_parallel_rpc_port
        )
        engine = cls(
            instance_id=instance_id,
            placement_group=placement_group,
            request_output_queue_type=request_output_queue_type,
            disable_async_output_proc=engine_args.disable_async_output_proc,
            backend_type=backend_type,
            request_output_forwarding_mode=request_output_forwarding_mode,
            abort_request_callback=abort_request_callback,
            vllm_config=engine_config,
            on_head_node=True,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
        )
        return engine

    async def _put_engine_core_outputs(
        self,
        outputs: Dict[int, EngineCoreOutputs]
    ):
        # collects engine_core_output from all clients
        engine_core_output_all = []
        for engine_core_outputs in outputs.values():
            for engine_core_output in engine_core_outputs.outputs:
                engine_core_output_all.append(engine_core_output)
                if engine_core_output.finished:
                    logger.info("Engine finished request {}".format(engine_core_output.request_id))
                
        set_timestamp(engine_core_output_all, 'engine_step_timestamp_begin', self.step_begin_time)
        set_timestamp(engine_core_output_all, 'engine_step_timestamp_end', self.step_end_time)
        set_timestamp(engine_core_output_all, 'engine_put_queue_timestamp', time.time())
        
        if outputs:
            server_request_outputs, server_info_dict = self._gen_server_request_outputs(outputs)
            if server_request_outputs:
                await self.output_mediator.put_request_outputs_to_server(server_request_outputs, server_info_dict)
        
        set_timestamp(engine_core_output_all, 'engine_step_postprocess_timestamp_end', time.time())
        
    def _gen_server_request_outputs(
        self,
        engine_core_outputs_dict: Dict[int, EngineCoreOutputs]
    ) -> Tuple[Dict[str, LlumnixRequestOutputs], Dict[str, ServerInfo]]:
        server_request_outputs = {}
        server_info_dict = {}
        for client_index, engine_core_outputs in engine_core_outputs_dict.items():
            server_info = self.server_info_table.get_server_info(client_index)
            server_id = server_info.server_id
            
            server_request_outputs[server_id] = LlumnixRequestOutputs(
                instance_id=self.instance_id,
                engine_outputs=engine_core_outputs,
                request_timestamps_dict=None,
            )
            if server_id not in server_info_dict:
                server_info_dict[server_id] = server_info
            
        return server_request_outputs, server_info_dict
        
    def _update_instance_info(self):
        """Update instance info from executor and scheduler after step"""
        instance_info: InstanceInfo = self.instance_info # type: ignore
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.profiling_data=(instance_info.inference_type.value if instance_info.inference_type else "",
                                      instance_info.num_seqs,
                                      sum(instance_info.running_seq_lens),
                                      self.model_executor.last_inference_latency)
        reqs: List[Request] = self.scheduler.running
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

    async def _process_engine_step_async(self) -> bool:
        """Overloading super()._process_engine_step_async() to update instance info"""

        # Step the engine core.
        self.step_begin_time = time.time()
        outputs, model_executed = await self.step_fn_async()
        self.step_end_time = time.time()
        # Update instance info after step
        self._update_instance_info()
        # Put EngineCoreOutputs into output_mediator
        await self._put_engine_core_outputs(outputs)
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
    async def add_request_async(
        self,
        request_id: str,
        server_info: ServerInfo,
        expected_steps: int,
        *args, **kwargs,
    ):
        request_type = EngineCoreRequestType.ADD
        request: EngineCoreRequest = kwargs["engine_core_request"]
        self.server_info_table.add_server_info(request.client_index, server_info)
        await self.input_queue.put((request_type, request))


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
        self.engine: AsyncEngineCoreProcLlumnix = AsyncEngineCoreProcLlumnix.from_engine_args(
            instance_id=instance_id,
            placement_group=placement_group,
            request_output_queue_type=request_output_queue_type,
            migration_config=self.migration_config,            
            engine_args=engine_args,
            backend_type=BackendType.VLLM_V1,
            request_output_forwarding_mode=instance_args.request_output_forwarding_mode,
            abort_request_callback=self.abort_request,
        )
        asyncio.create_task(self.engine.run_busy_loop_async())
        
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
        self.use_ray_spmd_worker = vllm_envs.VLLM_USE_RAY_SPMD_WORKER


    def stop(self):
        self.engine.stop()
        logger.info("Engine stops, instance_id: {}".format(self.instance_id))

    async def execute_driver_worker_method_async(self, method, *args, **kwargs):
        return await make_async(self.engine.model_executor.driver_worker.execute_method)(method, *args, **kwargs)

    async def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        await self.engine.add_request_async(request_id, server_info, expected_steps, *args, **kwargs)

    async def commit_dst_request(self,
                                 request_id: RequestIDType,
                                 backend_request) -> MigrationResponse:
        raise NotImplementedError("commit_dst_request not implemented in vllm v1")

    async def send_cache(self,
                         dst_instance_actor: ray.actor.ActorHandle,
                         src_blocks: List[int],
                         dst_blocks: List[int],
                         request_id: str,
                         is_last_stage: bool) -> MigrationResponse:
        raise NotImplementedError("send_cache not implemented in vllm v1")

    async def recv_cache(self,
                         request_id: RequestIDType,
                         src_worker_handle_list: List[ray.actor.ActorHandle],
                         src_blocks: List[int],
                         dst_blocks: List[int],
                         is_last_stage: bool) -> MigrationResponse:
        raise NotImplementedError("recv_cache is not implemented in vllm v1.")


    def _run_workers(self, *args, timeout=RAY_RPC_TIMEOUT, **kwargs):
        # pylint: disable=protected-access
        return self.engine.model_executor._run_workers(*args, timeout=timeout, **kwargs)

    async def _run_workers_async(self, *args, timeout=RAY_RPC_TIMEOUT, **kwargs) -> List[Any]:
        # pylint: disable=protected-access
        return await make_async(self.engine.model_executor._run_workers)(*args, timeout=timeout, **kwargs)

    # TODO(zhaozhiyu): determine whether handshake result check is needed
    async def is_ready(self):
        return True

    async def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
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
