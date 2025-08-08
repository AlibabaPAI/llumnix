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

import copy
from dataclasses import dataclass
from functools import partial
import time
from typing import List, Optional, Union, Iterable, Tuple, Dict, Any, Coroutine
import queue
import threading
import asyncio
import itertools

import pickle
import ray
from ray.util.placement_group import PlacementGroup
import ray.actor

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import VllmConfig
from vllm import envs as vllm_envs
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType, EngineCoreOutputs, EngineCoreOutput, UtilityOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.hybrid_connector.kvtbackend import D_DISAGG
from vllm.v1.engine.dpcoord import Participant
from vllm.v1.engine.core import EngineCore, EngineCoreProc, DPEngineCoreProc, _core_init
from vllm.v1.core.sched.output import ScheduledDPMetaData
from vllm.v1.utils import sync_dp_metadata
from vllm.v1.hybrid_connector.engine_proxy import req2corereq
from vllm.v1.request import Request

from llumnix.arg_utils import InstanceArgs, LlumnixEngineArgs
from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendBaseInterface
from llumnix.backends.vllm_v1.scheduler import SchedulerLlumnix
from llumnix.backends.vllm_v1.migration_frontend import MigrationFrontend
from llumnix.backends.profiling import LatencyMemData
from llumnix.server_info import ServerInfo
from llumnix.request_processing_context import RequestProcessingContext
from llumnix.internal_config import MigrationConfig
from llumnix.queue.utils import QueueType
from llumnix.backends.utils import EngineState
from llumnix.backends.output_forwarder import RequestOutputForwardingMode, OutputForwarder
from llumnix.utils import asyncio_wait_for_with_timeout, get_ip_address, log_instance_exception
from llumnix.metrics.timestamps import set_timestamp
from llumnix.utils import (
    BackendType,
    get_free_port,
    InstanceContext,
    InstanceType,
    MigrationType,
    RequestIDType
)
from llumnix.request_output import LlumnixRequestOutputs
import llumnix.envs as llumnix_envs

logger = init_logger(__name__)


class EngineCoreProcLlumnix(EngineCoreProc, EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool, # pylint: disable=unused-argument
        handshake_address: str, # pylint: disable=unused-argument
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None, # pylint: disable=unused-argument
        engine_index: int = 0,
        driver_tensor_queue_union: Union[None, Any] = None,
    ):
        # pylint: disable=line-too-long
        """
        The original codes of vLLM (commit id: 6ffc9896eb3c9e2ecdaa779c5d54ac16b1c93f62):

        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")

        with self._perform_handshakes(handshake_address, identity,
                                      local_client, vllm_config,
                                      client_handshake_address) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address)
            # Only publish request queue stats to coordinator for "internal"
            # LB mode.
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb)

            self._init_data_parallel(vllm_config)
            _core_init(self, vllm_config)

            super().__init__(vllm_config, executor_class, log_stats,
                             executor_fail_callback, driver_tensor_queue_union)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        threading.Thread(target=self.process_input_sockets,
                         args=(addresses.inputs, addresses.coordinator_input,
                               identity),
                         daemon=True).start()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(addresses.outputs, addresses.coordinator_output,
                  self.engine_index),
            daemon=True)
        self.output_thread.start()

        # In batch mode, start a background heartbeat thread to maintain upstream connections by sending heartbeats every heartbeat_interval for requests in scheduler_proxy and the waiting queue.
        if self.vllm_config.scheduler_config.policy == "batch":
            threading.Thread(
                target=self._heartbeat_loop,
                args=(self.vllm_config.scheduler_config.heartbeat_interval, ),
                daemon=True,
            ).start()
        """

        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        self.engine_index = engine_index

        # NOTE(shejiarui): we don't launch vLLM's DPCoordinator
        self.has_coordinator = False
        self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb)

        self._init_data_parallel(vllm_config)
        _core_init(self, vllm_config)

        EngineCore.__init__(self, vllm_config, executor_class, log_stats,
                            executor_fail_callback, driver_tensor_queue_union)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)


class DPEngineCoreProcLlumnix(EngineCoreProcLlumnix, DPEngineCoreProc):
    def __init__(
        self,
        vllm_config: VllmConfig,
        on_head_node: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        self._decorate_logs()

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        EngineCoreProcLlumnix.__init__(self, vllm_config, on_head_node, handshake_address,
                                       executor_class, log_stats, dp_rank)

    def _init_data_parallel(self, vllm_config: VllmConfig):
        """
        The original codes of vLLM (commit id: 6ffc9896eb3c9e2ecdaa779c5d54ac16b1c93f62):

        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self._dppart = Participant(vllm_config, self)
        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )

            if not vllm_config.parallel_config.data_parallel_external_lb:
                vllm_config.kv_transfer_config.engine_available_port += vllm_config.parallel_config.data_parallel_rank_local

            logger.info("adjust kvt cfg: %s", vllm_config.kv_transfer_config)
            logger.debug("Setting kv_transfer_config.engine_id to %s",
                         vllm_config.kv_transfer_config.engine_id)

        from vllm.platforms import current_platform
        device_control_env_var = current_platform.device_control_env_var
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            os.environ[device_control_env_var] = ",".join(
                str(current_platform.device_id_to_physical_device_id(i))
                for i in range(local_dp_rank *
                               world_size, (local_dp_rank + 1) * world_size))
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f"base value: \"{os.getenv(device_control_env_var)}\"") from e

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()
        """
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self._dppart = Participant(vllm_config, self)

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )

            if not vllm_config.parallel_config.data_parallel_external_lb:
                vllm_config.kv_transfer_config.engine_available_port += vllm_config.parallel_config.data_parallel_rank_local

            logger.info("adjust kvt cfg: %s", vllm_config.kv_transfer_config)
            logger.debug("Setting kv_transfer_config.engine_id to %s",
                         vllm_config.kv_transfer_config.engine_id)

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()


class EngineCoreProcWrapperLlumnix(EngineCoreProcLlumnix):
    def __init__(
        self,
        instance_id: str,
        instance_type: InstanceType,
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
        utility_output_queue: queue.Queue,
        engine_index: int = 0,
        dp_rank: int = 0,
    ) -> None:
        # Change EngineCore.scheduler to SchedulerLlumnix
        vllm_config.scheduler_config.scheduler_cls = SchedulerLlumnix
        super().__init__(vllm_config, on_head_node, handshake_address, executor_class, log_stats, engine_index)
        self.instance_id = instance_id
        self.instance_type = instance_type
        self.instance_info = InstanceInfo(instance_id=instance_id, instance_type=instance_type)
        self.output_forwarder = OutputForwarder(
            instance_id,
            request_output_queue_type,
            request_output_forwarding_mode,
            abort_request_callback,
            placement_group,
            backend_type,
            dp_rank,
        )
        self.main_loop = asyncio.get_event_loop()
        self.scheduler.add_update_instance_info_callback(self.update_instance_info_threadsafe)
        self.disable_async_output_proc = disable_async_output_proc

        self.reqeust_processing_context_table: Dict[str, RequestProcessingContext] = {}

        # used in UTILITY call
        self.utility_output_queue: queue.Queue = utility_output_queue

    # pylint: disable=W0221
    @classmethod
    def from_engine_args(
        cls,
        instance_id: str,
        instance_type: InstanceType,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: AsyncEngineArgs,
        backend_type: BackendType,
        request_output_forwarding_mode: RequestOutputForwardingMode,
        abort_request_callback: Coroutine,
        utility_output_queue: queue.Queue,
        latency_mem: Optional[LatencyMemData] = None,
    ) -> "EngineCoreProcWrapperLlumnix":
        """Creates an EngineCoreProc from the engine arguments."""
        # FIXME(zhaozhiyu): This is a bug of pai-vllm, engine_args.speculative_config
        # must be set to None before calling engine_args.create_engine_config()
        if hasattr(engine_args, "speculative_config") and engine_args.speculative_config is not None:
            if len(engine_args.speculative_config) == 0:
                engine_args.speculative_config = None
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        # Hack to pass placement_group for init workers.
        engine_config.parallel_config.placement_group = placement_group
        # Initialize the cluster and specify the executor class.
        # pylint: disable=import-outside-toplevel
        if latency_mem is not None:
            raise NotImplementedError('vLLM v1 sim_executor not implemented yet')
        if engine_config.parallel_config.use_ray:
            from llumnix.backends.vllm_v1.executor import LlumnixRayDistributedExecutor
            executor_class = LlumnixRayDistributedExecutor
            executor_class.migration_config = migration_config
            executor_class.instance_id = instance_id
        else:
            executor_class = Executor.get_class(engine_config)

        engine = cls(
            instance_id=instance_id,
            instance_type=instance_type,
            placement_group=placement_group,
            request_output_queue_type=request_output_queue_type,
            disable_async_output_proc=engine_args.disable_async_output_proc,
            backend_type=backend_type,
            request_output_forwarding_mode=request_output_forwarding_mode,
            abort_request_callback=abort_request_callback,
            vllm_config=engine_config,
            on_head_node=True,
            handshake_address=None, # handshake is removed in llumnix
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            utility_output_queue=utility_output_queue
        )
        return engine

    def _put_engine_core_outputs(
        self,
        outputs: Dict[int, EngineCoreOutputs]
    ) -> None:
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
                self.output_forwarder.put_request_outputs_to_server(server_request_outputs, server_info_dict)

        set_timestamp(engine_core_output_all, 'engine_step_postprocess_timestamp_end', time.time())

    def add_request(self, request: EngineCoreRequest):
        request_processing_context = request.sampling_params.extra_args.get(
                'llumnix_request_processing_context', None)
        try:
            assert request_processing_context is not None
            request_processing_context = pickle.loads(request_processing_context)
            self.reqeust_processing_context_table[request.request_id] = request_processing_context
        # pylint: disable=broad-except
        except Exception:
            logger.exception("Failed to load request processing context from table, request id: {}, " \
                "processing_context: {}.".format(request.request_id, request_processing_context))
            self.reqeust_processing_context_table[request.request_id] = None

        super().add_request(request)

    def abort_requests(self, request_ids: list[str]):
        for request_id in request_ids:
            self.reqeust_processing_context_table.pop(request_id, None)
        super().abort_requests(request_ids)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            _, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                output.result = method(
                    *self._convert_msgspec_args(method, args))
            # pylint: disable=broad-except
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            if call_id is None:
                # Not interested in the result
                return

            self.utility_output_queue.put_nowait(output)

        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error("Unrecognized input request type encountered: %s",
                         request_type)

    def _gen_server_request_outputs(
        self,
        engine_core_outputs_dict: Dict[int, EngineCoreOutputs]
    ) -> Tuple[Dict[str, LlumnixRequestOutputs], Dict[str, ServerInfo]]:
        server_request_outputs = {}
        server_info_dict = {}

        for _, engine_core_outputs in engine_core_outputs_dict.items():
            request_processing_context_dict: Dict[str, RequestProcessingContext] = {}
            server_id = None
            server_info = None

            new_engine_core_outputs: List[EngineCoreOutput] = [] # used to filter prefill responce under pdd
            for engine_core_output in engine_core_outputs.outputs:
                request_id = engine_core_output.request_id
                request_processing_context: RequestProcessingContext = \
                    self.reqeust_processing_context_table.get(request_id, None)

                if request_processing_context is None or "hybridsched" in request_id:
                    if "hybridsched" not in request_id:
                        logger.warning("Request {} is not found in request_processing_context_table".format(request_id))
                    continue

                new_engine_core_outputs.append(engine_core_output)
                request_processing_context_dict[request_id] = request_processing_context
                if engine_core_output.finished:
                    self.reqeust_processing_context_table.pop(request_id, None)

                if server_id is None:
                    server_id = request_processing_context.server_id
                    server_info = request_processing_context.get_server_info()
            engine_core_outputs.outputs = new_engine_core_outputs

            if server_id is not None:
                server_request_outputs[server_id] = LlumnixRequestOutputs(
                    instance_id=self.instance_id,
                    engine_outputs=engine_core_outputs,
                    request_processing_context_dict=request_processing_context_dict
                )
                server_info_dict[server_id] = server_info

        return server_request_outputs, server_info_dict

    def _process_engine_step(self) -> bool:
        """Overloading super()._process_engine_step() to update instance info and forward outputs."""
        # Step the engine core.
        self.step_begin_time = time.time()
        outputs, model_executed = self.step_fn()
        self.step_end_time = time.time()
        # Put EngineCoreOutputs into output_mediator
        self._put_engine_core_outputs(outputs)
        return model_executed

    def stop(self) -> None:
        super().shutdown()

    def update_instance_info(self, instance_info: InstanceInfo) -> None:
        self.instance_info = instance_info

    def update_instance_info_threadsafe(self, instance_info: InstanceInfo):
        # Read and write instance_info in main process asyncio loop.
        self.main_loop.call_soon_threadsafe(self.update_instance_info, instance_info)

    def core_add_request(self, request: EngineCoreRequest) -> None:
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, request))

    def core_abort_requests(self, request_ids: List[str]) -> None:
        self.input_queue.put_nowait((EngineCoreRequestType.ABORT, request_ids))

    def utility_call(self, call_id: int, method: str, *args):
        request = (0, call_id, method, args)
        self.input_queue.put_nowait((EngineCoreRequestType.UTILITY, request))

    # ---- methods for utility call ----

    def get_migrated_requests(self, migration_type: MigrationType) -> List[Request]:
        def is_migrating(req: Request) -> bool:
            if req.sampling_params.extra_args is None:
                return False
            if req.sampling_params.extra_args.get("kv_transfer_params", None) is None:
                return False
            return req.sampling_params.extra_args['kv_transfer_params'].get('is_migrating', False)

        def set_migrating_status(req: Request):
            if req.sampling_params.extra_args is None:
                req.sampling_params.extra_args = {}
            if req.sampling_params.extra_args.get("kv_transfer_params", None) is None:
                req.sampling_params.extra_args["kv_transfer_params"] = {}
            req.sampling_params.extra_args['kv_transfer_params']['is_migrating'] = True

        migrated_out_request = []
        try:
            if migration_type == MigrationType.PRE_STOP_MIGRATION:
                for req in itertools.chain(self.scheduler.waiting, self.scheduler.running):
                    if "hybridsched" not in req.request_id and not is_migrating(req):
                        migrated_out_request = copy.deepcopy(req)
                        set_migrating_status(req)
            else:
                for req in self.scheduler.running:
                    if "hybridsched" not in req.request_id and not is_migrating(req):
                        migrated_out_request = copy.deepcopy(req)
                        set_migrating_status(req)
                        break
        # pylint: disable=broad-except
        except Exception:
            logger.exception("Failed to get migrated out request, migration_type: {}".format(migration_type))
            migrated_out_request = []

        return migrated_out_request

    def reset_request_migrating_state(self, request_id: int):
        for req in itertools.chain(self.scheduler.waiting, self.scheduler.running):
            if req.request_id == request_id:
                if (
                    req.sampling_params.extra_args \
                    and req.sampling_params.extra_args.get("kv_transfer_params", False) \
                    and req.sampling_params.extra_args['kv_transfer_params'].get('is_migrating', False)
                ):
                    req.sampling_params.extra_args['kv_transfer_params']['is_migrating'] = False
                else:
                    return

class DPEngineCoreProcWrapperLlumnix(DPEngineCoreProcLlumnix, EngineCoreProcWrapperLlumnix):
    def __init__(self, *args, **kwargs):
        # pylint: disable=super-init-not-called
        EngineCoreProcWrapperLlumnix.__init__(self, *args, **kwargs)

    # pylint: disable=arguments-differ
    @classmethod
    def from_engine_args(
        cls,
        instance_id: str,
        instance_type: InstanceType,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: AsyncEngineArgs,
        backend_type: BackendType,
        request_output_forwarding_mode: RequestOutputForwardingMode,
        abort_request_callback: Coroutine,
        utility_output_queue: queue.Queue,
        latency_mem: Optional[LatencyMemData] = None,
        dp_rank: int = 0,
        dp_rank_local: Optional[int] = None,
    ) -> "DPEngineCoreProcWrapperLlumnix":
        """Creates an DPEngineCoreProc from the engine arguments."""
        if hasattr(engine_args, "speculative_config") and engine_args.speculative_config is not None:
            if len(engine_args.speculative_config) == 0:
                engine_args.speculative_config = None
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        engine_config.parallel_config.data_parallel_master_port = vllm_envs.VLLM_DP_MASTER_PORT
        logger.info("engine_config: {}".format(engine_config))
        # Hack to pass placement_group for init workers.
        engine_config.parallel_config.placement_group = placement_group
        engine_config.parallel_config.data_parallel_rank = dp_rank
        engine_config.parallel_config.data_parallel_rank_local = dp_rank_local
        # Initialize the cluster and specify the executor class.
        # pylint: disable=import-outside-toplevel
        if latency_mem is not None:
            raise NotImplementedError('vLLM v1 sim_executor not implemented yet')
        if engine_config.parallel_config.use_ray:
            from llumnix.backends.vllm_v1.executor import LlumnixRayDistributedExecutor
            executor_class = LlumnixRayDistributedExecutor
            executor_class.migration_config = migration_config
            executor_class.instance_id = instance_id
        else:
            executor_class = Executor.get_class(engine_config)

        engine = cls(
            instance_id=instance_id,
            instance_type=instance_type,
            placement_group=placement_group,
            request_output_queue_type=request_output_queue_type,
            disable_async_output_proc=engine_args.disable_async_output_proc,
            backend_type=backend_type,
            request_output_forwarding_mode=request_output_forwarding_mode,
            abort_request_callback=abort_request_callback,
            vllm_config=engine_config,
            on_head_node=None,
            handshake_address=None,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            dp_rank=dp_rank,
            utility_output_queue=utility_output_queue,
        )
        return engine

    def _process_engine_step(self):
        """
        The original codes of vLLM (commit id: 6ffc9896eb3c9e2ecdaa779c5d54ac16b1c93f62):

        forward_executed = super()._process_engine_step()
        self._maybe_publish_request_counts()
        if forward_executed:
            return

        if self.engines_running and not self.scheduler.has_unfinished_requests(
        ):
            self._dppart.new_step()
            dp_metadata = None
            if self.vllm_config.scheduler_config.async_scheduling:
                num_pad, num_tokens_across_dp, is_prompt_batch, _, skip_cuda_graphs, all_prefill, all_decode = sync_dp_metadata(
                    self.dp_group,
                    1,
                    0,
                    False,
                    dp_size=self.vllm_config.parallel_config.data_parallel_size
                )
                dp_metadata = ScheduledDPMetaData(
                    num_pad=num_pad,
                    num_tokens_across_dp=num_tokens_across_dp,
                    is_prompt_batch=is_prompt_batch,
                    skip_cuda_graphs=skip_cuda_graphs,
                    all_prefill=all_prefill,
                    all_decode=all_decode)
            self.execute_dummy_batch(dp_metadata=dp_metadata)
        return

        """
        forward_executed = EngineCoreProcWrapperLlumnix._process_engine_step(self)
        self._maybe_publish_request_counts()
        if forward_executed:
            return

        if self.engines_running and not self.scheduler.has_unfinished_requests(
        ):
            self._dppart.new_step()
            dp_metadata = None
            if self.vllm_config.scheduler_config.async_scheduling:
                num_pad, num_tokens_across_dp, is_prompt_batch, _, skip_cuda_graphs, all_prefill, all_decode = sync_dp_metadata(
                    self.dp_group,
                    1,
                    0,
                    False,
                    dp_size=self.vllm_config.parallel_config.data_parallel_size
                )
                dp_metadata = ScheduledDPMetaData(
                    num_pad=num_pad,
                    num_tokens_across_dp=num_tokens_across_dp,
                    is_prompt_batch=is_prompt_batch,
                    skip_cuda_graphs=skip_cuda_graphs,
                    all_prefill=all_prefill,
                    all_decode=all_decode)
            self.execute_dummy_batch(dp_metadata=dp_metadata)
        return


@dataclass
class UtilityOutputHandler:
    result: Any = None
    ready_event: asyncio.Event = None

    method: str = None
    method_args: Any = None


class BackendVLLMV1(BackendBaseInterface):
    def __init__(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        instance_args: InstanceArgs,
        llumnix_engine_args: LlumnixEngineArgs,
        dp_rank: int = 0,
        dp_rank_local: Optional[int] = None
    ):
        self.instance_id = instance_id
        self.migrated_requests = set()
        self.host = get_ip_address()
        engine_args = self._load_and_reconfig_engine_args(llumnix_engine_args)
        self.migration_config = instance_args.create_migration_config()
        self.enable_vllm_v1_engine_pd_disagg = instance_args.enable_vllm_v1_engine_pd_disagg

        self.utility_output_queue = queue.Queue()
        self.utility_output_handlers: Dict[int, UtilityOutputHandler] = {}
        self.call_id = 0
        self.process_utility_output_loop_thread = threading.Thread(
            target=self._process_utility_output_loop, args=(asyncio.get_event_loop(),),
            daemon=True, name="process_utility_output_loop_thread"
        )
        self.process_utility_output_loop_thread.start()

        if engine_args.data_parallel_size > 1:
            self.engine: DPEngineCoreProcWrapperLlumnix = DPEngineCoreProcWrapperLlumnix.from_engine_args(
                instance_id=instance_id,
                instance_type=instance_args.instance_type,
                placement_group=placement_group,
                request_output_queue_type=request_output_queue_type,
                migration_config=self.migration_config,
                engine_args=engine_args,
                backend_type=BackendType.VLLM_V1,
                request_output_forwarding_mode=instance_args.request_output_forwarding_mode,
                abort_request_callback=self.abort_request,
                dp_rank=dp_rank,
                dp_rank_local=dp_rank_local,
                utility_output_queue=self.utility_output_queue,
            )
        else:
            self.engine: EngineCoreProcWrapperLlumnix = EngineCoreProcWrapperLlumnix.from_engine_args(
                instance_id=instance_id,
                instance_type=instance_args.instance_type,
                placement_group=placement_group,
                request_output_queue_type=request_output_queue_type,
                migration_config=self.migration_config,
                engine_args=engine_args,
                backend_type=BackendType.VLLM_V1,
                request_output_forwarding_mode=instance_args.request_output_forwarding_mode,
                abort_request_callback=self.abort_request,
                utility_output_queue=self.utility_output_queue,
            )
        self.run_busy_loop_thread = threading.Thread(
            target=self.engine.run_busy_loop, args=(), daemon=True, name="run_busy_loop_thread"
        )
        self.run_busy_loop_thread.start()

        if self.migration_config.enable_migration:
            self.migration_frontend = MigrationFrontend(self.engine.vllm_config, dp_rank=dp_rank)
            self.is_migrating = False # used to control migration concurrency

        self.state = EngineState.INIT
        logger.info("engine {} current state: {}".format(self.instance_id, self.state))

        self._step_done_event_queue = queue.Queue()
        self._remove_running_request_ret: Dict[str] = {}
        self.use_ray_spmd_worker = vllm_envs.VLLM_USE_RAY_SPMD_WORKER

    def _get_next_call_id(self) -> int:
        self.call_id += 1
        return self.call_id

    def _process_utility_output_loop(self, loop: asyncio.AbstractEventLoop):
        async def process_utility_output(output: UtilityOutput):
            utility_output_handle = self.utility_output_handlers[output.call_id]
            utility_output_handle.result = output.result
            utility_output_handle.ready_event.set()

        while True:
            output: UtilityOutput = self.utility_output_queue.get()
            asyncio.run_coroutine_threadsafe(process_utility_output(output), loop)

    def _load_and_reconfig_engine_args(self, llumnix_engine_args: LlumnixEngineArgs):
        engine_args: AsyncEngineArgs = llumnix_engine_args.load_engine_args()
        if engine_args.kv_transfer_config is not None:
            engine_args.kv_transfer_config.kv_connector_extra_config["rpc_port"] = None
            engine_args.kv_transfer_config.engine_available_port = get_free_port()
        return engine_args

    def stop(self):
        self.engine.stop()
        self.run_busy_loop_thread.join(timeout=5.0)
        if self.run_busy_loop_thread.is_alive():
            logger.exception("Failed to shutdown engine run_busy_loop thread.")
        self.process_utility_output_loop_thread.join(timeout=5.0)
        if self.process_utility_output_loop_thread.is_alive():
            logger.exception("Failed to shutdown engine process_utility_output_loop thread.")
        logger.info("Engine stops, instance_id: {}".format(self.instance_id))

    # pylint: disable=unused-argument
    async def add_request(
        self,
        request_id: str,
        request_processing_context: RequestProcessingContext,
        expected_steps: int,
        *args,
        **kwargs,
    ) -> None:
        request: EngineCoreRequest = kwargs["engine_core_request"]
        if request.sampling_params.extra_args is None:
            request.sampling_params.extra_args = {}

        if "llumnix_scheduler" in kwargs:
            prefill_instance_id = kwargs["prefill_instance_id"]
            decode_instance_id = kwargs["decode_instance_id"]
            assert decode_instance_id == self.instance_id

            kv_transfer_params = {}
            request.sampling_params.extra_args['kv_transfer_params'] = kv_transfer_params

            if prefill_instance_id == decode_instance_id:
                if self.instance_info.instance_type == InstanceType.DECODE:
                    kv_transfer_params[D_DISAGG] = False
            else:
                kv_transfer_params["remote_host"] = kwargs["prefill_engine_host"]
                kv_transfer_params["remote_port"] = kwargs["prefill_kvt_engine_available_port"]

        request.sampling_params.extra_args['llumnix_request_processing_context'] = \
            pickle.dumps(request_processing_context)

        if not self.enable_vllm_v1_engine_pd_disagg:
            kv_transfer_params = request.sampling_params.extra_args.get('kv_transfer_params', None)
            if kv_transfer_params is None:
                request.sampling_params.extra_args['kv_transfer_params'] = {}
            request.sampling_params.extra_args['kv_transfer_params'][D_DISAGG] = False

        self.engine.core_add_request(request)

    async def is_ready(self):
        return True

    async def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids: List[str] = list(request_id)
        self.engine.core_abort_requests(request_ids)

    def get_instance_info(self):
        return self.engine.instance_info

    def get_engine_context(self):
        kvt_engine_available_port = None
        if self.engine.vllm_config.kv_transfer_config is not None:
            kvt_engine_available_port = self.engine.vllm_config.kv_transfer_config.engine_available_port

        return InstanceContext(
            instance_id=self.instance_id,
            local_engine_id=self.instance_id,
            kvt_engine_available_port=kvt_engine_available_port,
            engine_host=self.host)

    def untility_call(self, method: str, *args) -> int:
        call_id = self._get_next_call_id()
        utility_output_ready_event = asyncio.Event()
        self.utility_output_handlers[call_id] = UtilityOutputHandler(
            ready_event=utility_output_ready_event,
            method=method,
            method_args=args,
        )
        self.engine.utility_call(call_id, method, *args)
        return call_id

    async def wait_untility_call_output(self, call_id: int):
        try:
            utility_output_ready_event = self.utility_output_handlers[call_id].ready_event
            await asyncio_wait_for_with_timeout(utility_output_ready_event.wait(),
                                                timeout=llumnix_envs.UTILITY_CALL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error('Timeout waiting for utility call: {}'.format(self.utility_output_handlers[call_id]))
        except Exception: # pylint: disable=broad-except
            logger.exception('Failed to waiting for utility output call {}.'.format(
                self.utility_output_handlers[call_id]))
            return None
        output = self.utility_output_handlers[call_id].result
        self.utility_output_handlers.pop(call_id, None)
        return output

    # pylint: disable=unused-argument
    async def migrate_out(
        self,
        dst_instance_actor: ray.actor.ActorHandle,
        dst_instance_context: InstanceContext,
        migration_type: Optional[MigrationType] = None
    ) -> List[RequestIDType]:
        if self.is_migrating:
            return []

        self.is_migrating = True

        call_id = self.untility_call('get_migrated_requests', migration_type)
        migrated_out_requests = await self.wait_untility_call_output(call_id)

        if migrated_out_requests is None or len(migrated_out_requests) == 0:
            self.is_migrating = False
            return []

        logger.info("Migrate out requests {} from {} to {}, migration_type: {}.".format(
            migrated_out_requests, self.instance_id, dst_instance_context.local_engine_id,
            migration_type
        ))

        migration_tasks = []
        finish_request_ids = []

        def do_migration_callback(request_id: RequestIDType, fut):
            ret = fut.result()[0]
            if isinstance(ret, Exception):
                log_instance_exception(ret, self.instance_id, "frontend_migration", request_id)
                self.untility_call('reset_request_migrating_state', request_id)
            else:
                finish_request_ids.append(request_id)

        for migrate_out_request in migrated_out_requests:
            target_request: EngineCoreRequest = req2corereq(migrate_out_request)
            migration_task = asyncio.gather(self.migration_frontend.migrate(target_request, dst_instance_context),
                                            return_exceptions=True)
            migration_task.add_done_callback(partial(do_migration_callback, migrate_out_request.request_id))
            migration_tasks.append(migration_task)

        await asyncio.gather(*migration_tasks)
        self.is_migrating = False

        return finish_request_ids
