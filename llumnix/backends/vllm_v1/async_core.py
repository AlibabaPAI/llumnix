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

from logging import DEBUG
from typing import Dict, Optional, Any, Tuple, Union
import signal
import asyncio
import queue

import vllm.envs as vllm_envs
from vllm.v1.engine.core import EngineCore, EngineCoreProc, DPEngineCoreProc, _core_init
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.utils import make_async
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.logger import init_logger
from vllm.config import ParallelConfig, VllmConfig
from vllm.v1.executor.abstract import Executor
from vllm.v1.engine.dpcoord import Participant

from llumnix.utils import get_ip_address

logger = init_logger(__name__)


class AsyncEngineCore(EngineCore):
    """Extension of EngineCore to add async methods."""

    async def execute_model_async(self, scheduler_output: SchedulerOutput):
        """
        The origin codes of vLLM (commit id: 6c01a):

        try:
            return self.model_executor.execute_model(scheduler_output)
        except SystemExit:
            raise
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(self.vllm_config, scheduler_output,
                                  self.scheduler.make_stats())
            raise err
        """
        try:
            return await self.model_executor.execute_model_async(scheduler_output)
        # pylint: disable=try-except-raise
        except SystemExit:
            raise
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(self.vllm_config, scheduler_output,
                                  self.scheduler.make_stats())
            raise err

    async def step_async(self) -> Tuple[Dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # pylint: disable=pointless-string-statement
        """
        The origin codes of vLLM (commit id: 6c01a):

        kvconn = self.scheduler.get_kv_connector()
        if kvconn:
            kvconn.step()

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()

        if scheduler_output.total_num_scheduled_tokens > 0:
            _dppart: Optional[Participant] = getattr(self, "_dppart", None)
            if _dppart is not None:
                _dppart.new_step()

        model_output = self.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)  # type: ignore

        return (engine_core_outputs,
                scheduler_output.total_num_scheduled_tokens > 0)
        """

        kvconn = self.scheduler.get_kv_connector()
        if kvconn:
            kvconn.step()

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()

        if scheduler_output.total_num_scheduled_tokens > 0:
            # pylint: disable=invalid-name
            _dppart: Optional[Participant] = getattr(self, "_dppart", None)
            if _dppart is not None:
                _dppart.new_step()

        model_output = await self.execute_model_async(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)  # type: ignore

        return (engine_core_outputs,
                scheduler_output.total_num_scheduled_tokens > 0)

    async def execute_dummy_batch_async(self):
        return await make_async(self.execute_dummy_batch)()


class AsyncEngineCoreProc(EngineCoreProc, AsyncEngineCore):
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
        """
        The original codes of vLLM (commit id: 6c01a):

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
        """

        self.input_queue = asyncio.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        self.engine_index = engine_index

        self.has_coordinator = False
        self._init_data_parallel(vllm_config)
        _core_init(self, vllm_config)

        AsyncEngineCore.__init__(self, vllm_config, executor_class, log_stats,
                                 executor_fail_callback, driver_tensor_queue_union)

        # NOTE(shejiarui): we don't launch vLLM's DPCoordinator
        self.has_coordinator = False
        self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)
        self.step_fn_async = self.step_async


    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        # pylint: disable=unused-argument
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: Optional[EngineCoreProc] = None
        try:
            parallel_config: ParallelConfig = kwargs[
                "vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = AsyncDPEngineCoreProc(*args, **kwargs)
            else:
                engine_core = AsyncEngineCoreProc(*args, **kwargs)

            asyncio.create_task(engine_core.run_busy_loop_async())

        # pylint: disable=try-except-raise
        except SystemExit:
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead() # pylint: disable=protected-access
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    async def run_busy_loop_async(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            await self._process_input_queue_async()
            # 2) Step the engine core and return the outputs.
            await self._process_engine_step_async()

    async def _process_input_queue_async(self):
        """
        Asynchronously processes the input queue.
        Exits when an engine step needs to be performed.
        """
        waited = False
        # Loop and wait for work if the engine has no pending requests.
        while not self.engines_running and not self.scheduler.has_requests():
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            # Change input_queue to asyncio.Queue, use await q.get()
            req = await self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any remaining requests that arrived while processing.
        # get_nowait() is non-blocking and works similarly for asyncio.Queue.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    async def _process_engine_step_async(self):
        """Called only when there are unfinished local requests."""
        # Step the engine core.
        outputs, model_executed = await self.step_fn_async()
        # Put EngineCoreOutputs into the output queue.
        for output in (outputs.items() if outputs else ()):
            self.output_queue.put_nowait(output)

        return model_executed


class AsyncDPEngineCoreProc(AsyncEngineCoreProc, DPEngineCoreProc):
    def __init__(self,
                 vllm_config: VllmConfig,
                 on_head_node: bool,
                 handshake_address: str,
                 executor_class: type[Executor],
                 log_stats: bool,):
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        AsyncEngineCoreProc.__init__(self, vllm_config, on_head_node, handshake_address,
                                     executor_class, log_stats, dp_rank)

    def _init_data_parallel(self, vllm_config: VllmConfig):
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

        # NOTE(shejiarui): code below will not be used in Llumnix.
        # from vllm.platforms import current_platform
        # device_control_env_var = current_platform.device_control_env_var
        # world_size = vllm_config.parallel_config.world_size
        # # Set CUDA_VISIBLE_DEVICES or equivalent.
        # try:
        #     os.environ[device_control_env_var] = ",".join(
        #         str(current_platform.device_id_to_physical_device_id(i))
        #         for i in range(local_dp_rank *
        #                        world_size, (local_dp_rank + 1) * world_size))
        # except IndexError as e:
        #     raise Exception(
        #         f"Error setting {device_control_env_var}: "
        #         f"local range: [{local_dp_rank * world_size}, "
        #         f"{(local_dp_rank + 1) * world_size}) "
        #         f"base value: \"{os.getenv(device_control_env_var)}\"") from e

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    async def _process_engine_step_async(self):
        forward_executed = await AsyncEngineCoreProc._process_engine_step_async(self)
        self._maybe_publish_request_counts()
        if forward_executed:
            return

        if self.engines_running:
            self._dppart.new_step()
            self.execute_dummy_batch()
        return
