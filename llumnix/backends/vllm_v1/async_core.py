from logging import DEBUG
from typing import Optional, Any, Union
import signal
import asyncio
import queue
import threading


from vllm.v1.engine.core import EngineCore, EngineCoreProc, DPEngineCoreProc
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.utils import make_async
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.logger import init_logger
from vllm.config import ParallelConfig, VllmConfig
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)


class AsyncEngineCore(EngineCore):
    """Extension of EngineCore to add async methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def execute_model_async(self, scheduler_output: SchedulerOutput):
        logger.debug("[execute_model_async]")
        try:
            return await self.model_executor.execute_model_async(scheduler_output)
        except BaseException as err:
            # NOTE: This method is exception-free
            dump_engine_exception(self.vllm_config, scheduler_output,
                                  self.scheduler.make_stats())
            # Re-raise exception
            raise err

    async def step_async(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output."""
        
        logger.debug("[step_async]")
        kvconn = self.scheduler.get_kv_connector()
        if kvconn:
            kvconn.step(self.scheduler)

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
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
        on_head_node: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        engine_index: int = 0,
    ):
        self.input_queue = asyncio.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        logger.debug("EngineCore handshake address: %s", handshake_address)
        with self._perform_handshake(handshake_address, identity, on_head_node,
                                     vllm_config) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            logger.debug("client_count: %s", str(self.client_count))
            self._init_data_parallel(vllm_config)

            AsyncEngineCore.__init__(self, vllm_config, executor_class, log_stats,
                             executor_fail_callback)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)
        self.step_fn_async = self.step_async

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


    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        logger.debug("[AsyncEngineCoreProc] run_engine_core")

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

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
                # TODO(zhaozhiyu): use EngineCoreLlumnix
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = AsyncDPEngineCoreProc(*args, **kwargs)
            else:
                engine_core = AsyncEngineCoreProc(*args, **kwargs)

            asyncio.create_task(engine_core.run_busy_loop_async())

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    async def run_busy_loop_async(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            logger.debug("[run_busy_loop_async] before _process_input_queue_async")
            # 1) Poll the input queue until there is work to do.
            await self._process_input_queue_async()
            logger.debug("[run_busy_loop_async] before _process_engine_step_async")
            # 2) Step the engine core and return the outputs.
            await self._process_engine_step_async()
            logger.debug("[run_busy_loop_async] after _process_engine_step_async")
            
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
            
            # Asynchronously get an item from the queue.
            # This will pause the task if the queue is empty, allowing the
            # event loop to run other tasks, without blocking the thread.
            req = await self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any remaining requests that arrived while processing.
        # get_nowait() is non-blocking and works similarly for asyncio.Queue.
        while not self.input_queue.empty():
            try:
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)
            except asyncio.QueueEmpty:
                # This can happen in concurrent scenarios; it's safe to just break.
                break

    async def _process_engine_step_async(self):
        """Called only when there are unfinished local requests."""
        # Step the engine core.
        outputs, model_executed = await self.step_fn_async()
        # Put EngineCoreOutputs into the output queue.
        for output in (outputs.items() if outputs else ()):
            self.output_queue.put_nowait(output)

        return model_executed


class AsyncDPEngineCoreProc(AsyncEngineCoreProc, DPEngineCoreProc):
    def __init__(self, *args, **kwargs):
        AsyncEngineCoreProc.__init__(self, *args, **kwargs)

    async def run_busy_loop_async(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()

            if local_unfinished_reqs:
                # 2) Step the engine core.
                await self._process_engine_step_async()

                # Check if we have now finished all requests.
                local_unfinished_reqs = (
                    self.scheduler.has_unfinished_requests())
            else:
                if self.scheduler.has_finished_requests():
                    # There are no unfinished requests, but there are some
                    # finished requests remaining to be removed from the
                    # batch state. This engine step won't perform a forward
                    # pass but will flush the finished requests to ensure
                    # up-to-date state is returned in the engine outputs.
                    await self._process_engine_step_async()

                if not self.engines_running:
                    # All engines are idle.
                    continue

                # There must be unfinished requests in DP peers, run a
                # dummy forward pass.
                await self.execute_dummy_batch_async()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs)

            if not self.engines_running:
                if self.local_dp_rank == 0:
                    # Notify client that we are pausing the loop.
                    logger.debug("Wave %d finished, pausing engine loop.",
                                 self.current_wave)
                    self.output_queue.put_nowait(
                        EngineCoreOutputs(wave_complete=self.current_wave))
                self.current_wave += 1
