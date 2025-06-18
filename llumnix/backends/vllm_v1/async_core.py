from typing import Optional, Any, Union
import signal
import asyncio
import queue
import threading

import zmq
import msgspec

from vllm.v1.engine.core import EngineCore, EngineCoreProc, DPEngineCoreProc
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType, disagg
from vllm.utils import make_async, make_zmq_socket
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

    async def step_async(self) -> EngineCoreOutputs:
        """Schedule, execute, and make output."""

        logger.debug("[step_async]")
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return EngineCoreOutputs(
                outputs=[],
                scheduler_stats=self.scheduler.make_stats(),
            )

        scheduler_output = self.scheduler.schedule()
        model_output = await self.execute_model_async(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)  # type: ignore

        return engine_core_outputs

    async def execute_dummy_batch_async(self):
        return await make_async(self.execute_dummy_batch)()


class AsyncEngineCoreProc(EngineCoreProc, AsyncEngineCore):    
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        vllm_config: VllmConfig,
        on_head_node: bool,
        input_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        engine_index: int = 0,
    ):
        input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()

        executor_fail_callback = lambda: input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        # Create input socket.
        input_ctx = zmq.Context()
        identity = engine_index.to_bytes(length=2, byteorder="little")
        input_socket = make_zmq_socket(input_ctx,
                                        input_address,
                                       zmq.DEALER,
                                       identity=identity,
                                       bind=False)
        try:
            # Register engine with front-end.
            output_address = self.startup_handshake(
                input_socket, on_head_node, vllm_config.parallel_config)

            # Update config which may have changed from the handshake.
            vllm_config.__post_init__()

            # Set up data parallel environment.
            self._init_data_parallel(vllm_config)

            # Initialize engine core and model.
            AsyncEngineCore.__init__(self, vllm_config, executor_class,
                                     log_stats, executor_fail_callback)

            self.step_fn = (self.step if self.batch_queue is None else
                            self.step_with_batch_queue)
            self.engines_running = False

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            input_socket.send(
                msgspec.msgpack.encode({
                    "status": "READY",
                    "local": on_head_node,
                    "num_gpu_blocks": num_gpu_blocks,
                }))

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            self.input_queue = input_queue
            self.output_queue = queue.Queue[Union[EngineCoreOutputs, bytes]]()
            threading.Thread(target=self.process_input_socket,
                             args=(input_socket, ),
                             daemon=True).start()
            input_socket = None
            self.output_thread = threading.Thread(
                target=self.process_output_socket,
                args=(output_address, engine_index),
                daemon=True)
            self.output_thread.start()
        finally:
            if input_socket is not None:
                input_socket.close(linger=0)

        disagg.init(vllm_config, self)

        self.step_fn_async = self.step_async

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
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = AsyncDPEngineCoreProc(*args, **kwargs)
            else:
                engine_core = AsyncEngineCoreProc(*args, **kwargs)

            asyncio.run(engine_core.run_busy_loop_async())

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
            logger.debug("[run_busy_loop_async] before _process_input_queue")
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            logger.debug("[run_busy_loop_async] before _process_engine_step_async")
            # 2) Step the engine core and return the outputs.
            await self._process_engine_step_async()
            logger.debug("[run_busy_loop_async] after _process_engine_step_async")

    async def _process_engine_step_async(self):
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs = await self.step_fn_async()
        # Put EngineCoreOutputs into the output queue.
        if outputs is not None:
            self.output_queue.put_nowait(outputs)


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
