import asyncio
import time
from typing import Dict, List, Tuple, Optional
from queue import Queue
import threading

from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine, AsyncActorLLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.instance_info import InstanceInfo

logger = init_logger(__name__)

TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds


class EngineRequestInput:
    def __init__(
        self,
        request_id: str,
        # session_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        arrival_time,
        prompt_token_ids: Optional[List[int]] = None,
    ) -> None:
        self.request_id = request_id
        # self.session_id = session_id
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.prompt_token_ids = prompt_token_ids


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.

        *args, *kwargs: Arguments for LLMEngine.
    """

    def __init__(self,
                 instance_id: int,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 async_engine_actor: bool = False,
                 **kwargs) -> None:
        self.instance_id = instance_id
        self.instance_name = f"instance_{instance_id}"
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.async_engine_actor = async_engine_actor
        # self.progress_callback = progress_callback
        self.log_requests = log_requests
        if not self.engine_use_ray:
            engine_class = LLMEngine
        elif self.worker_use_ray:
            if self.async_engine_actor:
                engine_class = ray.remote(num_cpus=0, name=self.instance_name, max_concurrency=2)(AsyncActorLLMEngine).remote
            else:
                engine_class = ray.remote(num_cpus=0, name=self.instance_name)(LLMEngine).remote
        else:
            engine_class = ray.remote(num_gpus=1, name=self.instance_name)(LLMEngine).remote
        self.engine = engine_class(instance_id, *args, **kwargs)
        # Request id -> request output.
        self.request_outputs: Dict[str, RequestOutput] = {}
        # Request id -> event to notify that there is new output.
        self.request_events: Dict[str, asyncio.Event] = {}
        self.request_instance_info: Dict[str, InstanceInfo] = {}
        self.num_finished_request = 0
        self.num_finished_added_request = 0
        self.num_finished_migrated_request = 0
        self.is_engine_running = False
        self.is_engine_migrating = False
        self.kicking_request_id: Optional[str] = None
        # @@@: loop
        # asyncio.get_event_loop().create_task(self.background_loop())
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
    
    async def tokenize(self, prompt: str) -> Tuple[List[int], int]:
        prompt_token_ids, num_block = await self.engine.tokenize.remote(prompt)
        return prompt_token_ids, num_block

    async def engine_step(self, kicking_request_id: Optional[str] = None):
        """Kick the engine to process the waiting requests."""
        self.is_engine_running = True
        self.kicking_request_id = kicking_request_id
        if self.engine_use_ray:
            if self.async_engine_actor:
                request_outputs, instance_info = await self.engine.step_async.remote()
            else:
                request_outputs, instance_info = await self.engine.step.remote()
        else:
            # Yield to the event loop to allow other coroutines to run
            # while is_engine_running is True. This let the engine to add new
            # requests into the queue.
            await asyncio.sleep(0)
            request_outputs, instance_info = self.engine.step()
        self.is_engine_running = False
        self.kicking_request_id = None

        # Notify the waiting coroutines that there are new outputs ready.
        for request_output in request_outputs:
            request_id = request_output.request_id
            if request_id in self.request_events:
                self.request_outputs[request_id] = request_output
                self.request_instance_info[request_id] = instance_info
                self.request_events[request_id].set()
            else:
                # When engine step return new request, it means that this new request is migrated in from other instance.
                # Therefore, we call generate_migrate to kick engine following the new request's step-by-step generation process.
                logger.info(f"Instance {self.instance_id} migrate in request {request_id}.")
                asyncio.create_task(self.generate_migrate(request_id, request_output, instance_info))

    async def generate(
            self,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        arrival_time = time.time()

        # Create an event to notify us that there is new output from the
        # vLLM engine.
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        if self.log_requests:
            logger.info(f"Instance {self.instance_id} received request {request_id}.")
            # logger.info(f"Received request {request_id}: "
            #             f"prompt: {prompt!r}, "
            #             f"sampling params: {sampling_params}, "
            #             f"prompt token ids: {prompt_token_ids}.")

        # Add the request into the vLLM engine's waiting queue.
        if self.engine_use_ray:
            await self.engine.add_request.remote(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time)
        else:
            self.engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids=prompt_token_ids,
                                    arrival_time=arrival_time)

        # The vLLM engine does not have a background loop that keeps
        # processing incoming requests. Therefore, we need to keep kicking
        # the engine to process the requests.
        while True:
            if request_id not in self.request_events:
                # The request has been aborted.
                return

            # Kick the engine if the engine is not running.
            if not self.is_engine_running:
                try:
                    await self.engine_step(request_id)
                except RuntimeError as e:
                    await self.abort(request_id)
                    raise e

            # Wait for new output. The group_event will be set in engine_step
            # when there is new output available for the sequence group.
            # Added a timeout to prevent deadlock.
            try:
                await asyncio.wait_for(request_event.wait(),
                                       timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            except asyncio.TimeoutError:
                continue
            # Reset the event to wait for the next output.
            request_event.clear()

            # Decode and return new outputs.
            request_output = self.request_outputs[request_id]
            instance_info = self.request_instance_info[request_id]
            yield request_output, instance_info

            # Once finished, release the resources of the sequence group.
            if request_output.finished:
                if self.log_requests:
                    logger.info(f"Instance {self.instance_id} finished request {request_id}.")
                    self.num_finished_request += 1
                    self.num_finished_added_request += 1
                    logger.info(f"Instance {self.instance_id} self.num_finished_request {self.num_finished_request}.")
                    logger.info(f"Instance {self.instance_id} self.num_finished_added_request {self.num_finished_added_request}.")
                    logger.info(f"Instance {self.instance_id} self.num_finished_migrated_request {self.num_finished_migrated_request}.")

                del self.request_outputs[request_id]
                del self.request_events[request_id]
                del self.request_instance_info[request_id]
                # Kick the engine if the engine is not running. This is to
                # prevent that there are still requests in engine's waiting
                # queue to be executed.
                if not self.is_engine_running:
                    await self.engine_step()
                break

    # Callback version of generate. Compared with original generate, generate_callback does not directly return output to manager,
    # it put output to output queue and callback the manager to fetch the output from output queue.
    async def generate_callback(
            self,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None) -> None:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        arrival_time = time.time()

        # Create an event to notify us that there is new output from the
        # vLLM engine.
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event
        self.request_outputs[request_id] = None
        self.request_instance_info[request_id] = None

        if self.log_requests:
            logger.info(f"Instance {self.instance_id} received request {request_id}.")
            # logger.info(f"Received request {request_id}: "
            #             f"prompt: {prompt!r}, "
            #             f"sampling params: {sampling_params}, "
            #             f"prompt token ids: {prompt_token_ids}.")

        # Add the request into the vLLM engine's waiting queue.
        if self.engine_use_ray:
            await self.engine.add_request.remote(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time)
        else:
            self.engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids=prompt_token_ids,
                                    arrival_time=arrival_time)

        # The vLLM engine does not have a background loop that keeps
        # processing incoming requests. Therefore, we need to keep kicking
        # the engine to process the requests.
        while True:
            if request_id not in self.request_events:
                # The request has been aborted.
                return

            # Kick the engine if the engine is not running.
            if not self.is_engine_running:
                try:
                    await self.engine_step(request_id)
                except RuntimeError as e:
                    await self.abort(request_id)
                    raise e

            # Wait for new output. The group_event will be set in engine_step
            # when there is new output available for the sequence group.
            # Added a timeout to prevent deadlock.
            try:
                await asyncio.wait_for(request_event.wait(),
                                       timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            except asyncio.TimeoutError:
                continue
            # Reset the event to wait for the next output.
            request_event.clear()

            # Decode and return new outputs.
            request_output = self.request_outputs[request_id]

            if request_output.migrated:
                if self.log_requests:
                    logger.info(f"Instance {self.instance_id} migrate out request {request_id}.")

                del self.request_events[request_id]
                del self.request_outputs[request_id]
                del self.request_instance_info[request_id]
                # Kick the engine if the engine is not running. This is to
                # prevent that there are still requests in engine's waiting
                # queue to be executed.
                if not self.is_engine_running:
                    await self.engine_step()
                break

            instance_info = self.request_instance_info[request_id]
            self.output_queue.put_nowait((request_output, instance_info))
            asyncio.create_task(self.manager_obj.progress_callback(self.instance_id))

            # Once finished, release the resources of the sequence group.
            if request_output.finished:
                if self.log_requests:
                    logger.info(f"Instance {self.instance_id} finished request {request_id}.")
                    self.num_finished_request += 1
                    self.num_finished_added_request += 1
                    logger.info(f"Instance {self.instance_id} self.num_finished_request {self.num_finished_request}.")
                    logger.info(f"Instance {self.instance_id} self.num_finished_added_request {self.num_finished_added_request}.")
                    logger.info(f"Instance {self.instance_id} self.num_finished_migrated_request {self.num_finished_migrated_request}.")

                del self.request_outputs[request_id]
                del self.request_events[request_id]
                del self.request_instance_info[request_id]
                # Kick the engine if the engine is not running. This is to
                # prevent that there are still requests in engine's waiting
                # queue to be executed.
                if not self.is_engine_running:
                    await self.engine_step()
                break

    # Migrate version of generate. It is called by the engine_step when it finds new request which is migrated from other instance.
    # Compared with generate_callback, it does not add request to engine.
    async def generate_migrate(
            self,
            request_id: str,
            request_output: RequestOutput = None,
            instance_info: InstanceInfo = None) -> None:
        # Create an event to notify us that there is new output from the
        # vLLM engine.
        if request_output and request_output.finished:
            self.output_queue.put_nowait((request_output, instance_info))
            asyncio.create_task(self.manager_obj.progress_callback(self.instance_id))
            return
        if request_id in self.request_events.keys():
            return
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event
        self.request_outputs[request_id] = request_output
        self.request_instance_info[request_id] = instance_info
        logger.info(f"Instance {self.instance_id} migrate in request {request_id}")
        # The vLLM engine does not have a background loop that keeps
        # processing incoming requests. Therefore, we need to keep kicking
        # the engine to process the requests.
        while True:
            if request_id not in self.request_events:
                # The request has been aborted.
                return

            # Kick the engine if the engine is not running.
            if not self.is_engine_running:
                try:
                    await self.engine_step(request_id)
                except RuntimeError as e:
                    await self.abort(request_id)
                    raise e

            # Wait for new output. The group_event will be set in engine_step
            # when there is new output available for the sequence group.
            # Added a timeout to prevent deadlock.
            try:
                await asyncio.wait_for(request_event.wait(),
                                       timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            except asyncio.TimeoutError:
                continue
            # Reset the event to wait for the next output.
            request_event.clear()

            # Decode and return new outputs.
            request_output = self.request_outputs[request_id]

            if request_output.migrated:
                if self.log_requests:
                    logger.info(f"Instance {self.instance_id} migrate out request {request_id}.")

                del self.request_events[request_id]
                del self.request_outputs[request_id]
                del self.request_instance_info[request_id]
                # Kick the engine if the engine is not running. This is to
                # prevent that there are still requests in engine's waiting
                # queue to be executed.
                if not self.is_engine_running:
                    await self.engine_step()
                break

            instance_info = self.request_instance_info[request_id]
            self.output_queue.put_nowait((request_output, instance_info))
            asyncio.create_task(self.manager_obj.progress_callback(self.instance_id))

            # Once finished, release the resources of the sequence group.
            if request_output.finished:
                if self.log_requests:
                    logger.info(f"Instance {self.instance_id} finished request {request_id}.")
                    self.num_finished_request += 1
                    self.num_finished_migrated_request += 1
                    logger.info(f"Instance {self.instance_id} self.num_finished_request {self.num_finished_request}.")
                    logger.info(f"Instance {self.instance_id} self.num_finished_added_request {self.num_finished_added_request}.")
                    logger.info(f"Instance {self.instance_id} self.num_finished_migrated_request {self.num_finished_migrated_request}.")
                
                del self.request_outputs[request_id]
                del self.request_events[request_id]
                del self.request_instance_info[request_id]
                # Kick the engine if the engine is not running. This is to
                # prevent that there are still requests in engine's waiting
                # queue to be executed.
                if not self.is_engine_running:
                    await self.engine_step()
                break

    # A background loop of AsyncLLMEngine. 
    # The background loop keep on fetching input queue, kicking engine and putting output queue.
    async def background_loop(self):
        self.is_engine_running = False
        while True:
            # await asyncio.sleep(0.01)
            while not self.input_queue.empty():
                self.is_engine_running = True
                request_input = self.input_queue.get_nowait()
                if self.log_requests:
                    logger.info(f"Instance {self.instance_id} Received request {request_input.request_id}.")
                    # logger.info(f"Received request {request_input.request_id}: "
                    #             f"prompt: {request_input.prompt!r}, "
                    #             f"sampling params: {request_input.sampling_params}, "
                    #             f"prompt token ids: {request_input.prompt_token_ids}.")
                # Add the request into the vLLM engine's waiting queue.
                await self.engine.add_request.remote(
                    request_input.request_id,
                    request_input.prompt,
                    request_input.sampling_params,
                    prompt_token_ids=request_input.prompt_token_ids,
                    arrival_time=request_input.arrival_time)
            
            if self.is_engine_running:
                request_outputs, instance_info = await self.engine.step.remote()

            if len(request_outputs) == 0:
                self.is_engine_running = False

            # needs_callback_progress = False
            for request_output in request_outputs:
                # if not request_output.finished:
                #     continue
                # needs_callback_progress = True
                assert len(request_output.outputs) == 1
                self.output_queue.put_nowait((request_output, instance_info))
            # if needs_callback_progress:
            #     asyncio.create_task(self.manager_obj.progress_callback(self.instance_id))
    
    async def input_queue_put_nowait(self, request_input: EngineRequestInput) -> None:
        self.input_queue.put_nowait(request_input)
    
    async def output_queue_empty(self) -> bool:
        return self.output_queue.empty()
    
    async def output_queue_get_nowait(self) -> Tuple[RequestOutput, InstanceInfo]:
        return self.output_queue.get_nowait()

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return

        if self.log_requests:
            logger.info(f"Instance {self.instance_id} Aborted request {request_id}.")

        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_id)
        else:
            self.engine.abort_request(request_id)

        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.request_outputs:
            del self.request_outputs[request_id]

        # To prevent deadlock when a request is aborted while the engine is
        # running.
        if self.kicking_request_id == request_id:
            self.is_engine_running = False
            self.kicking_request_id = None

    async def abort_loop(self, request_id: str) -> None:
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_id)
        else:
            self.engine.abort_request(request_id)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    # async def migrate_out(self, dst_instance_name: str, rank_offset: int):
    #     # migrate_out_handshake will call migrate_in_handshake inside
    #     migrating_requests = await self.engine.migrate_out_handshake.remote(dst_instance_name)
    #     # Let the generation coroutines of migrating_requests to exit.
    #     for migrating_request_id in migrating_requests:
    #         self.request_outputs[migrating_request_id] = RequestOutput(None, None, None, None, None)
    #         self.request_outputs[migrating_request_id].migrated = True
    #         self.request_events[migrating_request_id].set()
    #     # migrate_out_communicate will call migrate_in_communicate inside
    #     if migrating_requests:
    #         await self.engine.migrate_out_communicate.remote(rank_offset)

    async def migrate_out(self, dst_instance_name: str, rank_offset: int):
        if self.async_engine_actor:
            migrating_requests = await self.engine.migrate_out_multistage.remote(dst_instance_name, rank_offset)
        else:
            migrating_requests = await self.engine.migrate_out.remote(dst_instance_name, rank_offset)
        # Let the generation coroutines of migrating_requests to exit.
        for migrating_request_id in migrating_requests:
            if migrating_request_id in self.request_events:
                self.request_outputs[migrating_request_id] = RequestOutput(None, None, None, None, None, None, None)
                self.request_outputs[migrating_request_id].migrated = True
                self.request_events[migrating_request_id].set()
        return migrating_requests
    
    @classmethod
    def from_engine_args(cls,
                         engine_args: AsyncEngineArgs) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(engine_args.worker_use_ray,
                     engine_args.engine_use_ray,
                     engine_configs[0],
                     engine_configs[1],
                     engine_configs[2],
                     engine_configs[3],
                     distributed_init_method,
                     placement_group,
                     log_requests=not engine_args.disable_log_requests_engine,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def register_callback(self, manager_obj, progress_callback):
        self.manager_obj = manager_obj
        self.progress_callback = progress_callback
    
    async def is_ready(self):
        if self.engine_use_ray:
            return await self.engine.is_ready.remote()
        else:
            return self.engine.is_ready()
    
    async def shutdown_engine(self, do_migrate:bool = None):
        migrating_requests_list = await self.engine.shutdown_workers.remote(do_migrate)
        for migrating_request_id, _ in migrating_requests_list:
            if migrating_request_id in self.request_events:
                self.request_outputs[migrating_request_id] = RequestOutput(None, None, None, None, None, None, None)
                self.request_outputs[migrating_request_id].migrated = True
                self.request_events[migrating_request_id].set()
        return migrating_requests_list
    
    async def restart_engine(self):
        await self.engine.restart_workers.remote()
    
    async def get_num_unfinished_requests(self):
        return await self.engine.get_num_unfinished_requests.remote()

    async def stop_shutdown(self):
        await self.engine.stop_shutdown.remote()
