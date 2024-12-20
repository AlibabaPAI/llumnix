import copy
import time
import asyncio
import ray

from vllm.engine.async_llm_engine import AsyncStream
from vllm import SamplingParams

from llumnix.logger import init_logger
from llumnix.entrypoints.setup import LlumnixEntrypointsContext
from llumnix.server_info import RequestTimestamps
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.server_info import ServerInfo

logger = init_logger(__name__)

WAIT_MANAGER_INTERVAL = 5


class LlumnixClientVLLM:
    def __init__(self,
                 llumnix_entrypoints_context: LlumnixEntrypointsContext):
        self.engine_manager: LLMEngineManager = llumnix_entrypoints_context.engine_manager
        self.instances: Dict[str, Llumlet] = llumnix_entrypoints_context.instances
        self.request_output_queue: QueueServerBase = llumnix_entrypoints_context.request_output_queue
        self.server_info: ServerInfo = llumnix_entrypoints_context.server_info
        self.log_requests: bool = llumnix_entrypoints_context.log_requests
        self.log_request_timestamps: bool = llumnix_entrypoints_context.log_request_timestamps

        self.request_streams: Dict[str, AsyncStream] = {}
        self.instance_num_requests: Dict[str, int] = {}
        for ins_id in self.instances.keys():
            self.instance_num_requests[ins_id] = 0
        self.num_finished_requests: int = 0
        self.manager_available: bool = True

    async def generate(self,
                       prompt: str,
                       sampling_params: SamplingParams,
                       request_id: str,
                       *args,
                       **kwargs) -> AsyncStream:
        if sampling_params.n > 1 or sampling_params.use_beam_search:
            raise ValueError("Unsupported feature: multiple sequence decoding")

        results_generator = AsyncStream(request_id)
        self.request_streams[request_id] = results_generator
        server_info_copy = copy.deepcopy(self.server_info)

        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            await self._generate_by_manager(request_id, server_info_copy, prompt, sampling_params, *args, **kwargs)
            self.manager_available = True
        except ray.exceptions.RayActorError:
            # Do not re-generate the request to avoid duplicate requests.
            if self.manager_available:
                self.manager_available = False
                return results_generator
            await self._generate_by_instance(request_id, server_info_copy, prompt, sampling_params, *args, **kwargs)

        return results_generator

    async def _generate_by_manager(self,
                                   request_id: str,
                                   server_info: ServerInfo,
                                   prompt: str,
                                   sampling_params: SamplingParams,
                                   *args,
                                   **kwargs) -> AsyncStream:
        if self.log_request_timestamps:
            # Hack request timestamps in server_info for latency breakdown.
            server_info.request_timestamps = RequestTimestamps()
            server_info.request_timestamps.api_server_manager_generate_timestamp = time.time()
        await self.engine_manager.generate.remote(request_id, server_info, prompt, sampling_params, *args, **kwargs)

    async def _generate_by_instance(self,
                                    request_id: str,
                                    server_info: ServerInfo,
                                    prompt: str,
                                    sampling_params: SamplingParams,
                                    *args,
                                    **kwargs) -> AsyncStream:
        try:
            if self.instance_num_requests:
                instance_id = min(self.instance_num_requests, key=self.instance_num_requests.get)
                self.instance_num_requests[instance_id] += 1
                await self.instances[instance_id].generate.remote(request_id, server_info, prompt, sampling_params, *args, **kwargs)
                logger.info("LLMEngineManager is unavailable temporarily, dispatch request {} to instance {}".format(
                    request_id, instance_id))
            else:
                logger.info("LLMEngineManager is unavailable temporarily, but there is no instance behind this api server, "
                    "sleep {}s, waiting for manager available".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                return await asyncio.create_task(self.generate(prompt, sampling_params, request_id, *args, **kwargs))
        except (ray.exceptions.RayActorError, KeyError):
            if instance_id in self.instances:
                logger.info("[manager_generate] instance {} is dead".format(instance_id))
                del self.instances[instance_id]
                del self.instance_num_requests[instance_id]
                return await asyncio.create_task(self.generate(prompt, sampling_params, request_id, *args, **kwargs))

    async def abort(self, request_id: str) -> None:
        try:
            logger.info("abort request: {}.".format(request_id))
            await self.engine_manager.abort.remote(request_id)
        except ray.exceptions.RayActorError:
            logger.info("manager is unavailable")

    async def is_ready(self) -> bool:
        ready_status = await self.engine_manager.is_ready.remote()
        return ready_status

    # TODO(s5u13b): Fix the potential output token out-of-order issue caused by the migration.
    async def get_request_outputs_loop(self):
        while True:
            request_outputs = await self.request_output_queue.get()
            for request_output in request_outputs:
                if hasattr(request_output, 'request_timestamps'):
                    request_output.request_timestamps.api_server_background_process_get_queue_timestamp = time.time()
            for request_output in request_outputs:
                request_id = request_output.request_id
                # Request could be dispatched twice when manager is dead, the first request will free the request_streams when finished.
                if request_id not in self.request_streams:
                    continue
                self.request_streams[request_id].put(request_output)
                if request_output.finished:
                    self.request_streams[request_id].finish()
                    del self.request_streams[request_id]
