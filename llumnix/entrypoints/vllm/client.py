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
        results_generator = AsyncStream(request_id)
        self.request_streams[request_id] = results_generator

        if sampling_params.n > 1 or sampling_params.use_beam_search:
            raise ValueError("Unsupported feature: multiple sequence decoding")
        # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            server_info_copy = copy.deepcopy(self.server_info)
            if self.log_request_timestamps:
                # Hack request timestamps in server_info for latency breakdown.
                server_info_copy.request_timestamps = RequestTimestamps()
                server_info_copy.request_timestamps.api_server_manager_generate_timestamp = time.time()
            # await to catch exception
            await self.engine_manager.generate.remote(request_id, server_info_copy, prompt, sampling_params, *args, **kwargs)
            self.manager_available = True
        except ray.exceptions.RayActorError:
            # Do not re-generate the request to avoid duplicate requests.
            if self.manager_available:
                self.manager_available = False
                return results_generator
            try:
                if self.instance_num_requests:
                    instance_id = min(self.instance_num_requests, key=self.instance_num_requests.get)
                    self.instance_num_requests[instance_id] += 1
                    await self.instances[instance_id].generate.remote(request_id, server_info_copy, prompt, sampling_params)
                    logger.info("Manager is unavailable, directly pass request {} to instance {}".format(request_id, instance_id))
                else:
                    logger.info("Manager is unavailable, but there is no instance behind this api server, "
                        "sleep {}s, waiting for manager restarts".format(WAIT_MANAGER_INTERVAL))
                    await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                    return await asyncio.create_task(self.generate(prompt, sampling_params, request_id))
            except (ray.exceptions.RayActorError, KeyError):
                if instance_id in self.instances:
                    logger.info("[manager_generate] instance {} is dead".format(instance_id))
                    del self.instances[instance_id]
                    del self.instance_num_requests[instance_id]
                    return await asyncio.create_task(self.generate(prompt, sampling_params, request_id))

        return results_generator

    async def abort(self, request_id: str) -> None:
        try:
            logger.info("abort request: {}.".format(request_id))
            await self.engine_manager.abort.remote(request_id)
        except ray.exceptions.RayActorError:
            logger.info("manager is unavailable")

    async def is_ready(self) -> bool:
        ready_status = await self.engine_manager.is_ready.remote()
        return ready_status
