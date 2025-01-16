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

import json
import time
import asyncio
import copy
import random
from typing import Dict

import ray

from blade_llm.service.communications.engine_client import MultiProcessingLLMClient
from blade_llm.service.communications.protocol import Stats
from blade_llm.service.communications.response import LLMResponse
from blade_llm.service.args import ServingArgs
from blade_llm.protocol import ServerRequest, GenerateStreamResponse
from blade_llm.service.communications.response import error_resp

from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.logging.logger import init_logger
from llumnix.constants import WAIT_MANAGER_INTERVAL

logger = init_logger(__name__)

# TODO(KuilongCui): Update LlumnixCient of BladeLLM.


class LlumnixClientBladeLLM(MultiProcessingLLMClient):
    def __init__(self, args: ServingArgs, llumnix_context: EntrypointsContext, loop: asyncio.AbstractEventLoop):
        super().__init__(args, -1)
        self.entrypoint_id2llumnix_id = {}
        self.llumnix_id2entrypoint_id = {}
        self.llumnix_context = llumnix_context
        self.request_streams: Dict[str, asyncio.Queue] = {}
        loop.create_task(self.background_process_outputs())

    async def background_process_outputs(self):
        while True:
            request_outputs = await self.llumnix_context.request_output_queue.get()
            if request_outputs is None:
                continue
            for (request_id, request_output) in request_outputs:
                request_output = GenerateStreamResponse(**json.loads(request_output))
                # Request could be dispatched twice when manager is dead, the first request will free the request_streams when finished.
                if request_id not in self.request_streams:
                    continue
                await self.request_streams[request_id].put(request_output)
                if request_output.is_finished:
                    logger.debug("client recv request output: {}".format(request_output))
                    del self.entrypoint_id2llumnix_id[self.llumnix_id2entrypoint_id[request_id]]
                    del self.llumnix_id2entrypoint_id[request_id]
                    del self.request_streams[request_id]

    async def _add_request(self, request: ServerRequest) -> LLMResponse:
        if request.sampling_params.n > 1 or request.sampling_params.use_beam_search:
            return error_resp(request.id, err_code=400, err_msg="Unsupported feature: multiple sequence decoding in Llumnix.")

        llumnix_id = random.randint(0, 2147483647) # 1<<31-1
        self.llumnix_id2entrypoint_id[str(llumnix_id)] = request.id
        self.entrypoint_id2llumnix_id[request.id] = llumnix_id
        request.id = llumnix_id
        resp_stream = await self._manager_generate(request.model_dump_json(), str(llumnix_id))
        return resp_stream

    async def _manager_generate(self, request, request_id: str) -> LLMResponse:
        logger.debug("client add request: {}:{}".format(request_id, request))

        results_queue = asyncio.Queue()
        self.request_streams[request_id] = results_queue

        # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            server_info_copy = copy.deepcopy(self.llumnix_context.server_info)
            if self.llumnix_context.log_request_timestamps:
                # Hack request timestamps in server_info for latency breakdown.
                server_info_copy.request_timestamps = RequestTimestamps()
                server_info_copy.request_timestamps.api_server_generate_timestamp = time.time()
            # await to catch exception
            await self.llumnix_context.manager.generate.remote(str(request_id), server_info_copy, server_request=request)
            self.llumnix_context.manager_available = True
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Error in manager generate: {}".format(e))
            # Do not re-generate the request to avoid duplicate requests.
            if self.llumnix_context.manager_available:
                self.llumnix_context.manager_available = False
                return LLMResponse(request_id, resp_queue=results_queue)
            try:
                if self.llumnix_context.instance_num_requests:
                    instance_id = min(self.llumnix_context.instance_num_requests, key=self.llumnix_context.instance_num_requests.get)
                    self.llumnix_context.instance_num_requests[instance_id] += 1
                    # TODO(Xinyi): set expected step here
                    await self.llumnix_context.instances[instance_id].generate.remote(request_id, server_info_copy, -1, request)
                    logger.info("Manager is unavailable, directly pass request {} to instance {}.".format(request_id, instance_id))
                else:
                    logger.info("Manager is unavailable, but there is no instance behind this api server, "
                        "sleep {}s, waiting for manager restarts.".format(WAIT_MANAGER_INTERVAL))
                    await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                    return await asyncio.create_task(self._manager_generate(request, request_id))
            except (ray.exceptions.RayActorError, KeyError):
                if instance_id in self.llumnix_context.instances:
                    logger.info("Instance {} is dead.".format(instance_id))
                    del self.llumnix_context.instances[instance_id]
                    del self.llumnix_context.instance_num_requests[instance_id]
                    return await asyncio.create_task(self._manager_generate(request, request_id))
        return LLMResponse(request_id, resp_queue=results_queue)

    async def drop_request(self, req_id: int):
        llumnix_id = self.entrypoint_id2llumnix_id.get(req_id, None)
        if llumnix_id:
            try:
                logger.info("Abort request: {}.".format(req_id))
                await self.llumnix_context.manager.abort.remote(str(req_id))
                self.entrypoint_id2llumnix_id.pop(req_id, None)
            except ray.exceptions.RayActorError:
                logger.info("Manager is unavailable.")

    def connect(self):
        pass

    def close(self):
        pass

    async def get_stats(self) -> Stats:
        pass

    async def get_metrics(self) -> str:
        pass

    async def start_profiler(self) -> None:
        pass

    async def stop_profiler(self) -> None:
        pass
