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
from typing import Dict, Tuple

import ray

from blade_llm.service.communications.engine_client import MultiProcessingLLMClient
from blade_llm.service.communications.protocol import Stats, GenerateStreamMessage
from blade_llm.service.communications.response import LLMResponse
from blade_llm.service.args import ServingArgs
from blade_llm.protocol import ServerRequest
from blade_llm.service.communications.response import error_resp

from llumnix.metrics.timestamps import RequestTimestamps
from llumnix.entrypoints.utils import EntrypointsContext
from llumnix.logging.logger import init_logger
from llumnix.constants import WAIT_MANAGER_INTERVAL
from llumnix.metrics.timestamps import set_timestamp
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_server_base import QueueServerBase
from llumnix.server_info import ServerInfo

logger = init_logger(__name__)


class LlumnixClientBladeLLM(MultiProcessingLLMClient):
    def __init__(self,
                 args: ServingArgs,
                 entrypoints_context: EntrypointsContext,
                 loop: asyncio.AbstractEventLoop):
        super().__init__(args, -1, -1)
        self.entrypoint_id2llumnix_id = {} # int32 -> int32
        self.llumnix_id2entrypoint_id = {} # int32 -> int32

        self.manager: "Manager" = entrypoints_context.manager
        self.instances: Dict[str, Llumlet] = entrypoints_context.instances
        self.request_output_queue: QueueServerBase = entrypoints_context.request_output_queue
        self.server_info: ServerInfo = entrypoints_context.server_info
        self.log_requests: bool = entrypoints_context.log_requests
        self.log_request_timestamps: bool = entrypoints_context.log_request_timestamps

        self.request_streams: Dict[int, asyncio.Queue] = {}
        self.instance_num_requests: Dict[str, int] = {}
        for ins_id in self.instances.keys():
            self.instance_num_requests[ins_id] = 0
        self.num_finished_requests = 0
        self.manager_available = True

        loop.create_task(self.get_request_outputs_loop())

    async def _add_request(self, request: ServerRequest) -> LLMResponse:
        if request.sampling_params.n > 1 or request.sampling_params.use_beam_search:
            return error_resp(request.id, err_code=400, err_msg="Unsupported feature: multiple sequence decoding in Llumnix.")

        # To prevent different api_servers from generating the same request_id, a random number is used to replace original req_id.
        llumnix_id = random.randint(1, (1 << 31) - 1)
        self.llumnix_id2entrypoint_id[llumnix_id] = request.id
        self.entrypoint_id2llumnix_id[request.id] = llumnix_id
        logger.info("request id is replaced from [{},{}] to {}".format(request.id, request.external_id, llumnix_id))
        request.id = llumnix_id

        resp_stream = await self._generate(llumnix_id, request.model_dump_json())
        return resp_stream

    async def _generate(self, request_id: int, request: ServerRequest) -> LLMResponse:
        logger.info("Client add request: {}:{}".format(request_id, request))
        results_queue = asyncio.Queue()
        self.request_streams[request_id] = results_queue
        server_info_copy = copy.deepcopy(self.server_info)

        # This request's outputs will be put to the request_output_queue of this api server no matter which instance it's running in.
        # If manager is unavailable, request will be directly added to the llumlet held by api server.
        try:
            await self._generate_by_manager(request_id, server_info_copy, request)
            self.manager_available = True
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Error in manager generate: {}".format(e))
            # Do not re-generate the request to avoid duplicate requests.
            if self.manager_available:
                self.manager_available = False
                return LLMResponse(request_id, resp_queue=results_queue)
            await self._generate_by_instance(request_id, server_info_copy, request)
        return LLMResponse(request_id, resp_queue=results_queue)

    async def _generate_by_manager(self, request_id: int, server_info: ServerInfo, request: ServerRequest):
        if self.log_request_timestamps:
            # Hack request timestamps in server_info for latency breakdown.
            server_info.request_timestamps = RequestTimestamps()
            set_timestamp(server_info, "api_server_generate_timestamp", time.time())
        # await to catch exception
        await self.manager.generate.remote(str(request_id), server_info, server_request=request)

    async def _generate_by_instance(self, request_id: int, server_info: ServerInfo, request: ServerRequest):
        try:
            if self.instance_num_requests:
                instance_id = min(self.instance_num_requests, key=self.instance_num_requests.get)
                self.instance_num_requests[instance_id] += 1
                await self.instances[instance_id].generate.remote(request_id, server_info, -1, request)
                logger.info("Manager is unavailable, directly pass request {} to instance {}.".format(request_id, instance_id))
            else:
                logger.info("Manager is unavailable, but there is no instance behind this api server, "
                    "sleep {}s, waiting for manager restarts.".format(WAIT_MANAGER_INTERVAL))
                await asyncio.sleep(WAIT_MANAGER_INTERVAL)
                return await asyncio.create_task(self._generate(request_id, request))
        except (ray.exceptions.RayActorError, KeyError):
            if instance_id in self.instances:
                logger.info("Instance {} is dead.".format(instance_id))
                del self.instances[instance_id]
                del self.instance_num_requests[instance_id]
                return await asyncio.create_task(self._generate(request_id, request))

    async def drop_request(self, req_id: int):
        llumnix_id = self.entrypoint_id2llumnix_id.get(req_id, None)
        if llumnix_id:
            try:
                logger.info("Abort request: {}.".format(req_id))
                self.manager.abort.remote(str(req_id))
                self.entrypoint_id2llumnix_id.pop(req_id, None)
            except ray.exceptions.RayActorError:
                logger.info("Manager is unavailable.")

    async def is_ready(self) -> bool:
        ready_status = await self.manager.is_ready.remote()
        return ready_status

    async def get_request_outputs_loop(self):
        while True:
            request_output_jsons = await self.request_output_queue.get()
            if request_output_jsons is None:
                continue
            for request_output_json in request_output_jsons:
                output_message = GenerateStreamMessage(**json.loads(request_output_json))
                request_id = output_message.req_id
                request_output = output_message.resp
                # Request could be dispatched twice when manager is dead, the first request will free the request_streams when finished.
                if request_id not in self.request_streams:
                    continue
                self.request_streams[request_id].put_nowait(request_output)

                if request_output.is_finished:
                    # TODO(KuilongCui): fix logging output error
                    print("Client finish request output: {}".format(output_message), flush=True)
                    logger.debug("Client finish request output: {}".format(output_message))
                    del self.entrypoint_id2llumnix_id[self.llumnix_id2entrypoint_id[request_id]]
                    del self.llumnix_id2entrypoint_id[request_id]
                    del self.request_streams[request_id]

    def connect(self):
        pass

    def close(self):
        pass

    async def get_stats(self) -> Stats:
        pass

    async def get_metrics(self) -> str:
        pass

    def support_beam_search(self) -> Tuple[bool, str]:
        return False, "Unsupported feature: multiple sequence decoding in Llumnix."

    async def start_profiler(self) -> None:
        pass

    async def stop_profiler(self) -> None:
        pass
