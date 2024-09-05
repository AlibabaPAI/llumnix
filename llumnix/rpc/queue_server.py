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

import asyncio
from typing import (Coroutine, Any)
from typing_extensions import Never

import zmq
import zmq.asyncio
import cloudpickle

from llumnix.rpc.utils import (RPC_ZMQ_HWM, RPC_SUCCESS_STR, RPC_SOCKET_LIMIT_CUTOFF,
                               RPCPutNoWaitBatchQueueRequest, RPCUtilityRequest)
from llumnix.logger import init_logger

logger = init_logger(__name__)

class Empty(Exception):
    pass

class Full(Exception):
    pass


class QueueServer:
    def __init__(self, rpc_path: str, maxsize=0):
        self.context = zmq.asyncio.Context()

        # Maximum number of sockets that can be opened (typically 65536).
        # ZMQ_SOCKET_LIMIT (http://api.zeromq.org/4-2:zmq-ctx-get)
        socket_limit = self.context.get(zmq.constants.SOCKET_LIMIT)
        if socket_limit < RPC_SOCKET_LIMIT_CUTOFF:
            raise ValueError(
                f"Found zmq.constants.SOCKET_LIMIT={socket_limit}, which caps "
                "the number of concurrent requests Llumnix can process.")

        # We only have 1 ipc connection that uses unix sockets, so
        # safe to set MAX_SOCKETS to the zmq SOCKET_LIMIT (i.e. will
        # not run into ulimit issues)
        self.context.set(zmq.constants.MAX_SOCKETS, socket_limit)

        self.socket = self.context.socket(zmq.constants.ROUTER)
        self.socket.set_hwm(RPC_ZMQ_HWM)
        self.socket.bind(rpc_path)
        logger.info("QueueServer's socket bind to: {}".format(rpc_path))

        self.maxsize = maxsize
        self.queue = asyncio.Queue(maxsize)

    def cleanup(self):
        self.socket.close()
        self.context.destroy()

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()

    async def put(self, item, timeout=None):
        try:
            await asyncio.wait_for(self.queue.put(item), timeout)
        except asyncio.TimeoutError:
            raise Full

    async def get(self, timeout=None):
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            raise Empty

    def put_nowait(self, item):
        self.queue.put_nowait(item)

    def put_nowait_batch(self, items):
        # If maxsize is 0, queue is unbounded, so no need to check size.
        if self.maxsize > 0 and len(items) + self.qsize() > self.maxsize:
            raise Full(
                f"Cannot add {len(items)} items to queue of size "
                f"{self.qsize()} and maxsize {self.maxsize}."
            )
        for item in items:
            self.queue.put_nowait(item)

    def get_nowait(self):
        return self.queue.get_nowait()

    def get_nowait_batch(self, num_items):
        if num_items > self.qsize():
            raise Empty(
                f"Cannot get {num_items} items from queue of size " f"{self.qsize()}."
            )
        return [self.queue.get_nowait() for _ in range(num_items)]

    def _make_handler_coro(self, identity,
                           message) -> Coroutine[Any, Any, Never]:
        request = cloudpickle.loads(message)
        if request == RPCUtilityRequest.IS_SERVER_READY:
            return self._is_server_ready(identity)
        elif isinstance(request, RPCPutNoWaitBatchQueueRequest):
            return self._put_nowait_batch(identity, request)

    async def _is_server_ready(self, identity):
        await self.socket.send_multipart(
            [identity, cloudpickle.dumps(RPC_SUCCESS_STR)])

    async def _put_nowait_batch(self, identity, put_nowait_batch_queue_request: RPCPutNoWaitBatchQueueRequest):
        try:
            self.put_nowait_batch(put_nowait_batch_queue_request.items)
            await self.socket.send_multipart(
                [identity, cloudpickle.dumps(RPC_SUCCESS_STR)])
        except Exception as e:
            await self.socket.send_multipart([identity, cloudpickle.dumps(e)])

    async def run_server_loop(self):
        running_tasks = set()
        while True:
            identity, message = await self.socket.recv_multipart()
            task = asyncio.create_task(
                self._make_handler_coro(identity, message))
            # We need to keep around a strong reference to the task,
            # to avoid the task disappearing mid-execution as running tasks
            # can be GC'ed. Below is a common "fire-and-forget" tasks
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
            running_tasks.add(task)
            task.add_done_callback(running_tasks.discard)