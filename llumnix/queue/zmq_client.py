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

from typing import Any, Dict
from asyncio import Lock, Queue, QueueEmpty
import asyncio
from collections.abc import Iterable
import time

import zmq
import zmq.asyncio
import cloudpickle

from llumnix.logging.logger import init_logger
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.server_info import ServerInfo

from llumnix.queue.zmq_utils import (
    RPC_SUCCESS_STR,
    RPC_REQUEST_TYPE,
    RPCUtilityRequest,
    RPCPutNoWaitQueueRequest,
    RPCPutNoWaitBatchQueueRequest,
    get_open_zmq_ipc_path,
    get_zmq_pool_name,
)
from llumnix.constants import (
    RPC_GET_DATA_TIMEOUT_MS,
    SOCKET_POOL_MAXSIZE
)
from llumnix.metrics.timestamps import set_timestamp

logger = init_logger(__name__)


class AsyncZmqSocketPool:
    def __init__(self, context: zmq.asyncio.Context, ip, port, pool_size=10):
        self.context = context
        self.ip = ip
        self.port = port

        # max size of the pool
        # sockets in the pool will be long live
        # sockets out the poll will be closed after used
        self.pool_size = pool_size

        self.pool = Queue(maxsize=pool_size)

    async def _create_socket(self) -> zmq.asyncio.Socket:
        socket = self.context.socket(zmq.DEALER)
        dst_address = get_open_zmq_ipc_path(self.ip, self.port)
        socket.connect(dst_address)
        return socket

    async def get_socket(self) -> zmq.asyncio.Socket:
        socket = None
        try:
            socket = self.pool.get_nowait()
        except QueueEmpty:
            socket = await self._create_socket()
        return socket

    def full(self):
        return self.pool.full()

    def put(self, socket: zmq.asyncio.Socket):
        self.pool.put_nowait(socket)

    async def close_all_connections(self):
        while not self.pool.empty():
            socket = await self.pool.get()
            socket.close(linger=0)


class ZmqSocketPoolFactory:

    def __init__(self, context: zmq.asyncio.Context, pool_size=10):
        self.context: zmq.asyncio.Context = context
        self.pool_size: int = pool_size
        self.pools: Dict[str, AsyncZmqSocketPool] = {}
        self.lock: asyncio.Lock = Lock()

    async def get_pool(self, ip, port) -> AsyncZmqSocketPool:
        async with self.lock:
            dst_name = get_zmq_pool_name(ip, port)
            if dst_name not in self.pools:
                self.pools[dst_name] = AsyncZmqSocketPool(
                    ip=ip,
                    port=port,
                    context=self.context,
                    pool_size=self.pool_size,
                )
            return self.pools[dst_name]

    async def close_all_pools(self):
        for pool in self.pools.values():
            await pool.close_all_connections()


class ZmqClient(QueueClientBase):
    def __init__(self):
        self.context = zmq.asyncio.Context(8)
        self.socket_pool_factory: ZmqSocketPoolFactory = ZmqSocketPoolFactory(
            context=self.context, pool_size=SOCKET_POOL_MAXSIZE
        )
        self.zmq_timeout_ms: int = RPC_GET_DATA_TIMEOUT_MS
        self._conn_lock: asyncio.Lock = Lock()

    # This function is not called explicitly.
    def close(self):
        self.socket_pool_factory.close_all_pools()
        self.context.destroy()

    async def _send_one_way_rpc_request(
            self,
            request: RPC_REQUEST_TYPE,
            ip: str,
            port: int,
            error_message: str):
        async def do_rpc_call(socket: zmq.asyncio.Socket,
                              request: RPC_REQUEST_TYPE):

            await socket.send_multipart([cloudpickle.dumps(request)])

            if await socket.poll(timeout=self.zmq_timeout_ms) == 0:
                raise TimeoutError("Server didn't reply within "
                                    f"{self.zmq_timeout_ms} ms")

            return cloudpickle.loads(await socket.recv())

        socket_pool = await self.socket_pool_factory.get_pool(ip, port)
        socket = await socket_pool.get_socket()
        response = await do_rpc_call(socket, request)

        if not isinstance(response, str) or response != RPC_SUCCESS_STR:
            # close the socket if something wrong
            socket.close(linger=0)
            if isinstance(response, Exception):
                logger.error(error_message)
                raise response
            raise ValueError(error_message)

        # drop or put back the socket
        if not socket_pool.full():
            socket_pool.put(socket)
        else:
            socket.close(linger=0)

    async def wait_for_server_rpc(self,
                                  server_info: ServerInfo):
        await self._send_one_way_rpc_request(
                        request=RPCUtilityRequest.IS_SERVER_READY,
                        ip=server_info.request_output_queue_ip,
                        port=server_info.request_output_queue_port,
                        error_message="Unable to start RPC Server")

    async def put_nowait(self, item: Any, server_info: ServerInfo):
        set_timestamp(item, 'queue_client_send_timestamp', time.time())
        await self._send_one_way_rpc_request(
                        request=RPCPutNoWaitQueueRequest(item=item),
                        ip=server_info.request_output_queue_ip,
                        port=server_info.request_output_queue_port,
                        error_message="Unable to put items into queue.")

    async def put_nowait_batch(self, items: Iterable, server_info: ServerInfo):
        set_timestamp(items, 'queue_client_send_timestamp', time.time())
        await self._send_one_way_rpc_request(
                        request=RPCPutNoWaitBatchQueueRequest(items=items),
                        ip=server_info.request_output_queue_ip,
                        port=server_info.request_output_queue_port,
                        error_message="Unable to put items into queue.")
