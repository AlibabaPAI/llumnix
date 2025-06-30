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
from typing import Any, Dict
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
    get_zmq_socket_name,
)
from llumnix.constants import ZMQ_RPC_TIMEOUT, ZMQ_IO_THREADS
from llumnix.metrics.timestamps import set_timestamp

logger = init_logger(__name__)


class SocketConnection:
    def __init__(self, socket: zmq.asyncio.Socket):
        self.socket: zmq.asyncio.Socket = socket
        self.lock = asyncio.Lock()  # Ensures exclusive access to the socket across coroutines.

    async def __aenter__(self):
        await self.lock.acquire()
        return self.socket

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.lock.release()

    def close(self):
        self.socket.close(linger=0)


class ZmqSocketFactory:

    def __init__(self, context: zmq.asyncio.Context):
        self.context: zmq.asyncio.Context = context
        self.socket_connections: Dict[str, SocketConnection] = {}

    def get_socket(self, ip, port) -> SocketConnection:
        dst_name = get_zmq_socket_name(ip, port)
        if dst_name not in self.socket_connections:
            socket = self.context.socket(zmq.DEALER)
            dst_address = get_open_zmq_ipc_path(ip, port)
            socket.connect(dst_address)
            socket_connection = SocketConnection(socket)
            self.socket_connections[dst_name] = socket_connection
        return self.socket_connections[dst_name]

    def close_socket(self, ip, port):
        dst_name = get_zmq_socket_name(ip, port)
        if dst_name in self.socket_connections:
            self.socket_connections[dst_name].close()
            del self.socket_connections[dst_name]

    def close_all_sockets(self):
        for socket_connection in self.socket_connections.values():
            socket_connection.close()
        self.socket_connections.clear()


class ZmqClient(QueueClientBase):
    def __init__(self):
        super().__init__()
        self.context = zmq.asyncio.Context(ZMQ_IO_THREADS)
        self.socket_factory: ZmqSocketFactory = ZmqSocketFactory(context=self.context)

    def close(self):
        self.socket_factory.close_all_sockets()
        self.context.destroy()

    async def _send_one_way_rpc_request(
        self, request: RPC_REQUEST_TYPE, ip: str, port: int, error_message: str
    ):
        async def do_rpc_call(socket: zmq.asyncio.Socket, request: RPC_REQUEST_TYPE):
            if isinstance(
                request, (RPCPutNoWaitQueueRequest, RPCPutNoWaitBatchQueueRequest)
            ) and self.need_record_latency():
                request.send_time = time.perf_counter()

            await socket.send_multipart([cloudpickle.dumps(request)])

            if await socket.poll(timeout=(ZMQ_RPC_TIMEOUT * 1000)) == 0:
                raise TimeoutError(
                    f"Server didn't reply within {ZMQ_RPC_TIMEOUT * 1000} ms"
                )

            return cloudpickle.loads(await socket.recv())

        try:
            socket_connection = self.socket_factory.get_socket(ip, port)
            async with socket_connection as socket:
                response = await do_rpc_call(socket, request)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Failed to send one way rpc request, exception: {}.".format(e))
            response = e
        if not isinstance(response, str) or response != RPC_SUCCESS_STR:
            # close the socket if something wrong
            self.socket_factory.close_socket(ip, port)
            if isinstance(response, Exception):
                logger.error(error_message)
                raise response
            raise ValueError(error_message)

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
                        # ip=server_info.request_output_queue_ip,
                        ip="127.0.0.1", # TODO(shejiarui) fix this
                        port=server_info.request_output_queue_port,
                        error_message="Unable to put items into queue.")

    async def put_nowait_batch(self, items: Iterable, server_info: ServerInfo):
        set_timestamp(items, 'queue_client_send_timestamp', time.time())
        await self._send_one_way_rpc_request(
                        request=RPCPutNoWaitBatchQueueRequest(items=items),
                        ip=server_info.request_output_queue_ip,
                        port=server_info.request_output_queue_port,
                        error_message="Unable to put items into queue.")
