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

from typing import Any
from contextlib import contextmanager
from collections.abc import Iterable
import time

import zmq
import zmq.asyncio
import cloudpickle

from llumnix.logging.logger import init_logger
from llumnix.server_info import ServerInfo

from llumnix.queue.zmq_utils import (RPC_SUCCESS_STR, RPC_REQUEST_TYPE, RPCClientClosedError,
                                     RPCUtilityRequest, RPCPutNoWaitQueueRequest, RPCPutNoWaitBatchQueueRequest,
                                     get_open_zmq_ipc_path)
from llumnix.constants import RPC_GET_DATA_TIMEOUT_MS, RPC_SOCKET_LIMIT_CUTOFF, RPC_ZMQ_HWM
from llumnix.metrics.timestamps import set_timestamp

logger = init_logger(__name__)


class ZmqClient:
    def __init__(self):
        self.context = zmq.asyncio.Context()
        self._data_timeout = RPC_GET_DATA_TIMEOUT_MS
        self._errored = False

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

    # This function is not called explicitly.
    def close(self):
        self.context.destroy()

    @contextmanager
    def to_socket(self, rpc_path):
        # Raise a sensible error if the client was already closed.
        # This can happen if a server shutdown is triggered but some coroutines
        # are still running requests.
        # There should not be a race condition with this check because we don't
        # yield to the event loop between here and opening the socket.
        if self.context.closed:
            raise RPCClientClosedError("The ZMQ client has already shut down")

        # Note that we use DEALER to enable asynchronous communication
        # to enable streaming.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.set_hwm(RPC_ZMQ_HWM)
        try:
            socket.connect(rpc_path)
            yield socket
        finally:
            socket.close(linger=0)

    async def _send_one_way_rpc_request(
            self,
            request: RPC_REQUEST_TYPE,
            rpc_path: str,
            error_message: str):
        async def do_rpc_call(socket: zmq.asyncio.Socket,
                              request: RPC_REQUEST_TYPE):

            await socket.send_multipart([cloudpickle.dumps(request)])

            if await socket.poll(timeout=self._data_timeout) == 0:
                raise TimeoutError("Server didn't reply within "
                                   f"{self._data_timeout} ms")

            return cloudpickle.loads(await socket.recv())

        with self.to_socket(rpc_path) as socket:
            response = await do_rpc_call(socket, request)

        if not isinstance(response, str) or response != RPC_SUCCESS_STR:
            if isinstance(response, Exception):
                logger.error(error_message)
                raise response
            raise ValueError(error_message)

    async def wait_for_server_rpc(self,
                                  server_info: ServerInfo):
        rpc_path = get_open_zmq_ipc_path(server_info.request_output_queue_ip, server_info.request_output_queue_port)
        await self._send_one_way_rpc_request(
                        request=RPCUtilityRequest.IS_SERVER_READY,
                        rpc_path=rpc_path,
                        error_message="Unable to start RPC Server")

    async def put_nowait(self, item: Any, server_info: ServerInfo):
        rpc_path = get_open_zmq_ipc_path(server_info.request_output_queue_ip, server_info.request_output_queue_port)
        set_timestamp(item, 'queue_client_send_timestamp', time.time())
        await self._send_one_way_rpc_request(
                        request=RPCPutNoWaitQueueRequest(item=item),
                        rpc_path=rpc_path,
                        error_message="Unable to put items into queue.")

    async def put_nowait_batch(self, items: Iterable, server_info: ServerInfo):
        rpc_path = get_open_zmq_ipc_path(server_info.request_output_queue_ip, server_info.request_output_queue_port)
        set_timestamp(items, 'queue_client_send_timestamp', time.time())
        await self._send_one_way_rpc_request(
                        request=RPCPutNoWaitBatchQueueRequest(items=items),
                        rpc_path=rpc_path,
                        error_message="Unable to put items into queue.")
