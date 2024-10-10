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

from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Any

RPC_GET_DATA_TIMEOUT_MS: int = 5000
RPC_SOCKET_LIMIT_CUTOFF = 2000
RPC_ZMQ_HWM = 0
RPC_SUCCESS_STR = "SUCCESS"

@dataclass
class RPCPutNoWaitQueueRequest:
    items: List[Any] = None

class RPCUtilityRequest(Enum):
    IS_SERVER_READY = 1

# pylint: disable=C0103
RPC_REQUEST_TYPE = Union[RPCPutNoWaitQueueRequest, RPCUtilityRequest]

class RPCClientClosedError(Exception):
    """Exception class raised when the client is used post-close.

    The client can be closed, which closes the ZMQ context. This normally
    happens on server shutdown. In some cases, methods like abort and
    do_log_stats will still be called and then try to open a socket, which
    causes a ZMQError and creates a huge stack trace.
    So, we throw this error such that we can suppress it.
    """

def get_open_zmq_ipc_path(ip, port) -> str:
    return "tcp://{}:{}".format(ip, port)
