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
import signal
import pytest
import ray

from vllm.outputs import CompletionOutput, RequestOutput

from llumnix.queue.zmq_server import ZmqServer
from llumnix.queue.zmq_client import ZmqClient
from llumnix.queue.utils import get_open_zmq_ipc_path
from llumnix.utils import random_uuid
from llumnix.server_info import ServerInfo

# pylint: disable=W0611
from tests.conftest import ray_env

@ray.remote(num_cpus=1)
class Server:
    def __init__(self, ip, port):
        rpc_path = get_open_zmq_ipc_path(ip, port)
        self.server = ZmqServer(rpc_path)
        asyncio.create_task(self.server.run_server_loop())
        request_output_queue = self.server
        self.stop_signal = asyncio.Event()
        asyncio.create_task(self.get_request_outputs_loop(request_output_queue))
        asyncio.create_task(self._wait_until_done())

    async def get_request_outputs_loop(self, request_output_queue):
        while True:
            request_outputs = await request_output_queue.get()
            for request_output in request_outputs:
                if request_output.finished:
                    self.stop_signal.set()

    async def _wait_until_done(self):
        await self.stop_signal.wait()
        self.server.cleanup()

def gen_request_outputs(num_outputs):
    request_outputs = []
    for _ in range(num_outputs):
        request_id = random_uuid()
        completion_output = CompletionOutput(0, "", [], 0.0, None)
        request_output = RequestOutput(request_id, "", [], None, [completion_output], finished=False)
        request_outputs.append(request_output)
    request_outputs[-1].finished = True
    return request_outputs

async def async_request_output_gen(generator, qps):
    while True:
        try:
            item = next(generator)
            yield item
            await asyncio.sleep(1.0 / qps)
        except StopIteration:
            return

async def put_queue(request_output_queue, request_output, server_info):
    await request_output_queue.put_nowait([request_output], server_info)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")

async def benchmark_queue(qps, ip=None, port=None):
    rpc_client = ZmqClient()
    request_output_queue = rpc_client
    server = Server.remote(ip, port)
    server_id = random_uuid()
    server_info = ServerInfo(server_id, 'zmq', None, ip, port)
    await rpc_client.wait_for_server_rpc(server_info)

    num_request_outputs = 500
    request_outputs = gen_request_outputs(num_request_outputs)
    async_request_outputs = async_request_output_gen(iter(request_outputs), qps=qps)
    tasks = []
    async for request_output in async_request_outputs:
        tasks.append(asyncio.create_task(put_queue(request_output_queue, request_output, server_info)))
    _ = await asyncio.gather(*tasks)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        # Wait for server actor to finish.
        ray.get(server._wait_until_done.remote())
        rpc_client.close()
    # pylint: disable=W0706
    except TimeoutException:
        raise
    finally:
        signal.alarm(0)

@pytest.mark.asyncio
@pytest.mark.parametrize("qps", [128.0, 256.0, 512.0, 1024.0])
async def test_queue_zmq(ray_env, qps):
    ip = '127.0.0.1'
    port = 1234
    await benchmark_queue(qps, ip, port)
