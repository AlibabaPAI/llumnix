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

import random
from typing import Dict
import asyncio
from unittest.mock import MagicMock

import pytest
import ray

from vllm.engine.async_llm_engine import AsyncStream
from vllm.outputs import RequestOutput, CompletionOutput

from llumnix.entrypoints.vllm.client import LlumnixClientVLLM
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.utils import random_uuid
from llumnix.request_output import LlumnixRequestOuput as LlumnixRequestOuputVLLM
from llumnix.ray_utils import get_instance_name
from llumnix.metrics.timestamps import RequestTimestamps

# pylint: disable=unused-import
from tests.conftest import ray_env


def get_request_output_engine(request_id, instance_id="", finished=False):
    completion_output = CompletionOutput(0, "", [], 0.0, None)
    request_output = RequestOutput(request_id, "", [], None, [completion_output], finished=finished)
    request_output.request_timestamps = RequestTimestamps()
    llumnix_response = LlumnixRequestOuputVLLM(request_id, instance_id, request_output, request_output.request_timestamps)
    return llumnix_response


class MockLlumnixClientVLLM(LlumnixClientVLLM):
    def __init__(self, loop = None):  # pylint: disable=super-init-not-called
        self.request_stream: Dict[str, AsyncStream] = {}
        self.request_stream_last_completion_tokens: Dict[str, int] = {}
        self.request_output_queue = \
            init_request_output_queue_server(ip="127.0.0.1", queue_type="rayqueue")
        self.request_instance = {}
        self.global_instances = {}
        if loop:
            loop.create_task(self.get_request_outputs_loop())
            loop.create_task(self.request_output_queue.run_server_loop())

    # pylint: disable=arguments-differ,invalid-overridden-method
    def generate(self, request_id):
        results_generator = AsyncStream(request_id, cancel=self.abort_request)
        self.request_stream[request_id] = results_generator


@ray.remote(num_cpus=0)
class MockLlumlet:
    def __init__(self):
        self.num_aborts = 0

    def abort(self, request_id):
        self.num_aborts += 1

    def get_num_aborts(self):
        return self.num_aborts


def gen_completion_output(index: int = 0, tokens_length: int = 1) -> CompletionOutput:
    text = " ".join(str(i) for i in range(tokens_length))
    token_ids = list(range(tokens_length))
    return CompletionOutput(
        index=index,
        text=text,
        token_ids=token_ids,
        cumulative_logprob=None,
        logprobs=None,
    )


def gen_request_output(
    request_id: str, tokens_len: int = 1, output_len: int = 1, finished: bool = False
) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt="123",
        prompt_token_ids="123",
        prompt_logprobs=None,
        finished=finished,
        outputs=[gen_completion_output(i, tokens_len + i) for i in range(output_len)],
    )


def get_correct_order_output(
    request_id: str, length: int = 1000, max_outputs_len: int = 1
):
    i = 0
    while i < length - 1:
        output_len = random.randint(1, min(length - 1 - i, max_outputs_len))
        yield gen_request_output(request_id, i + 1, output_len)
        i += output_len
    yield gen_request_output(request_id, i + 1, 1, finished=True)


def get_out_of_order_output(
    request_id: str,
    length: int,
    max_outputs_len: int = 1,
    is_finished_flag_at_last_output: bool = True,
):
    i = 0
    reqeust_outputs = []
    while i < length - 1:
        output_len = random.randint(1, min(length - 1 - i, max_outputs_len))
        reqeust_outputs.append(gen_request_output(request_id, i + 1, output_len))
        i += output_len
    random.shuffle(reqeust_outputs)
    last_output: RequestOutput = gen_request_output(request_id, i + 1, 1, finished=True)
    if is_finished_flag_at_last_output:
        reqeust_outputs.append(last_output)
    else:
        reqeust_outputs.insert(len(reqeust_outputs) // 2, last_output)

    for output in reqeust_outputs:
        yield output


def check_processed_output(processed_output: list[RequestOutput], output_length):
    last_completion_token = 0
    assert len(processed_output[-1].outputs[-1].token_ids) == output_length
    if output_length == 0:
        return
    for request_output in processed_output:
        current_completion_token = len(request_output.outputs[-1].token_ids)
        assert current_completion_token > last_completion_token
        last_completion_token = current_completion_token
    assert processed_output[-1].finished


@pytest.mark.parametrize("max_outputs_len", [1, 2, 3])
def test_correct_order_output(max_outputs_len):
    client = MockLlumnixClientVLLM()
    for i in range(1, 100):
        output_length = i
        request_id = f"request_id{i}-{max_outputs_len}"
        print(f"test_correct_order_output {request_id}")
        res = []

        for request_output in get_correct_order_output(
            request_id=request_id, length=output_length, max_outputs_len=max_outputs_len
        ):
            res.append(client._process_output_order(request_id, request_output))

        check_processed_output(res, output_length)


@pytest.mark.parametrize("max_outputs_len", [1, 2, 3])
@pytest.mark.parametrize("is_finished_flag_at_last_output", [True, False])
def test_out_of_order_output(max_outputs_len, is_finished_flag_at_last_output):
    client = MockLlumnixClientVLLM()
    for i in range(1, 100):
        output_length = i
        request_id = (
            f"request_id{i}-{max_outputs_len}-{is_finished_flag_at_last_output}"
        )
        print(f"test_out_of_order_output {request_id}")
        res = []
        for request_output in get_out_of_order_output(
            request_id=request_id,
            length=output_length,
            max_outputs_len=max_outputs_len,
            is_finished_flag_at_last_output=True,
        ):
            processed_output: RequestOutput = client._process_output_order(request_id, request_output)
            if processed_output:
                res.append(processed_output)
        check_processed_output(res, output_length)


@pytest.mark.asyncio
async def test_abort_and_abort_request(ray_env):
    client = MockLlumnixClientVLLM(loop=asyncio.get_event_loop())
    # yield to run get_request_outputs_loop and run_server_loop
    await asyncio.sleep(3.0)

    request_id = random_uuid()
    # Add request_id to request_streams for get_request_outputs_loop.
    client.generate(request_id)
    instance_id = random_uuid()

    # test no instance_id case
    instance_id_returned, instance_returned = client._get_instance_for_abort(request_id)
    assert instance_id_returned is None and instance_returned is None

    request_output_engine = get_request_output_engine(request_id, instance_id, False)
    client.request_output_queue.queue.put(([request_output_engine], None), block=True)
    # yield to get request outputs
    await asyncio.sleep(3.0)

    # test no instance case
    assert request_id in client.request_instance and client.request_instance[request_id] == instance_id
    instance_id_returned, instance_returned = client._get_instance_for_abort(request_id)
    assert instance_id_returned == instance_id and instance_returned is None

    # why must set namespace?
    instance = MockLlumlet.options(name=get_instance_name(instance_id),
                                   namespace="llumnix").remote()

    # test correct case
    instance_id_returned, instance_returned = client._get_instance_for_abort(request_id)
    assert instance_id_returned == instance_id and instance_returned == instance
    await client.abort(request_id)
    assert client.global_instances[instance_id] == instance
    num_aborts = ray.get(instance.get_num_aborts.remote())
    assert num_aborts == 1

    # test abort_request
    request_id1 = random_uuid()
    client.request_instance[request_id1] = instance_id
    client.abort_request(request_id1)
    await asyncio.sleep(3.0)
    num_aborts = ray.get(instance.get_num_aborts.remote())
    assert num_aborts == 2

    # test request states
    assert request_id not in client.request_stream and \
        request_id not in client.request_stream_last_completion_tokens and \
        request_id not in client.request_instance


@pytest.mark.asyncio
async def test_clear_client_request_states(ray_env):
    client = MockLlumnixClientVLLM(loop=asyncio.get_event_loop())
    # yield to run get_request_outputs_loop and run_server_loop
    await asyncio.sleep(3.0)

    request_id = random_uuid()
    # Add request_id to request_streams for get_request_outputs_loop.
    client.generate(request_id)
    instance_id = random_uuid()

    request_output_engine = get_request_output_engine(request_id, instance_id, False)
    client.request_output_queue.queue.put(([request_output_engine], None), block=True)
    # yield to get request outputs
    await asyncio.sleep(3.0)

    # test request states
    assert request_id in client.request_stream and \
        request_id in client.request_instance

    request_output_engine.engine_output.finished = True
    client._process_output_order = MagicMock()
    client._process_output_order.return_value = request_output_engine
    client.request_output_queue.queue.put(([request_output_engine], None), block=True)
    # yield to get request outputs
    await asyncio.sleep(3.0)

    # test request states
    assert request_id not in client.request_stream and \
        request_id not in client.request_instance
