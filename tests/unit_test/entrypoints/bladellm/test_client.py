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
import random
from typing import Dict, List
import time

import pytest
import ray
from unittest.mock import MagicMock

from blade_llm.protocol import GenerateStreamResponse, Token, TokenUsage

from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM
from llumnix.utils import random_uuid
from llumnix.ray_utils import get_instance_name
from llumnix.request_output import LlumnixRequestOuput

# pylint: disable=unused-import
from tests.conftest import ray_env

# TODO(KuilongCui): cannot pass test, fix it


class MockLlumnixClientBladeLLM(LlumnixClientBladeLLM):
    def __init__(self, loop = None):  # pylint: disable=super-init-not-called
        self.request_stream: Dict[int, asyncio.Queue] = {}
        self.request_stream_last_completion_tokens: Dict[int, int] = {}
        self.request_stream_output_stash: Dict[int, list[GenerateStreamResponse]] = {}
        self.instance_num_requests: Dict[str, int] = {}
        self.num_finished_requests = 0
        self.manager_available = True
        self.log_request_timestamps = False

        self.request_output_queue = asyncio.Queue()
        self.llumnix_req_id_to_entrypoint_req_id = {}
        self.entrypoint_req_id_to_llumnix_req_id = {}
        self.request_instance = {}
        self.global_instances = {}
        if loop:
            loop.create_task(self.get_request_outputs_loop())

    def _process_output_order(
        self, request_id: int, request_output: GenerateStreamResponse
    ) -> List[GenerateStreamResponse]:
        res = super()._process_output_order(request_id, request_output)
        if res:
            self.request_stream_last_completion_tokens[request_id] = \
                res[-1].usage.completion_tokens
        return res

    # pylint: disable=arguments-differ,invalid-overridden-method
    async def _add_request(self, request_id):
        self.llumnix_req_id_to_entrypoint_req_id[request_id] = request_id
        self.entrypoint_req_id_to_llumnix_req_id[request_id] = request_id
        self.request_stream[request_id] = asyncio.Queue()


@ray.remote(num_cpus=0)
class MockLlumlet:
    def __init__(self):
        self.num_aborts = 0

    def abort(self, request_id):
        self.num_aborts += 1

    def get_num_aborts(self):
        return self.num_aborts


def get_correct_order_output(length: int = 1000, contain_block: bool = False):
    i = 0
    while i < length - 1:
        tokens_len = random.randint(1, min(length - 1 - i, 5)) if contain_block else 1
        yield GenerateStreamResponse(
            is_ok=True,
            is_finished=False,
            tokens=[Token(id=i + j, text=str(i + j)) for j in range(tokens_len)],
            usage=TokenUsage(
                prompt_tokens=0,
                completion_tokens=i + tokens_len,
                total_tokens=i + tokens_len,
            ),
        )
        i += tokens_len
    yield GenerateStreamResponse(
        is_ok=True,
        is_finished=True,
        tokens=[Token(id=length - 1, text=str(length - 1))],
        usage=TokenUsage(
            prompt_tokens=0, completion_tokens=length, total_tokens=length
        ),
    )


def get_out_of_order_output(
    length: int, contain_block: bool, is_finished_flag_at_last_output: bool
):
    i = 0
    reqeust_outputs = []
    while i < length - 1:
        tokens_len = random.randint(1, min(length - 1 - i, 5)) if contain_block else 1
        reqeust_outputs.append(
            GenerateStreamResponse(
                is_ok=True,
                is_finished=False,
                tokens=[Token(id=i + j, text=str(i + j)) for j in range(tokens_len)],
                usage=TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=i + tokens_len,
                    total_tokens=i + tokens_len,
                ),
            )
        )
        i += tokens_len
    random.shuffle(reqeust_outputs)
    last_output = GenerateStreamResponse(
        is_ok=True,
        is_finished=True,
        tokens=[Token(id=length - 1, text=str(length - 1))],
        usage=TokenUsage(prompt_tokens=0, completion_tokens=i + 1, total_tokens=i + 1),
    )
    if is_finished_flag_at_last_output:
        reqeust_outputs.append(last_output)
    else:
        reqeust_outputs.insert(len(reqeust_outputs) // 2, last_output)

    for output in reqeust_outputs:
        yield output


def check_processed_output(
    processed_output: list[GenerateStreamResponse], output_length
):
    last_completion_token = 0
    assert sum([len(output.tokens) for output in processed_output]) == output_length
    if output_length == 0:
        return
    for output in processed_output:
        assert output.usage.completion_tokens == last_completion_token + len(
            output.tokens
        )
        last_completion_token = output.usage.completion_tokens
    assert processed_output[-1].is_finished


@pytest.mark.parametrize("contain_block", [True, False])
def test_correct_order_output(contain_block):
    client = MockLlumnixClientBladeLLM()
    for i in range(1, 100):
        output_length = i
        request_id = f"request_id{i}-{contain_block}"
        print(f"test_correct_order_output {request_id}")
        res = []

        for request_output in get_correct_order_output(output_length, contain_block):
            res.extend(client._process_output_order(request_id, request_output))

        check_processed_output(res, output_length)


@pytest.mark.parametrize("contain_block", [True, False])
@pytest.mark.parametrize("is_finished_flag_at_last_output", [True, False])
def test_out_of_order_output(contain_block, is_finished_flag_at_last_output):
    client = MockLlumnixClientBladeLLM()
    for i in range(1, 100):
        output_length = i
        request_id = f"request_id{i}-{contain_block}-{is_finished_flag_at_last_output}"
        print(f"test_out_of_order_output {request_id}")
        res = []
        for request_output in get_out_of_order_output(
            length=output_length,
            contain_block=contain_block,
            is_finished_flag_at_last_output=True,
        ):
            res.extend(client._process_output_order(request_id, request_output))
        check_processed_output(res, output_length)

@pytest.mark.asyncio
async def test_drop_request(ray_env):
    client = MockLlumnixClientBladeLLM(loop=asyncio.get_event_loop())
    # yield to run get_request_outputs_loop and run_server_loop
    await asyncio.sleep(3.0)

    request_id = random.randint(0, 1024)
    # Add request_id to request_streams for get_request_outputs_loop.
    await client._add_request(request_id)
    instance_id = random_uuid()

    # test no instance_id case
    instance_id_returned, instance_returned = client._get_instance_for_abort(request_id)
    assert instance_id_returned is None and instance_returned is None

    request_output = GenerateStreamResponse(req_id=request_id)
    llumnix_response = LlumnixRequestOuput(request_output.req_id, instance_id, request_output.model_dump_json())
    await client.request_output_queue.put([llumnix_response])
    # yield to get request outputs
    await asyncio.sleep(3.0)

    # test no instance case
    assert request_id in client.request_instance and client.request_instance[request_id] == instance_id
    instance_id_returned, instance_returned = client._get_instance_for_abort(request_id)
    assert instance_id_returned == instance_id and instance_returned is None

    instance = MockLlumlet.options(name=get_instance_name(instance_id),
                                   namespace="llumnix").remote()

    # test correct case
    instance_id_returned, instance_returned = client._get_instance_for_abort(request_id)
    assert instance_id_returned == instance_id and instance_returned == instance
    await client.drop_request(request_id)
    time.sleep(3.0)
    assert client.global_instances[instance_id] == instance
    num_aborts = ray.get(instance.get_num_aborts.remote())
    assert num_aborts == 1

    # test request states
    assert request_id not in client.request_stream and \
        request_id not in client.request_stream_last_completion_tokens and \
        request_id not in client.request_instance and \
        request_id not in client.request_stream_output_stash and \
        request_id not in client.llumnix_req_id_to_entrypoint_req_id and \
        request_id not in client.entrypoint_req_id_to_llumnix_req_id


@pytest.mark.asyncio
async def test_clear_client_request_states(ray_env):
    client = MockLlumnixClientBladeLLM(loop=asyncio.get_event_loop())
    # yield to run get_request_outputs_loop and run_server_loop
    await asyncio.sleep(3.0)

    request_id = random.randint(0, 1024)
    # Add request_id to request_streams for get_request_outputs_loop.
    await client._add_request(request_id)
    instance_id = random_uuid()

    request_output = GenerateStreamResponse(req_id=request_id)
    llumnix_response = LlumnixRequestOuput(request_output.req_id, instance_id, request_output.model_dump_json())
    await client.request_output_queue.put([llumnix_response])
    # yield to get request outputs
    await asyncio.sleep(3.0)

    # test request states
    assert request_id in client.request_stream and \
        request_id in client.request_instance and \
        request_id in client.llumnix_req_id_to_entrypoint_req_id and \
        request_id in client.entrypoint_req_id_to_llumnix_req_id

    request_output.is_finished = True
    client._process_output_order = MagicMock()
    client._process_output_order.return_value = [request_output]
    llumnix_response = LlumnixRequestOuput(request_output.req_id, instance_id, request_output.model_dump_json())
    await client.request_output_queue.put([llumnix_response])
    # yield to get request outputs
    await asyncio.sleep(3.0)

    # test request states
    assert request_id not in client.request_stream and \
        request_id not in client.request_instance and \
        request_id not in client.llumnix_req_id_to_entrypoint_req_id and \
        request_id not in client.entrypoint_req_id_to_llumnix_req_id
