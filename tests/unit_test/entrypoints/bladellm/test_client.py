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
from blade_llm.protocol import GenerateStreamResponse, Token, TokenUsage
import pytest
from llumnix.entrypoints.bladellm.client import LlumnixClientBladeLLM


class MockLlumnixClientBladeLLM(LlumnixClientBladeLLM):
    def __init__(self):  # pylint: disable=super-init-not-called
        self.entrypoint_id2llumnix_id = {}  # int32 -> int32

        self.request_streams: Dict[int, asyncio.Queue] = {}
        self.request_streams_last_completion_tokens: Dict[str, int] = {}
        self.request_streams_output_stash: Dict[str, list[GenerateStreamResponse]] = {}
        self.instance_num_requests: Dict[str, int] = {}
        self.num_finished_requests = 0
        self.manager_available = True

    def _process_output_order(
        self, request_id: int, request_output: GenerateStreamResponse
    ) -> List[GenerateStreamResponse]:
        res = super()._process_output_order(request_id, request_output)
        if res:
            self.request_streams_last_completion_tokens[request_id] = \
                res[-1].usage.completion_tokens
        return res


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
