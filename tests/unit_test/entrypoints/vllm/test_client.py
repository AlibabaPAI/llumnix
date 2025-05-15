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
import pytest
from vllm.engine.async_llm_engine import AsyncStream
from vllm.outputs import RequestOutput, CompletionOutput


from llumnix.entrypoints.vllm.client import LlumnixClientVLLM


class MockLlumnixClientVLLM(LlumnixClientVLLM):
    def __init__(self):  # pylint: disable=super-init-not-called
        self.request_streams: Dict[str, AsyncStream] = {}
        self.request_streams_last_completion_tokens: Dict[str, int] = {}


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
