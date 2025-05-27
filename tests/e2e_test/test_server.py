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

import json
from typing import Type, TypeVar, Generator, Dict, Any, List
import uuid
from multiprocessing import Pool
import time

import pytest
import requests
import ray
import torch

from blade_llm.protocol import (
    OAIChatCompletionsResponse,
    OAICompletionsResponse,
)

from llumnix.utils import get_ip_address, wait_port_free
from llumnix.logging.logger import init_logger
from llumnix.ray_utils import get_actor_names_by_name_prefix, INSTANCE_NAME_PREFIX, get_instance

from tests.utils import try_convert_to_local_path
from tests.e2e_test.bladellm_utils import LlumnixServerProc
# pylint: disable=unused-import
from tests.e2e_test.utils import (
    wait_for_llumnix_service_ready,
    shutdown_llumnix_service_func,
    cleanup_ci_outputs_func,
)
from tests.conftest import cleanup_ray_env_func

logger = init_logger(__name__)

RESP_T = TypeVar('RESP_T', OAICompletionsResponse, OAIChatCompletionsResponse)
NAMING_URL = "file:/tmp/llumnix/naming"


def generate_bladellm_non_pd_serve_args(
    model: str = try_convert_to_local_path('Qwen/Qwen2.5-7B'),
    ip: str = get_ip_address(),
    port: int = 45000,
    max_instances: int = 1,
) -> List[str]:
    # pylint: disable=f-string-without-interpolation
    args = [
        f"--model={model}",
        f"--host={ip}",
        f"--port={port}",
        f"--enable_llumnix",
        f"--disable_frontend_multiprocessing",
        f"--disable_signal_handler",
        f"--disable_cuda_graph",
        f"--max-instances={max_instances}",
        f"--enable-port-increment",
    ]
    return args

def generate_bladellm_pdd_serve_args(
    model: str = try_convert_to_local_path('Qwen/Qwen2.5-7B'),
    ip: str = get_ip_address(),
    port: int = 45000,
    max_instances: int = 1,
) -> List[str]:
    # pylint: disable=f-string-without-interpolation
    args = [
        f"--model={model}",
        f"--host={ip}",
        f"--port={port}",
        f"--enable_llumnix",
        f"--disable_frontend_multiprocessing",
        f"--disable_signal_handler",
        f"--disable_cuda_graph",
        f"--enable_disagg",
        f"--disagg_pd.inst_id={str(uuid.uuid4().hex)[:8]}",
        f"--disagg_pd.inst_role=prefill", # useless, setted in scaler
        f"--disagg_pd.disagg_transfer_type=rdma",
        f"--naming_url={NAMING_URL}",
        f"--enable-engine-pd-disagg",
        f"--pd-ratio=1:1",
        f"--max-instances={max_instances}",
        f"--enable-port-increment",
    ]
    return args

@pytest.fixture(scope="session", params=["server", "pd_server"])
def server(request):
    cleanup_ci_outputs_func()
    model = try_convert_to_local_path('Qwen/Qwen2.5-7B')
    ip = get_ip_address()
    if not request.param == "pd_server":
        base_port = 45000
        max_instances = 2
        args = generate_bladellm_non_pd_serve_args(model, ip, base_port, max_instances)
    else:
        base_port = 45050
        max_instances = 4
        args = generate_bladellm_pdd_serve_args(model, ip, base_port, max_instances)
    ip_ports = []
    for i in range(max_instances):
        port = base_port + i
        wait_port_free(port, force=True)
        ip_port = f"{ip}:{port}"
        ip_ports.append(ip_port)
    server = LlumnixServerProc(args)
    wait_for_llumnix_service_ready(ip_ports)
    yield server
    cleanup_ray_env_func()
    shutdown_llumnix_service_func()
    server.destroy()

def sync_chunks(response, resp_cls: Type[RESP_T] = OAICompletionsResponse) -> Generator[RESP_T, None, None]:
    for line_in_bytes in response.iter_lines(chunk_size=8192, decode_unicode=False):
        line = line_in_bytes.decode('utf8').rstrip('\r\n')
        if not line:  # ignore empty lines
            continue
        # received line should be like 'data: {...}'
        if ':' in line:
            data = line.split(':', 1)[1].lstrip()
            if data != "[DONE]":
                yield resp_cls(**json.loads(data))
        else:
            logger.warning('receive unexpected content: {}'.format(line))

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize("max_tokens,finish_reason", [(1024, "stop"), (1, "length")])
async def test_http_chat_api_stream(server, max_tokens, finish_reason):
    req_dict = {
        'messages': [
            {'role': 'user', 'content': 'who is the author of One Hundred Years of Solitude'},
        ],
        'max_tokens': max_tokens,
        'temperature': 0,
        'seed': 42,
        'stream': True,
        "stop_tokens": [151645, 151644, 151643],
        "stop": ["<|endoftext|>"],
    }
    assert server is not None
    url = f"http://{server.addr}/v1/chat/completions"
    response_text = ''

    response = requests.post(
        url,
        json=req_dict,
        headers={"Content-Type": "application/json"},
        stream=True,
    )

    last_resp = None
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False):
        msg = chunk.decode("utf-8")
        if msg.startswith('data'):
            info = msg[6:]
            if info == '[DONE]':
                break
            resp = json.loads(info)
            response_text += resp['choices'][0]['delta']['content']
            last_resp = resp

    assert len(response_text) > 0
    assert finish_reason == last_resp['choices'][0]['finish_reason']

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize("max_tokens,finish_reason", [(1024, "stop"), (1, "length")])
async def test_http_chat_api_no_stream(server, max_tokens, finish_reason):
    req_dict = {
        'messages': [
            {'role': 'user', 'content': 'who is the author of One Hundred Years of Solitude'},
        ],
        'max_tokens': max_tokens,
        'temperature': 0,
        'seed': 42,
        "stop_tokens": [151645, 151644, 151643],
        "stop": ["<|endoftext|>"],
    }
    assert server is not None
    url = f"http://{server.addr}/v1/chat/completions"

    response = requests.post(
        url,
        json=req_dict,
        headers={"Content-Type": "application/json"},
    )
    choice = json.loads(response.text)['choices'][0]
    response_message = choice['message']

    assert 'assistant' == response_message['role']
    assert len(response_message['content']) > 0
    assert finish_reason == choice['finish_reason']

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize("logprobs,top_logprobs", [('true', None), ('true', 1), ('true', 5)])
async def test_http_chat_api_logprobs(server, logprobs, top_logprobs):
    req_dict = {
        'messages': [
            {'role': 'user', 'content': 'who is the author of One Hundred Years of Solitude'},
        ],
        'temperature': 0,
        'seed': 42,
        'logprobs': logprobs,
        'top_logprobs': top_logprobs,
        "stop_tokens": [151645, 151644, 151643],
        "stop": ["<|endoftext|>"],
    }
    assert server is not None
    url = f"http://{server.addr}/v1/chat/completions"

    response = requests.post(
        url,
        json=req_dict,
        headers={"Content-Type": "application/json"},
    )
    choice = json.loads(response.text)['choices'][0]
    response_logprobs = choice['logprobs']

    assert len(response_logprobs) > 0

    response_top_logprobs_example = response_logprobs['content'][0]['top_logprobs']
    assert response_top_logprobs_example is None or len(response_top_logprobs_example) == top_logprobs

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_chat_api_resume(server):
    req_dict = {
        'messages': [
            {'role': 'user', 'content': 'hello.'},
            {'role': 'assistant', 'content': 'what can I do for you?'},
            {'role': 'user', 'content': 'what is the capital of Canada?'},
        ],
        "resume_response": "Ottawa is the capital of",
        'temperature': 0,
        'seed': 42,
        "stop": ["<|endodtext|>", "<|im_start|>", "<|im_end|>"],
    }
    assert server is not None
    url = f"http://{server.addr}/v1/chat/completions"

    response = requests.post(
        url,
        json=req_dict,
        headers={"Content-Type": "application/json"},
    )
    choice = json.loads(response.text)['choices'][0]
    response_message = choice['message']

    assert len(response_message['content']) > 0

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_lade_http_oai_completions_stream(server):
    req_dict = {
        'prompt': 'hello',
        'stream': True,
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    texts = [c.choices[0].text for c in sync_chunks(response)]
    assert len(''.join(texts)) > 0

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_lade_http_oai_completions_non_stream(server):
    req_dict = {
        'prompt': 'hello',
        'stream': False,
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 200
    assert len(OAICompletionsResponse(**response.json()).choices[0].text) > 0

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_lade_http_oai_completions_external_request_id_stream(server):
    req_dict = {
        'prompt': 'hello',
        'stream': True,
    }
    request_id = 'cmpl-12345abcd'
    headers = {'X-DashScope-RequestId': request_id}
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
        headers=headers
    )
    ids = [c.id for c in sync_chunks(response)]
    assert all(x == request_id for x in ids)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_lade_http_oai_completions_external_request_id_no_stream(server):
    req_dict = {
        'prompt': 'hello',
        'stream': False,
    }
    request_id = 'cmpl-12345abcd'
    headers = {'X-DashScope-RequestId': request_id}
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
        headers=headers
    )
    assert response.status_code == 200
    assert OAICompletionsResponse(**response.json()).id == request_id

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_stream(server):
    req_dict = {
        'prompt': 'hello',
        'stream': True,
        'logprobs': True,
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    texts = []
    for c in sync_chunks(response):
        texts.append(c.choices[0].text)
        assert c.choices[0].logprobs.content[0].text == c.choices[0].text
        assert c.choices[0].logprobs.content[0].logprob is not None
    assert len(''.join(texts)) > 0

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize(
    "max_tokens,expected_finish_reason",
    [
        (1, 'length'),
        (1024, 'stop'),
    ],
)
async def test_http_oai_completions_stream_max_tokens_stream(
    server, max_tokens, expected_finish_reason
):
    req_dict = {
        'prompt': 'who is the author of One Hundred Years of Solitude',
        'stream': True,
        'max_tokens': max_tokens,
        'temperature': 0,
        'seed': 42,
        "stop_tokens": [151645, 151644, 151643],
        "stop": ["<|endoftext|>"],
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    sse_chunks = list(sync_chunks(response))
    texts = [c.choices[0].text for c in sse_chunks]
    assert len(''.join(texts)) > 0
    assert sse_chunks[-1].choices[0].finish_reason == expected_finish_reason

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_chat_completions_non_stream(server):
    req_dict = {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'hello'},
        ],
        'stream': False,
        'logprobs': True,
    }
    request_id = 'cmpl-12345abcd'
    headers = {'X-DashScope-RequestId': request_id}
    assert server is not None
    url = f"http://{server.addr}/v1/chat/completions"
    response = requests.post(
        url,
        json=req_dict,
        headers=headers,
    )
    assert response.status_code == 200
    resp = OAIChatCompletionsResponse(**response.json())
    assert len(resp.choices[0].message['content']) > 0
    content = resp.choices[0].logprobs.content
    assert len(''.join(c.text for c in content)) > 0
    assert all(c.logprob is not None for c in content)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize(
    "max_tokens,expected_finish_reason",
    [
        (1, 'length'),
        (1024, 'stop'),
    ],
)
async def test_http_oai_completions_stream_max_tokens_non_stream(
    server, max_tokens, expected_finish_reason
):
    req_dict = {
        'prompt': 'who is the author of One Hundred Years of Solitude',
        'stream': False,
        'max_tokens': max_tokens,
        'temperature': 0,
        'seed': 42,
        "stop_tokens": [151645, 151644, 151643],
        "stop": ["<|endoftext|>"],
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 200
    resp = OAICompletionsResponse(**response.json())
    assert len(resp.choices[0].text) > 0
    assert resp.choices[0].finish_reason == expected_finish_reason

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_non_stream(server):
    req_dict = {
        'prompt': 'hello',
        'stream': False,
        'logprobs': True,
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 200
    assert len(OAICompletionsResponse(**response.json()).choices[0].text) > 0
    content = OAICompletionsResponse(**response.json()).choices[0].logprobs.content
    assert len(''.join(c.text for c in content)) > 0
    assert all(c.logprob is not None for c in content)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_external_request_id_stream(server):
    req_dict = {
        'prompt': 'hello',
        'stream': True,
    }
    request_id = 'cmpl-12345abcd'
    headers = {'X-DashScope-RequestId': request_id}
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
        headers=headers
    )
    ids = [c.id for c in sync_chunks(response)]
    assert all(x == request_id for x in ids)

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_external_request_id_no_stream(server):
    req_dict = {
        'prompt': 'hello',
        'stream': False,
    }
    request_id = 'cmpl-12345abcd'
    headers = {'X-DashScope-RequestId': request_id}
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
        headers=headers
    )
    assert response.status_code == 200
    assert OAICompletionsResponse(**response.json()).id == request_id

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_empty_request(server):
    req_dict = {}
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 400
    text = response.text
    assert 'Invalid request' in text
    assert 'Field required' in text

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_unsupported_request_param(server):
    # unsupported request params are ignored.
    req_dict = {
        'prompt': 'hello',
        'foo': 'bar',
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 200

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_invalid_type_in_list(server):
    req_dict = {
        'prompt': 'hello',
        'stop': [123],
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 400
    text = response.text
    assert 'Invalid request' in text
    assert 'stop[0]' in text
    assert 'Input should be a valid string' in text

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_invalid_type(server):
    req_dict = {
        'prompt': 'hello',
        'seed': 'abc',
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 400
    text = response.text
    assert 'Invalid request' in text
    assert 'seed' in text
    assert 'Input should be a valid int' in text

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_chat_completions_unsupported_request_param(server):
    # unsupported request param `prompt` are ignored, but `message` is required.
    req_dict = {
        'prompt': 'hello',
    }
    assert server is not None
    url = f"http://{server.addr}/v1/chat/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 400
    text = response.text
    assert 'Field required' in text

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize("penalty", ['presence_penalty', 'frequency_penalty'])
@pytest.mark.parametrize("penalty_val", [-2.1, 2.1])
async def test_http_oai_completions_invalid_precense_freqency_penalty(
    server, penalty: str, penalty_val: float
):
    # unsupported request param `prompt` are ignored, but `message` is required.
    req_dict = {
        'prompt': 'hello',
        'stream': False,
        penalty: penalty_val,
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 400
    text = response.text
    assert 'Value error' in text
    assert 'has to be in [-2, 2],' in text

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize("penalty", ['presence_penalty', 'frequency_penalty'])
@pytest.mark.parametrize("penalty_val", [-2.1, 2.1])
async def test_http_oai_chat_completions_invalid_precens_frequency_penalty(
    server, penalty: str, penalty_val: float
):
    # unsupported request param `prompt` are ignored, but `message` is required.
    req_dict = {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'hello'},
        ],
        'stream': False,
        penalty: penalty_val,
    }
    assert server is not None
    url = f"http://{server.addr}/v1/chat/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 400
    text = response.text
    assert 'Value error' in text
    assert 'has to be in [-2, 2],' in text

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
async def test_http_oai_completions_invalid_repetition_penalty(server):
    # unsupported request param `prompt` are ignored, but `message` is required.
    req_dict = {
        'prompt': 'hello',
        'stream': False,
        'repetition_penalty': 0.0,
    }
    assert server is not None
    url = f"http://{server.addr}/v1/completions"
    response = requests.post(
        url,
        json=req_dict,
    )
    assert response.status_code == 400
    text = response.text
    assert 'Value error' in text
    assert 'has to be a strictly positive float' in text

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_chat", [True, False])
async def test_model_name_in_oai_completion(server, stream: bool, is_chat: bool):
    url = f"http://{server.addr}/v1/chat/completions" if is_chat else f"http://{server.addr}/v1/completions"
    payload: Dict[str, Any] = (
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'hello'},
            ]
        }
        if is_chat
        else {'prompt': 'hello'}
    )
    payload['model'] = 'mm'
    payload['stream'] = stream
    response = requests.post(
        url,
        json=payload,
    )
    if stream:
        resp_cls = OAIChatCompletionsResponse if is_chat else OAICompletionsResponse
        for chunk in sync_chunks(response, resp_cls=resp_cls):
            assert chunk.model == 'mm'
    else:
        assert response.status_code == 200
        resp = response.json()
        assert resp['model'] == 'mm'

def _query_server(args):
    url, payload = args
    response = requests.post(
        url,
        json=payload,
    )
    response.raise_for_status()
    return response.json()

@pytest.mark.asyncio
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="at least 4 gpus required for migration bench")
@pytest.mark.parametrize("is_chat", [True, False])
async def test_http_oai_completions_drop_request(server, is_chat: bool):
    url = f"http://{server.addr}/v1/chat/completions" \
        if is_chat else f"http://{server.addr}/v1/completions"
    payload: Dict[str, Any] = (
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'hello'},
            ]
        }
        if is_chat
        else {'prompt': 'hello'}
    )
    payload['stream'] = True
    payload['max_tokens'] = 4096
    payload['ignore_eos'] = True

    with Pool(32) as pool:
        urls = [url] * 100
        pyloads = [payload] * 100
        pool.map_async(_query_server, zip(urls, pyloads))
        # give it some times to add requests
        time.sleep(1.0)
        pool.terminate()
        pool.join()

    # give it some times to drop requests
    time.sleep(5.0)

    ray.init(ignore_reinit_error=True, namespace="llumnix")

    curr_instances = []
    curr_instance_names = get_actor_names_by_name_prefix(name_prefix=INSTANCE_NAME_PREFIX)
    for curr_instance_name in curr_instance_names:
        instance_id = curr_instance_name.split("_")[-1]
        curr_instances.append(instance_id)
    for instance_id in curr_instances:
        instance = get_instance(instance_id)
        running_request_ids = ray.get(instance.execute_engine_method.remote("get_running_queue"))
        waiting_request_ids = ray.get(instance.execute_engine_method.remote("get_waiting_queue"))
        # check that all requests are dropped
        assert len(running_request_ids) == 0 and len(waiting_request_ids) == 0
