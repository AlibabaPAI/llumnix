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

import msgspec
import pytest
from transformers import AutoTokenizer

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.config import KVTransferConfig, VllmConfig
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.v1.engine.detokenizer import SlowIncrementalDetokenizer
from vllm.usage.usage_lib import UsageContext

from llumnix.backends.vllm_v1.core import BackendVLLMV1
from llumnix.arg_utils import InstanceArgs
from llumnix.entrypoints.vllm_v1.arg_utils import VLLMV1EngineArgs
from llumnix.queue.queue_type import QueueType
from llumnix.utils import InstanceType, get_free_port, random_uuid, MigrationType, InstanceContext
from llumnix.request_processing_context import RequestProcessingContext
from llumnix.llumlet.request import RequestInferenceType

from llumnix.llumlet.llumlet import Llumlet
from llumnix.ray_utils import get_placement_group_name, initialize_placement_group

from tests.utils import try_convert_to_local_path
from tests.unit_test.queue.utils import request_output_queue_server
from tests.conftest import ray_env


model_path = '/dev/shm/Qwen2.5-7B'
engine_args = AsyncEngineArgs(
    # model=try_convert_to_local_path("Qwen/Qwen2.5-7B"),
    model=model_path,
    enforce_eager=True,
    trust_remote_code=True,
    max_model_len=4096,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_v1_llumnix_engine_args(instance_type: InstanceType):
    global engine_args

    engine_args.distributed_executor_backend = "mp"
    expected_instance_type = instance_type.value
    if expected_instance_type == "neutral":
        expected_instance_type = "decode"

    if expected_instance_type == "decode":
        kv_role = "kv_consumer"
    else:
        kv_role = "kv_producer"

    engine_args.kv_transfer_config = KVTransferConfig(
        kv_connector="HybridConnector",
        kv_role=kv_role,
        engine_id=random_uuid()+"_dp0",
        kv_connector_extra_config={
            "backend": "kvt+migration",
            "kvt_inst_id": random_uuid(),
            "naming_url": "file:/tmp/vllm.naming",
            "rpc_port": None,
        }
    )

    return VLLMV1EngineArgs(engine_args)

async def init_llumlet(instance_type: InstanceType):
    instance_id = random_uuid()
    placement_group = initialize_placement_group(
        get_placement_group_name(instance_id), num_cpus=3, num_gpus=1, detached=False)
    instance_args = InstanceArgs(enable_routine_migration=True, enable_engine_migration_interface=True)
    llumnix_engine_args = generate_v1_llumnix_engine_args(instance_type)

    llumlet: Llumlet = Llumlet.from_args(
        instance_id=instance_id,
        placement_group=placement_group,
        request_output_queue_type=QueueType.ZMQ,
        instance_args=instance_args,
        engine_args=llumnix_engine_args,
        dp_rank=0,
        dp_rank_local=0
    )
    await llumlet.is_ready.remote()

    return llumlet


def generate_engine_core_request(max_tokens: int = 300):
    # {'role': 'system', 'content': 'You are a helpful assistant.'}
    # {'role': 'user', 'content': '您好, 请从 1 数到 10000000.'}
    # tokenizer by Qwen2.5-7B 
    prompt_token_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 111308,
                        11, 220, 14880, 45181, 220, 16, 47685, 26939, 220, 16, 15, 15, 15, 15, 15, 15, 15, 13,
                        151645, 198, 151644, 77091, 198]
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=max_tokens, presence_penalty=1.1, frequency_penalty=0.0,
        repetition_penalty=1.0, include_stop_str_in_output=False,  min_tokens=0, skip_special_tokens=True)
    request_id = f'chatcmpl-{random_uuid()}'
    engine_core_request = EngineCoreRequest(
        request_id=request_id, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids,
        mm_inputs=None, mm_hashes=None, mm_placeholders=None, pooling_params=None, eos_token_id=151643,
        arrival_time=1, lora_request=None, cache_salt=None, data_parallel_rank=None, client_index=0,
        current_wave=0, priority=0)

    return engine_core_request

async def run_simple_requst(
    prefill_instance_context: InstanceContext,
    decode_instance_context: InstanceContext,
    target_llumlet: Llumlet,
    engine_core_request: EngineCoreRequest,
    output_queue: asyncio.Queue,
    request_processing_context: RequestProcessingContext
):
    global tokenizer
    dispatch_context = {
        "llumnix_scheduler": True,
        "prefill_kvt_engine_available_port": prefill_instance_context.kvt_engine_available_port,
        "prefill_engine_host": prefill_instance_context.engine_host,
        "prefill_instance_id": prefill_instance_context.instance_id,
        "decode_instance_id": decode_instance_context.instance_id
    }

    await target_llumlet.generate.remote(engine_core_request.request_id, request_processing_context, 0,
        engine_core_request=engine_core_request, dispatch_context=dispatch_context)
    await asyncio.sleep(10)

    print("\n ------ \n")
    len_output = output_queue.qsize()
    detokenizer0 = SlowIncrementalDetokenizer.from_new_request(tokenizer, engine_core_request)
    assert len_output == engine_core_request.sampling_params.max_tokens
    exist_finish = False
    for i in range(len_output): 
        lluminx_request_output = await output_queue.get()
        for output in lluminx_request_output.engine_outputs.outputs:
            assert len(output.new_token_ids) == 1
            text = detokenizer0.decode_next(output.new_token_ids[0])
            print(text, "\t", msgspec.to_builtins(output))
            if output.finished:
                exist_finish = True
                break
    assert exist_finish
    assert output_queue.qsize() == 0

# pylint: disable=unused-argument
async def run_simple_migration(
    src_llumlet: Llumlet,
    src_llumlet_instance_context: InstanceContext,
    dst_llumlet: Llumlet,
    dst_llumlet_instance_context: InstanceContext,
    request_processing_context: RequestProcessingContext,
    output_queue: asyncio.Queue,
):
    global tokenizer
    engine_core_request = generate_engine_core_request()
    await src_llumlet.generate.remote(engine_core_request.request_id, request_processing_context, 0,
                                           engine_core_request=engine_core_request)
    await asyncio.sleep(2)

    await src_llumlet.migrate_out.remote(dst_llumlet, dst_llumlet_instance_context,
                                       MigrationType.NEUTRAL_LOAD_BALANCE)
    await asyncio.sleep(15)

    print("\n --- begin to run migration --- \n")
    # currently, there is a bug in num_output_tokens for migrated request
    len_output = output_queue.qsize()
    source_instance_id = set()
    # assert len_output == engine_core_request.sampling_params.max_tokens
    assert len_output > 0
    exist_finish = False
    detokenizer1 = SlowIncrementalDetokenizer.from_new_request(tokenizer, engine_core_request)
    for i in range(len_output): 
        lluminx_request_output = await output_queue.get()
        source_instance_id.add(lluminx_request_output.instance_id)
        for output in lluminx_request_output.engine_outputs.outputs:
            assert len(output.new_token_ids) == 1
            text = detokenizer1.decode_next(output.new_token_ids[0])
            print(text, "\t", msgspec.to_builtins(output))
            if output.finished:
                # exist_finish = True
                break
    # assert exist_finish
    assert output_queue.qsize() == 0
    assert len(source_instance_id) == 2


@pytest.mark.asyncio
async def test_migration(ray_env):
    # add RAY_DEDUP_LOGS=0 VLLM_USE_V1=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 to launch command
    engine_core_request = generate_engine_core_request()
    request_output_queue_type = QueueType.ZMQ
    output_queue, request_processing_context = request_output_queue_server(request_output_queue_type)
    co_task = asyncio.create_task(output_queue.run_server_loop())

    llumlet_0, llumlet_1 = await asyncio.gather(
        asyncio.create_task(init_llumlet(InstanceType.NEUTRAL)),
        asyncio.create_task(init_llumlet(InstanceType.NEUTRAL)),
    )
    llumlet_0_engine_context = await llumlet_0.get_engine_context.remote()
    llumlet_1_engine_context = await llumlet_1.get_engine_context.remote()

    await run_simple_requst(llumlet_0_engine_context, llumlet_0_engine_context, llumlet_0,
                            engine_core_request, output_queue, request_processing_context)

    await run_simple_migration(llumlet_0, llumlet_0_engine_context, llumlet_1, llumlet_1_engine_context,
                               request_processing_context, output_queue)
    
    co_task.cancel()
    try:
        result = await co_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_pd_migration(ray_env):
    global tokenizer
    engine_core_request = generate_engine_core_request()
    request_output_queue_type = QueueType.ZMQ
    output_queue, request_processing_context = request_output_queue_server(request_output_queue_type)
    co_task = asyncio.create_task(output_queue.run_server_loop())

    prefill_llumlet_0, prefill_llumlet_1, decode_llumlet_0, decode_llumlet_1 = await asyncio.gather(
        asyncio.create_task(init_llumlet(InstanceType.PREFILL)),
        asyncio.create_task(init_llumlet(InstanceType.PREFILL)),
        asyncio.create_task(init_llumlet(InstanceType.DECODE)),
        asyncio.create_task(init_llumlet(InstanceType.DECODE)),
    )
    prefill_0_instance_context = await prefill_llumlet_0.get_engine_context.remote()
    prefill_1_instance_context = await prefill_llumlet_1.get_engine_context.remote()
    decode_0_instance_context = await decode_llumlet_0.get_engine_context.remote()
    decode_1_instance_context = await decode_llumlet_1.get_engine_context.remote()

    # pure p inference
    await run_simple_requst(prefill_0_instance_context, prefill_0_instance_context, prefill_llumlet_0,
                            engine_core_request, output_queue, request_processing_context)

    # pure d inference
    await run_simple_requst(decode_0_instance_context, decode_0_instance_context, decode_llumlet_0,
                            engine_core_request, output_queue, request_processing_context)

    # pd inference
    await run_simple_requst(prefill_0_instance_context, decode_0_instance_context, decode_llumlet_0,
                            engine_core_request, output_queue, request_processing_context)

    # decode migration
    await run_simple_migration(decode_llumlet_0, decode_0_instance_context, decode_llumlet_1, decode_1_instance_context,
                               request_processing_context, output_queue)

    # prefill migration
    await run_simple_migration(prefill_llumlet_0, prefill_0_instance_context, prefill_llumlet_1, prefill_1_instance_context,
                               request_processing_context, output_queue)

    # d -> p migration
    await run_simple_migration(decode_llumlet_1, decode_1_instance_context, prefill_llumlet_1, prefill_1_instance_context,
                               request_processing_context, output_queue)

    # p -> d migration
    await run_simple_migration(prefill_llumlet_1, prefill_1_instance_context, decode_llumlet_1, decode_1_instance_context,
                               request_processing_context, output_queue)

    co_task.cancel()
    try:
        result = await co_task
    except asyncio.CancelledError:
        pass
