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
import pytest

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from llumnix.backends.vllm_v1.core import BackendVLLMV1
from llumnix.arg_utils import InstanceArgs
from llumnix.entrypoints.vllm_v1.arg_utils import VLLMV1EngineArgs
from llumnix.queue.queue_type import QueueType
from llumnix.utils import InstanceType, get_free_port, random_uuid, MigrationType
from llumnix.tests.unit_test.queue.utils import request_output_queue_server
from llumnix.tests.utils import try_convert_to_local_path
from llumnix.llumlet.llumlet import Llumlet
from llumnix.ray_utils import get_placement_group_name, initialize_placement_group
from tests.conftest import ray_env


port_offset = 0

def generate_v1_llumnix_engine_args(instance_type: InstanceType):
    global port_offset
    port_offset += 1

    engine_args = AsyncEngineArgs(
        model=try_convert_to_local_path("facebook/opt-125m"),
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=4096,
    )
    engine_args.parallel_config.distributed_executor_backend = "ray"
    expected_instance_type = instance_type.value
    if expected_instance_type == "neutral":
        expected_instance_type = "decode"

    if expected_instance_type == "decode":
        kv_role = "kv_consumer"
    else:
        kv_role = "kv_producer"
    engine_args.kv_transfer_config = {
        "kv_connector": "HybridConnector",
        "kv_connector_extra_config": {
            "backend": "kvt+migration",
            "kvt_inst_id": random_uuid(),
            "kv_role": kv_role, 
            "naming_url": "file:/tmp/vllm.naming",
            "kv_port": 14579 + port_offset,
            "rpc_port": 24579 + port_offset,
        }
    }

    return VLLMV1EngineArgs(engine_args)

async def init_llumlet(instance_type: InstanceType):
    instance_id = random_uuid()
    placement_group = initialize_placement_group(
        get_placement_group_name(instance_id), num_cpus=3, num_gpus=1, detached=False)
    instance_args = InstanceArgs(enable_migration=True, enable_engine_migration_interface=True)
    llumnix_engine_args = generate_v1_llumnix_engine_args(InstanceType.DECODE)

    llumlet: Llumlet = Llumlet.from_args(
        instance_id=instance_id,
        placement_group=placement_group,
        request_output_queue_type=QueueType.ZMQ,
        instance_args=instance_args,
        llumnix_engine_args=llumnix_engine_args,
        dp_rank=0,
        dp_rank_local=0
    )
    await llumlet.is_ready.remote()

    return instance_id, llumlet


@pytest.make.asyncio
async def test_migration(ray_env):
    request_output_queue_type = QueueType.ZMQ
    que, request_processing_context = request_output_queue_server(request_output_queue_type)
    asyncio.create_task(que.run_server_loop())
    instance_id_0, llumlet_0 = await init_llumlet(InstanceType.NEUTRAL)
    llumlet_0_engine_context = await llumlet_0.get_engine_context.remote()
    instance_id_1, llumlet_1 = await init_llumlet(InstanceType.NEUTRAL)
    llumlet_1_engine_context = await llumlet_1.get_engine_context.remote()

    # {'role': 'system', 'content': 'You are a helpful assistant.'}
    # {'role': 'user', 'content': 'How to learn english, 请用中文回答'}
    # tokenizer by Qwen2.5-7B   
    prompt_token_ids=[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 4340,
                      311, 3960, 28963, 11, 220, 14880, 11622, 104811, 102104, 151645, 198, 151644, 77091, 198]
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=16, presence_penalty=1.1,
                                     repetition_penalty=1.1)
    request_id = 'chatcmpl-5a5739e39dc34837a8ae37b5a37b7ff5'
    engine_core_request = EngineCoreRequest(request_id=request_id, sampling_params=sampling_params,
                                     prompt_token_ids=prompt_token_ids)
    await llumlet_1.generate(request_id, request_processing_context, 0, engine_core_request=engine_core_request)

    await asyncio.sleep(5)
    
    len_output = que.qsize()
    assert  len_output > 0
    for i in range(len_output): 
        engine_core_output = await que.get()
        print(engine_core_output)

    
    long_decode_sampling_params= SamplingParams(ignore_eos=True, max_tokens=3000, presence_penalty=1.1,
                                     repetition_penalty=1.1)
    engine_core_request = EngineCoreRequest(request_id=request_id, sampling_params=long_decode_sampling_params,
                                     prompt_token_ids=prompt_token_ids)
    await llumlet_1.generate(request_id, request_processing_context, 0,
                                           engine_core_request=engine_core_request)
    await asyncio.sleep(1)

    await llumlet_1.migrate_out(llumlet_0, llumlet_0_engine_context, MigrationType.NO_CONSTRAINTS_LOAD_BALANCE)
    await asyncio.sleep(10)
    len_output = que.qsize()
    assert  len_output > 0
    for i in range(len_output): 
        engine_core_output = await que.get()
        print(engine_core_output)


@pytest.make.asyncio
async def test_pd_migration(ray_env):
    request_output_queue_type = QueueType.ZMQ
    que, request_processing_context = request_output_queue_server(request_output_queue_type)
    asyncio.create_task(que.run_server_loop())

    prefill_0_instance_id, prefill_0_llumlet = await init_llumlet(InstanceType.PREFILL)
