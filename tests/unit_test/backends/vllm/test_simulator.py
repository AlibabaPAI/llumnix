import asyncio
import math
import pytest
import ray

from vllm import EngineArgs, SamplingParams
from vllm.utils import random_uuid
from vllm.sequence import ExecuteModelRequest

from llumnix.backends.vllm.sim_executor import SimGPUExecutor
from llumnix.backends.vllm.sim_llm_engine import BackendSimVLLM
from llumnix.backends.profiling import LatencyMemData
from llumnix.internal_config import MigrationConfig
from llumnix.queue.queue_type import QueueType
from llumnix.utils import initialize_placement_group, get_placement_group_name

# pylint: disable=unused-import
from tests.conftest import ray_env
from tests.unit_test.queue.utils import request_output_queue_server

from .utils import create_dummy_prompt, initialize_scheduler

class MockBackendSim(BackendSimVLLM):

    def _get_lantecy_mem(self, *args, **kwargs):
        latency_mem = LatencyMemData({}, {}, {})
        latency_mem.prefill_model_params = (0,0)
        latency_mem.decode_model_params = (0,0,0)
        return latency_mem

@pytest.mark.asyncio
async def test_executor():
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    engine_config = engine_args.create_engine_config()
    latency_mem = LatencyMemData({},{},{})
    latency_mem.prefill_model_params = (1,1)
    latency_mem.decode_model_params = (1,1,1)
    SimGPUExecutor.latency_mem = latency_mem
    executor = SimGPUExecutor(
            model_config=engine_config.model_config,
            cache_config=engine_config.cache_config,
            parallel_config=engine_config.parallel_config,
            scheduler_config=engine_config.scheduler_config,
            device_config=engine_config.device_config,
            lora_config=engine_config.lora_config,
            speculative_config=engine_config.speculative_config,
            load_config=engine_config.load_config,
            prompt_adapter_config=engine_config.prompt_adapter_config,
            observability_config=engine_config.observability_config)
    scheduler = initialize_scheduler()
    metas, out, _ = scheduler.schedule()
    _, seq_group_0 = create_dummy_prompt(
        "0", prompt_length=7, block_size=4
    )
    _, seq_group_1 = create_dummy_prompt(
        "1", prompt_length=7, block_size=4
    )
    scheduler.add_seq_group(seq_group_0)
    scheduler.add_seq_group(seq_group_1)
    metas, out, _ = scheduler.schedule()
    execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=metas,
                blocks_to_swap_in=out.blocks_to_swap_in,
                blocks_to_swap_out=out.blocks_to_swap_out,
                blocks_to_copy=out.blocks_to_copy,
                num_lookahead_slots=out.num_lookahead_slots,
                running_queue_size=out.running_queue_size,
            )
    outputs = await executor.execute_model_async(execute_model_req)
    assert len(outputs[0].outputs) == 2

@pytest.mark.asyncio
async def test_backend(ray_env):
    # TODO(ZeldaHuang): add tests for BackendSimVLLM methods
    # (currently BackendSimVLLM is just a wrapper of BackendVLLM)
    engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True)
    migration_config = MigrationConfig("SR", "gloo", 16, 1, 4, 5, 20)

    request_output_queue_type = QueueType.RAYQUEUE
    que, server_info = request_output_queue_server(request_output_queue_type)
    asyncio.create_task(que.run_server_loop())
    class DummyActor:
        def __init__(self):
            pass
    dummy_actor = ray.remote(num_cpus=1,
                             name="instance_0",
                             namespace='llumnix',
                             max_concurrency=4)(DummyActor)
    dummy_actor = dummy_actor.remote()
    placement_group = initialize_placement_group(get_placement_group_name("0"), num_cpus=2, num_gpus=0, detached=True)
    sim_backend = MockBackendSim(instance_id="0",
                                 request_output_queue_type=request_output_queue_type,
                                 migration_config=migration_config,
                                 placement_group=placement_group,
                                 profiling_result_file_path="",
                                 engine_args=engine_args)

    sampling_params = SamplingParams(top_k=1, temperature=0, ignore_eos=True, max_tokens=100)
    request_id0 = random_uuid()
    sim_backend.add_request(request_id0, server_info, math.inf, "hello world", sampling_params)

    async def check_output_len():
        request_output_queue = que
        finished = False
        output = None
        while not finished:
            request_outputs = await request_output_queue.get()
            for request_output in request_outputs:
                output = request_output.outputs[0]
                finished = request_output.finished
        assert output is not None and len(output.token_ids)==100

    await check_output_len()

    que.cleanup()
