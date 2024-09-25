from vllm import EngineArgs
from vllm.sequence import ExecuteModelRequest

from llumnix.backends.vllm.executor import SimGPUExecutor
from llumnix.backends.profiling import LatencyMemData
from .utils import create_dummy_prompt, initialize_scheduler


def test_executor():
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
            vision_language_config=engine_config.vision_language_config,
            speculative_config=engine_config.speculative_config,
            load_config=engine_config.load_config)
    scheduler = initialize_scheduler()
    scheduler.schedule()
    _, seq_group_0 = create_dummy_prompt(
        "0", prompt_length=7, block_size=4
    )
    _, seq_group_1 = create_dummy_prompt(
        "1", prompt_length=7, block_size=4
    )
    scheduler.add_seq_group(seq_group_0)
    scheduler.add_seq_group(seq_group_1)
    metas, out = scheduler.schedule()
    execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=metas,
                blocks_to_swap_in=out.blocks_to_swap_in,
                blocks_to_swap_out=out.blocks_to_swap_out,
                blocks_to_copy=out.blocks_to_copy,
                num_lookahead_slots=out.num_lookahead_slots,
                running_queue_size=out.running_queue_size,
            )
    outputs = executor.execute_model(execute_model_req)
    assert len(outputs[0].outputs) == 2

def test_backend():
    pass
