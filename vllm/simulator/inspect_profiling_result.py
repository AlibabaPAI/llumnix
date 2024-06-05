"""Inspect the profiling database."""
import pickle

import numpy as np

from vllm.simulator.profiling import *
# prof = pickle.load(open("profiling_result.pkl", "rb"))

parallel_configs = SimParallelConfig(1, 1)
bs = 16

#for model_name in ["bert-1.3b", "bert-2.6b", "bert-6.7b", "moe-1.3b", "moe-2.4b", "moe-7.1b"]:
database = ProfilingDatabase("profiling_result.pkl", False)
profiling_result = database.get("llama-7b")
print(profiling_result.para_dict[SimParallelConfig(1,1)].prefill_latency.keys())
for model_name in ["llama-7b"]:
    for parallel_config in [SimParallelConfig(1,1)]:
        for bs in [8, 16 ,32]:
            pass
