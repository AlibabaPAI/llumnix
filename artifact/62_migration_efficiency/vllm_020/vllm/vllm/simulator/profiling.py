"""Profile the running time and memory usage of models"""
from collections import namedtuple
import dataclasses
import pickle
import pandas as pd
from typing import List, Dict, Union, Any
from vllm.utils import InferenceType


# 2D parallel configuration
# (tensor parallel, pipeline parallel)
SimParallelConfig = namedtuple("SimParallelConfig", ("tp", "pp"))
# vllm blocks gpu cache configuration
SimCacheConfig = namedtuple("SimCacheConfig", ("gpu", "gpu_memory_utilization", "block_size", "max_num_batched_tokens"))

@dataclasses.dataclass
class LatencyMemData:
    # The latency of each stage
    # Type: Dict[(batch_size, tot_seq_len) -> List[stage_latency]]
    prefill_latency: Dict
    # Type: Dict[(batch_size, tot_seq_len) -> List[stage_latency]]
    decode_latency: Dict
    # Type: Dict[cache_config -> num_gpu_blocks]
    cache_dict: Dict
    # Type: Dict[gpu -> bandwidth(GB/s)]
    migrate_bw: Dict
    # Type: Dict[gpu -> bandwidth(GB/s)]
    swap_bw: Dict
    # Metadata for parallel strategy
    metadata: Any = None

    def add_latency_result(self, inference_type: InferenceType, batch_size: int, latency: List[float]):
        if inference_type == InferenceType.PREFILL:
            self.prefill_latency[batch_size] = latency
        else:
            self.decode_latency[batch_size] = latency

    def add_cache_result(self, cache_config: SimCacheConfig, num_blocks):
        self.cache_dict[cache_config] = num_blocks


@dataclasses.dataclass
class ProfilingResult:
    """Store the profiling result of a model."""
    model_name: str
    # The latency and memory usage of each pipeline stage on GPUs.
    # type: Dict[parallel_config -> latency_mem]
    para_dict: Dict
    # The latency of preprocess on CPU.
    preprocess_cpu: float = 0.0
    # The latency of postprocess on CPU.
    postprocess_cpu: float = 0.0

    def add_latency_result(self, parallel_config: SimParallelConfig, inference_type: InferenceType, batch_size: int,
                   stage_latency: List[float], metadata: Any = None):
        """Add or overwrite the profiling results of a model."""
        if parallel_config not in self.para_dict:
            self.para_dict[parallel_config] = LatencyMemData(
                metadata=metadata, prefill_latency={}, decode_latency={}, cache_dict={}, migrate_bw={"a10":15}, swap_bw={"a10":22})
            if inference_type == InferenceType.PREFILL:
                self.para_dict[parallel_config].prefill_latency = {batch_size: stage_latency}
            else:
                self.para_dict[parallel_config].decode_latency = {batch_size: stage_latency}
        else:
            self.para_dict[parallel_config].add_latency_result(inference_type, batch_size,
                                                       stage_latency)
        

    def add_cache_result(self, parallel_config: SimParallelConfig, cache_config: SimCacheConfig, num_blocks: int,
                     metadata: Any = None):
        if parallel_config not in self.para_dict:
            self.para_dict[parallel_config] = LatencyMemData(
                metadata=metadata, prefill_latency={}, decode_latency={}, cache_dict={}, migrate_bw={"a10":15}, swap_bw={"a10":22})
        self.para_dict[parallel_config].add_cache_result(cache_config, num_blocks)



class ProfilingDatabase:
    """Store the profiling results of all the models"""
    def __init__(self, database_filename: str, new_database: bool = False):
        # The file that backs up the profiling results.
        self.database_filename = database_filename
        # Dict[model_name -> ProfilingResult]
        self.results = {}
        if not new_database:
            with open(database_filename, "rb") as f:
                self.results = pickle.load(f)

    def get(self, model_name: str) -> ProfilingResult:
        return self.results.get(model_name)

    def update(self, result: ProfilingResult):
        self.results[result.model_name] = result

    def _extract_data(self, row):
        """Extract the profiling results from a row of the profiling CSV file."""
        inference_type = InferenceType.PREFILL if row["inference_type"] == "prefill" else InferenceType.DECODE
        stage_latencies = list(map(float, row["StageLatencies(ms)"].strip("[]").split(",")))
        parallel_config = SimParallelConfig(int(row["TP"]),int(row["PP"]))
        return row["ModelName"], parallel_config, (int(row["BS"]),int(row["len_sum"])), stage_latencies, inference_type

    def update_from_csv(self, file_name: str):
        df = pd.read_csv(file_name)
        # read lines
        for idx,row in df.iterrows():
            model_name, parallel_config, batch_size, stage_latencies, inference_type = self._extract_data(row)
            if model_name not in self.results:
                self.results[model_name] = ProfilingResult(model_name,{})
            self.results[model_name].add_latency_result(parallel_config, inference_type, batch_size, stage_latencies)

    def materialize(self):
        """Write the profiling results to the database file."""
        with open(self.database_filename, "wb") as f:
            pickle.dump(self.results, f)


# database = ProfilingDatabase("profiling_result_new.pkl", True)
# database.update_from_csv("/mnt/wencong.xwc/huangziming/vllm/benchmarks-repo/results/prefill_poisson_512|8192_512|2048_16/disable_migrate_2023-11-22_17:16:11.log_decode_inference.csv")
# database.update_from_csv("/mnt/wencong.xwc/huangziming/vllm/benchmarks-repo/results/prefill_poisson_512|8192_512|2048_16/disable_migrate_2023-11-22_17:16:11.log_prefill_inference.csv")
# model_result = database.get('llama-7b')
# model_result.add_cache_result(SimParallelConfig(1,1),SimCacheConfig("a10", 0.90, 16, 4400), 862)
# model_result.add_cache_result(SimParallelConfig(1,1),SimCacheConfig("a10", 0.90, 16, 8700), 831)
# model_result.add_cache_result(SimParallelConfig(1,1),SimCacheConfig("a10", 0.90, 16, 7500), 832)
# model_result.add_cache_result(SimParallelConfig(1,1),SimCacheConfig("a10", 0.90, 16, 12000), 792)

# database.materialize()
