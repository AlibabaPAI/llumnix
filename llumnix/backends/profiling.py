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

"""Profile the running time and memory usage of models"""
import os
import dataclasses

from collections import namedtuple
from typing import List, Dict, Any
from scipy.optimize import curve_fit

import cloudpickle
import pandas as pd
import numpy as np

from llumnix.backends.backend_interface import BackendType
from llumnix.llumlet.request import RequestInferenceType

# 2D parallel configuration
# (gpu, tensor parallel, pipeline parallel)
SimParallelConfig = namedtuple("SimParallelConfig", ("tp", "pp"))
# vllm blocks gpu cache configuration
SimCacheConfig = namedtuple("SimCacheConfig", ("gpu_memory_utilization", "block_size", "max_num_batched_tokens"))

def _pad_to_alignment(x, multiple_of):
    return x + ((-1*x) % multiple_of)


@dataclasses.dataclass
class LatencyMemData:
    # The latency of each stage
    # Type: Dict[tot_seq_len -> List[stage_latency]]
    prefill_latency: Dict
    # Type: Dict[(batch_size, tot_seq_len) -> List[stage_latency]]
    decode_latency: Dict
    # Type: Dict[SimCacheConfig -> num_gpu_blocks]
    cache_dict: Dict
    # Type: bandwidth(GB/s)
    migration_bandwidth: float = 1.0
    # Type: bandwidth(GB/s)
    swap_bandwidth: float = 32.0
    # Metadata for parallel strategy
    metadata: Any = None
    # model params for fitting latency
    decode_model_params: Any = None
    prefill_model_params: Any = None

    def add_latency_result(self, inference_type: RequestInferenceType, batch_size: int, tot_seq_len: int, latency: List[float]):
        if inference_type == RequestInferenceType.PREFILL:
            self.prefill_latency[tot_seq_len] = latency
        else:
            self.decode_latency[(batch_size, tot_seq_len)] = latency

    def add_cache_result(self, cache_config: SimCacheConfig, num_blocks):
        self.cache_dict[cache_config] = num_blocks

    def get_prefill_dict_kv(self):
        return map(list, zip(*self.prefill_latency.items()))

    def get_decode_dict_kv(self):
        return map(list, zip(*self.decode_latency.items()))


@dataclasses.dataclass
class ProfilingResult:
    """Store the profiling result of a model."""
    model_name: str
    # The latency and memory usage of each pipeline stage on GPUs.
    para_dict: Dict[SimParallelConfig, LatencyMemData]
    # The latency of preprocess on CPU.
    preprocess_cpu: float = 0.0
    # The latency of postprocess on CPU.
    postprocess_cpu: float = 0.0

    def add_latency_result(self, parallel_config: SimParallelConfig, inference_type: RequestInferenceType, batch_size: int,
                           tot_seq_len: int, stage_latency: List[float], metadata: Any = None):
        """Add or overwrite the profiling results of a model."""
        if parallel_config not in self.para_dict:
            self.para_dict[parallel_config] = LatencyMemData(
                metadata=metadata, prefill_latency={}, decode_latency={}, cache_dict={})
            if inference_type == RequestInferenceType.PREFILL:
                self.para_dict[parallel_config].prefill_latency = {tot_seq_len: stage_latency}
            else:
                self.para_dict[parallel_config].decode_latency = {(batch_size, tot_seq_len): stage_latency}
        else:
            self.para_dict[parallel_config].add_latency_result(inference_type, batch_size, tot_seq_len, stage_latency)

    def add_cache_result(self, parallel_config: SimParallelConfig, cache_config: SimCacheConfig, num_blocks: int,
                         metadata: Any = None):
        if parallel_config not in self.para_dict:
            self.para_dict[parallel_config] = LatencyMemData(
                metadata=metadata, prefill_latency={}, decode_latency={}, cache_dict={})
        self.para_dict[parallel_config].add_cache_result(cache_config, num_blocks)

    def fit_from_database(self, parallel_config: SimParallelConfig):
        # fit prefill
        tot_seq_len_list, stage_latency_list = self.para_dict[parallel_config].get_prefill_dict_kv()
        latency_list = [stage_latency[0] for stage_latency in stage_latency_list]
        x = np.array(tot_seq_len_list)
        y = np.array(latency_list)
        params, _ = curve_fit(model_prefill, x, y)
        self.para_dict[parallel_config].prefill_model_params = params
        avg_loss = 0
        for idx, seq_len in enumerate(tot_seq_len_list):
            sim_lat = model_prefill(seq_len, *params)
            avg_loss += abs(sim_lat - latency_list[idx])
        print(f"prefill sim avg_loss={avg_loss/len(latency_list)}")
        # fit decode
        keys, stage_latency_list = self.para_dict[parallel_config].get_decode_dict_kv()
        bs_list, tot_seq_len_list = map(list, zip(*keys))
        latency_list = [stage_latency[0] for stage_latency in stage_latency_list]
        x = np.vstack((np.array(bs_list), np.array(tot_seq_len_list)))
        y = np.array(latency_list)
        params, _ = curve_fit(model_decode, x, y)
        avg_loss = 0
        self.para_dict[parallel_config].decode_model_params = params
        for idx, (bs, seq_len) in enumerate(zip(bs_list, tot_seq_len_list)):
            sim_lat = model_decode((bs, seq_len), *params)
            avg_loss += abs(sim_lat - latency_list[idx])
        print(f"decode sim avg_loss={avg_loss/len(latency_list)}")


class ProfilingDatabase:
    """Store the profiling results of all the models"""
    def __init__(self, database_filename: str, new_database: bool = False):
        # The file that backs up the profiling results.
        self.database_filename = database_filename
        # Dict[model_name -> ProfilingResult]
        self.results: Dict[str, ProfilingResult] = {}
        if not new_database:
            with open(database_filename, "rb") as f:
                self.results = cloudpickle.load(f)

    def get(self, model_name: str) -> ProfilingResult:
        return self.results.get(model_name)

    def update(self, result: ProfilingResult):
        self.results[result.model_name] = result

    def _extract_data(self, row):
        """Extract the profiling results from a row of the profiling CSV file."""
        # assert pp==1
        profiling_data = row["profiling_data"].strip('"()"').split(",")
        inference_type = RequestInferenceType.PREFILL if profiling_data[0] == "'prefill'" else RequestInferenceType.DECODE
        batch_size = _pad_to_alignment(int(profiling_data[1]), 8)
        tot_seq_len =_pad_to_alignment(int(profiling_data[2]), 8)
        stage_latencies = [float(profiling_data[3])]

        return stage_latencies, inference_type, batch_size, tot_seq_len

    def update_from_instance_log(self, file_name: str, model: str, parallel_config: SimParallelConfig):
        df = pd.read_csv(file_name)
        df = df[df['bs'] > 0]
        # read lines
        if model not in self.results:
            self.results[model] = ProfilingResult(model, {})
        for _, row in df.iterrows():
            stage_latencies, inference_type, batch_size, tot_seq_len = self._extract_data(row)
            self.results[model].add_latency_result(parallel_config, inference_type, batch_size, tot_seq_len, stage_latencies)

    def materialize(self):
        """Write the profiling results to the database file."""
        with open(self.database_filename, "wb") as f:
            cloudpickle.dump(self.results, f)

def model_prefill(x, a, b):
    return a * x + b

def model_decode(x, a, b, c):
    bs, tot_seq_len = x
    return a * bs + b * tot_seq_len + c

def get_latency_mem(backend_type: BackendType, profiling_database: ProfilingDatabase, **backend_args):
    assert BackendType.is_sim_backend(backend_type)
    if backend_type == BackendType.SIM_VLLM:
        # TODO(ZeldaHuang): support multi-lora, more device, vision language model
        model_config = backend_args.get("model_config")
        _ = backend_args.get("cache_config")
        parallel_config = backend_args.get("parallel_config")
        _ = backend_args.get("scheduler_config")
        model_name = model_config.model
        # get model_name from model path
        if model_name.endswith('/'):
            model_name = model_name[:-1]
        model_name = os.path.basename(model_name)
        profiling_result: ProfilingResult = profiling_database.get(model_name)
        sim_parallel_config = SimParallelConfig(parallel_config.tensor_parallel_size,
                                                parallel_config.pipeline_parallel_size)
        assert sim_parallel_config in profiling_result.para_dict.keys(), "sim parallel config not in database"
        latency_mem: LatencyMemData = profiling_result.para_dict[sim_parallel_config]
        return latency_mem
    raise ValueError(f'Unsupported simulator backend: {backend_type}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="profiling.pkl")
    parser.add_argument("--log-csv-path", type=str, required=True)
    parser.add_argument("--model", type=str, help="filename of your model, like 'Meta-Llama-3-8B-Instruct'")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8000)
    parser.add_argument("--num-gpu-blocks", type=int, required=True, help="kv cache blocks number")
    parser.add_argument("--new-data", action="store_true")

    args = parser.parse_args()
    args_parallel_config = SimParallelConfig(args.tp, args.pp)
    args_cache_config = SimCacheConfig(args.gpu_memory_utilization, args.block_size, args.max_num_batched_tokens)
    database = ProfilingDatabase(args.database, args.new_data)
    database.update_from_instance_log(args.log_csv_path, args.model, args_parallel_config)
    model_result = database.get(args.model)
    model_result.fit_from_database(parallel_config=args_parallel_config)
    model_result = database.get(args.model)
    model_result.add_cache_result(args_parallel_config, args_cache_config, args.num_gpu_blocks)

    database.materialize()
