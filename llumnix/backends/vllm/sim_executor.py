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

from typing import List, Tuple

from vllm.executor.ray_gpu_executor import RayGPUExecutor

from vllm.sequence import Logprob, SequenceOutput, SequenceGroupOutput, SamplerOutput, ExecuteModelRequest
from vllm.config import _GB

from llumnix.logging.logger import init_logger
from llumnix.backends.vllm.utils import get_cache_block_size
from llumnix.backends.profiling import LatencyMemData, SimCacheConfig, model_prefill, model_decode, _pad_to_alignment

logger = init_logger(__name__)


class SimGPUExecutor(RayGPUExecutor):
    latency_mem: LatencyMemData = None

    def __init__(self, *args, **kwargs) -> None:
        RayGPUExecutor.__init__(self, *args, **kwargs)
        self.last_inference_latency = 0
        self.migration_bandwidth = self.latency_mem.migration_bandwidth
        # TODO(ZeldaHuang): add swap bandwidth

        self.cache_block_size = get_cache_block_size(
            self.cache_config.block_size, self.model_config, self.parallel_config)
        self.cache_block_size /= _GB
        self.sim_cache_config = SimCacheConfig(self.cache_config.gpu_memory_utilization,
                                               self.cache_config.block_size,
                                               self.scheduler_config.max_num_batched_tokens)

    def _init_executor(self) -> None:
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_gpu_blocks = self.latency_mem.cache_dict.get(self.sim_cache_config, 880)
        num_cpu_blocks = 2048
        return (num_gpu_blocks, num_cpu_blocks)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        logger.info("# GPU blocks: {}, # CPU blocks: {}".format(num_gpu_blocks, num_cpu_blocks))

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        prefill_seq_len = 0
        decode_seq_len = 0
        decode_bs = 0
        for meta_data in execute_model_req.seq_group_metadata_list:
            if meta_data.is_prompt:
                prefill_seq_len += meta_data.token_chunk_size
            else:
                decode_bs += meta_data.token_chunk_size
                decode_seq_len += list(meta_data.seq_data.values())[0].get_len()
        decode_bs = _pad_to_alignment(decode_bs, 8)
        prefill_seq_len = _pad_to_alignment(prefill_seq_len, 8)
        latency = 0
        if prefill_seq_len:
            latency += self.latency_mem.prefill_latency[prefill_seq_len][0] if prefill_seq_len in self.latency_mem.prefill_latency \
                       else model_prefill(prefill_seq_len, *self.latency_mem.prefill_model_params)
        if decode_bs:
            decode_meta_data = (decode_bs, decode_seq_len)
            latency += self.latency_mem.decode_latency[decode_meta_data][0] if decode_meta_data in self.latency_mem.decode_latency \
                       else model_decode((decode_bs, decode_seq_len), *self.latency_mem.decode_model_params)
        await asyncio.sleep(latency/1000)
        sampler_outputs = []
        for meta_data in execute_model_req.seq_group_metadata_list:
            samples = []
            for seq_id in meta_data.seq_data.keys():
                dummy_sample_output = SequenceOutput(seq_id, 20, {20: Logprob(1.0)})
                samples.append(dummy_sample_output)
            if samples:
                output = SequenceGroupOutput(samples, None)
                sampler_outputs.append(output)
        return [SamplerOutput(outputs=sampler_outputs)]

    async def send_blocks(self, blocks_len) -> None:
        migration_latency = (self.cache_block_size * blocks_len) / self.migration_bandwidth
        await asyncio.sleep(migration_latency)
