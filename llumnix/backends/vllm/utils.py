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

from functools import wraps
from typing import Dict, List, Optional, Tuple
import torch

from vllm.config import ModelConfig, ParallelConfig
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sampling_params import SamplingType
from vllm.model_executor.layers.sampler import SampleResultType, _multinomial, _greedy_sample, _random_sample,\
                                               _modify_greedy_probs_inplace, _beam_search_sample

from llumnix.logger import init_logger
from llumnix.arg_utils import EngineManagerArgs

logger = init_logger(__name__)


def detect_unsupported_feature(engine_args: EngineArgs) -> None:
    unsupported_feature = None
    if engine_args.enable_lora:
        unsupported_feature = "multi-lora serving"
    elif engine_args.enable_prefix_caching:
        unsupported_feature = "automatic prefix caching"
    elif engine_args.enable_chunked_prefill:
        unsupported_feature = "chunked prefill"
    elif engine_args.use_v2_block_manager or engine_args.speculative_model:
        unsupported_feature = "speculative decoding"
    if unsupported_feature:
        raise ValueError(f'Unsupported feature: Llumnix does not support "{unsupported_feature}" currently.')

def check_engine_args(engine_args: AsyncEngineArgs, engine_manager_args: EngineManagerArgs) -> None:
    assert engine_args.engine_use_ray and engine_args.worker_use_ray, \
            ("In Llumnix, engine and worker must be ray actor.")
    migration_config = engine_manager_args.create_migration_config()
    engine_config = engine_args.create_engine_config()
    parallel_config = engine_config.parallel_config
    if parallel_config.world_size > 1 and migration_config.migration_backend == 'nccl':
        # TODO(s5u13b): fix logger
        print("Llumnix does not support TP or PP enabled model when the migration backend is nccl, change migration backend to gloo.")
        engine_manager_args.migration_backend = 'gloo'
    detect_unsupported_feature(engine_args)

def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_layers = model_config.get_num_layers(parallel_config)

    key_cache_block = block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_layers * (key_cache_block + value_cache_block)
    dtype_size = _get_dtype_size(model_config.dtype)
    return dtype_size * total

# overwrite vllm function to adapt llumnix
def _sample_with_torch(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    include_gpu_probs_tensor: bool,
    modify_greedy_probs: bool,
) -> Tuple[SampleResultType, Optional[torch.Tensor]]:
    categorized_seq_group_ids: Dict[SamplingType,
                                    List[int]] = {t: []
                                                  for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}
    multinomial_samples = {}

    # Create output tensor for sampled token ids.
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.empty(logprobs.shape[0],
                                               1,
                                               dtype=torch.long,
                                               device=logprobs.device)
    else:
        sampled_token_ids_tensor = None

    # Counterintiutively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type][:, 0]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue

        seq_group_id = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
        sample_metadata[sampling_type] = (seq_group_id, seq_groups)
        long_sample_indices = sample_indices.long()
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = torch.argmax(logprobs[long_sample_indices],
                                          dim=-1)

            if include_gpu_probs_tensor:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[
                    long_sample_indices] = greedy_samples.unsqueeze(-1)

            if modify_greedy_probs:
                # If required, modify the probabilities such that sampling from
                # the modified distribution would always sample the argmax
                # token id.
                _modify_greedy_probs_inplace(logprobs, probs,
                                             long_sample_indices,
                                             greedy_samples)

        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            max_best_of_in_batch = 1
            for seq_group in seq_groups:
                if seq_group.is_prompt:
                    sampling_params = seq_group.sampling_params
                    max_best_of_in_batch = max(max_best_of_in_batch,
                                               sampling_params.best_of)
            seeded_args = {} if sampling_type == SamplingType.RANDOM else {
                "seq_groups": seq_groups,
            }

            multinomial_samples[sampling_type] = _multinomial(
                probs[long_sample_indices], max_best_of_in_batch,
                **seeded_args)

            if include_gpu_probs_tensor:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[
                    long_sample_indices] = multinomial_samples[sampling_type]

        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    # GPU<->CPU sync happens in the loop below.
    torch.cuda.current_stream().synchronize()
    # This also converts the sample output to Python objects.
    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        (seq_group_id, seq_groups) = sample_metadata[sampling_type]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(seq_groups, greedy_samples)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = _random_sample(seq_groups,
                                            multinomial_samples[sampling_type])
        elif sampling_type == SamplingType.BEAM:
            sample_results = _beam_search_sample(seq_groups,
                                                 beam_search_logprobs)
        sample_results_dict.update(zip(seq_group_id, sample_results))

    sample_results = [
        sample_results_dict.get(i, ([], []))
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results, sampled_token_ids_tensor

def scheduler_lock(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.scheduler_lock:
            return func(self, *args, **kwargs)
    return wrapper
