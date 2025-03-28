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
from typing import Dict, List, Optional
import torch

from vllm.config import ModelConfig, ParallelConfig
from vllm.sampling_params import SamplingType
from vllm.model_executor.layers.sampler import SamplingMetadata, SamplingTensors, SampleResultArgsType, SampleReturnType, \
                                                SampleResultsDictType, SampleMetadataType, MultinomialSamplesType, \
                                                flashinfer_top_k_top_p_sampling, _top_k_top_p_multinomial_with_flashinfer, \
                                                VLLM_INVALID_TOKEN_ID, _multinomial, _modify_greedy_probs_inplace, get_pythonized_sample_results

from llumnix.logging.logger import init_logger

logger = init_logger(__name__)


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
    sampling_tensors: SamplingTensors,
    include_gpu_probs_tensor: bool,
    modify_greedy_probs: bool,
) -> SampleReturnType:
    '''Torch-oriented _sample() implementation.

    Single-step scheduling:
    * Perform GPU-side sampling computation
    * Immediately Pythonize sampling result

    Multi-step scheduling:
    * Perform GPU-side sampling computation
    * Defer Pythonization & preserve GPU-side
      tensors required for Pythonization
    '''

    categorized_seq_group_ids: Dict[SamplingType,
                                    List[int]] = {t: []
                                                  for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: SampleResultsDictType = {}
    sample_metadata: SampleMetadataType = {}
    multinomial_samples: MultinomialSamplesType = {}
    greedy_samples: Optional[torch.Tensor] = None
    beam_search_logprobs: Optional[torch.Tensor] = None

    # Create output tensor for sampled token ids.
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.full((logprobs.shape[0], 1),
                                              VLLM_INVALID_TOKEN_ID,
                                              dtype=torch.long,
                                              device=logprobs.device)
    else:
        sampled_token_ids_tensor = None

    # Counterintiutively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type]
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

            if sampled_token_ids_tensor is not None:
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
            max_n_in_batch = 1
            for seq_group in seq_groups:
                if seq_group.is_prompt:
                    sampling_params = seq_group.sampling_params
                    max_n_in_batch = max(max_n_in_batch, sampling_params.n)
            seq_groups_arg = (None if sampling_type == SamplingType.RANDOM else
                              seq_groups)

            if flashinfer_top_k_top_p_sampling is not None:
                multinomial_samples[
                    sampling_type] = _top_k_top_p_multinomial_with_flashinfer(
                        probs[long_sample_indices],
                        sampling_tensors.top_ks[long_sample_indices],
                        sampling_tensors.top_ps[long_sample_indices],
                        max_n_in_batch,
                        seq_groups_arg,
                    )
            else:
                multinomial_samples[sampling_type] = _multinomial(
                    probs[long_sample_indices],
                    max_n_in_batch,
                    seq_groups=seq_groups_arg)

            if sampled_token_ids_tensor is not None:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[long_sample_indices] = \
                    multinomial_samples[sampling_type].to(torch.long)

        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    # Encapsulate arguments for computing Pythonized sampler
    # results, whether deferred or otherwise.
    maybe_deferred_args = SampleResultArgsType(
        sampling_metadata=sampling_metadata,
        sample_metadata=sample_metadata,
        multinomial_samples=multinomial_samples,
        greedy_samples=greedy_samples,
        beam_search_logprobs=beam_search_logprobs,
        sample_results_dict=sample_results_dict)

    if not sampling_metadata.skip_sampler_cpu_output:
        # GPU<->CPU sync happens here.
        torch.cuda.current_stream().synchronize()
        # This also converts the sampler output to a Python object.
        # Return Pythonized sampler result & sampled token ids
        return get_pythonized_sample_results(
            maybe_deferred_args), sampled_token_ids_tensor

    # Defer sampler result Pythonization; return deferred
    # Pythonization args & sampled token ids
    return (
        maybe_deferred_args,
        sampled_token_ids_tensor,
    )


def scheduler_lock(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.scheduler_lock:
            return func(self, *args, **kwargs)
    return wrapper
