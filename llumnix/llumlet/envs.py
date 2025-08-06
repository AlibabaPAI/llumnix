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

# BLLM_KVTRANS_FSNAMING_KEEPALIVE_S 36000
# BLLM_KVTRANS_FSNAMING_TOLERATE_S 360000

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

# if TYPE_CHECKING:
#     BLLM_KVTRANS_FSNAMING_KEEPALIVE_S: Optional[str] = None
#     BLLM_KVTRANS_FSNAMING_TOLERATE_S: Optional[str] = None


environment_variables: Dict[str, Callable[[], Any]] = {
    # ================== Llumlet environment variables ==================

    # ACCL
    "ACCL_C4_STATS_MODE":
    lambda: int(os.getenv("ACCL_C4_STATS_MODE")),
    "ACCL_IB_GID_INDEX_FIX":
    lambda: int(os.getenv("ACCL_IB_GID_INDEX_FIX")),
    "ACCL_IB_QPS_LOAD_BALANCE":
    lambda: int(os.getenv("ACCL_IB_QPS_LOAD_BALANCE")),
    "ACCL_IB_SPLIT_DATA_NUM":
    lambda: int(os.getenv("ACCL_IB_SPLIT_DATA_NUM")),
    "ACCL_LOAD_BALANCE":
    lambda: int(os.getenv("ACCL_LOAD_BALANCE")),
    "ACCL_LOG_TIME":
    lambda: int(os.getenv("ACCL_LOG_TIME")),
    "ACCL_LOW_LATENCY_COMBINE_USE_FP8":
    lambda: int(os.getenv("ACCL_LOW_LATENCY_COMBINE_USE_FP8")),
    "ACCL_LOW_LATENCY_OPTIMIZE":
    lambda: int(os.getenv("ACCL_LOW_LATENCY_OPTIMIZE")),
    "ACCL_NORMAL_MODE":
    lambda: os.getenv("ACCL_NORMAL_MODE"),
    "ACCL_PROXY_NTHREADS":
    lambda: int(os.getenv("ACCL_PROXY_NTHREADS")),
    "ACCL_TOPO_FIX":
    lambda: int(os.getenv("ACCL_TOPO_FIX")),

    # DeepGeMM and DeepEP
    "DEEPEP_LL_BUFFER_FP8_OPT":
    lambda: int(os.getenv("DEEPEP_LL_BUFFER_FP8_OPT")),
    "DEEPEP_LL_COMBINE_USE_FP8":
    lambda: int(os.getenv("DEEPEP_LL_COMBINE_USE_FP8")),
    "DEEPEP_LL_COMBINE_USE_NVL":
    lambda: int(os.getenv("DEEPEP_LL_COMBINE_USE_NVL")),
    "DEEPEP_LL_DISPATCH_USE_NVL":
    lambda: int(os.getenv("DEEPEP_LL_DISPATCH_USE_NVL")),
    "DG_CACHE_DIR":
    lambda: os.getenv("DG_CACHE_DIR"),

    # NCCL and NVSHMEM and CUDA
    "NCCL_CUMEM_ENABLE":
    lambda: int(os.getenv("NCCL_CUMEM_ENABLE")),
    "NCCL_DEBUG":
    lambda: os.getenv("NCCL_DEBUG"),
    "NCCL_IB_QPS_PER_CONNECTION":
    lambda: int(os.getenv("NCCL_IB_QPS_PER_CONNECTION")),
    "NCCL_IB_TC":
    lambda: int(os.getenv("NCCL_IB_TC")),
    "NCCL_MIN_NCHANNELS":
    lambda: int(os.getenv("NCCL_MIN_NCHANNELS")),
    "NCCL_NVLS_ENABLE":
    lambda: int(os.getenv("NCCL_NVLS_ENABLE")),
    "NCCL_SOCKET_FAMILY":
    lambda: os.getenv("NCCL_SOCKET_FAMILY"),
    "NCCL_SOCKET_IFNAME":
    lambda: os.getenv("NCCL_SOCKET_IFNAME"),
    "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME":
    lambda: os.getenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"),
    "CUDA_DEVICE_MAX_CONNECTIONS":
    lambda: int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS")),

    # ray
    "RAY_CGRAPH_get_timeout":
    lambda: int(os.getenv("RAY_CGRAPH_get_timeout")),
    "RAY_CGRAPH_submit_timeout":
    lambda: int(os.getenv("RAY_CGRAPH_submit_timeout")),
    "RAY_DEDUP_LOGS":
    lambda: int(os.getenv("RAY_DEDUP_LOGS")),

    # other
    "SHLVL":
    lambda: int(os.getenv("SHLVL")),

    # torch
    "TORCH_CUDA_ARCH_LIST":
    lambda: os.getenv("TORCH_CUDA_ARCH_LIST"),

    # vllm
    "VLLM_ATTENTION_BACKEND":
    lambda: os.getenv("VLLM_ATTENTION_BACKEND"),
    "VLLM_CAPTURE_DRAFT_MODEL":
    lambda: int(os.getenv("VLLM_CAPTURE_DRAFT_MODEL")),
    "VLLM_DP_MASTER_IP":
    lambda: os.getenv("VLLM_DP_MASTER_IP"),
    "VLLM_DP_MASTER_PORT":
    lambda: int(os.getenv("VLLM_DP_MASTER_PORT")),
    "VLLM_DP_META_USE_CPU_GROUP":
    lambda: int(os.getenv("VLLM_DP_META_USE_CPU_GROUP")),
    "VLLM_ENABLE_LLUMNIX":
    lambda: int(os.getenv("VLLM_ENABLE_LLUMNIX")),
    "VLLM_ENABLE_PYTHON_GC_INTERVAL":
    lambda: int(os.getenv("VLLM_ENABLE_PYTHON_GC_INTERVAL")),
    "VLLM_ENABLE_TBO_OPT":
    lambda: int(os.getenv("VLLM_ENABLE_TBO_OPT")),
    "VLLM_ENABLE_TORCH_COMPILE":
    lambda: int(os.getenv("VLLM_ENABLE_TORCH_COMPILE")),
    "VLLM_FORCE_DETOKENIZE":
    lambda: int(os.getenv("VLLM_FORCE_DETOKENIZE")),
    "VLLM_FP8_USE_BLADNN":
    lambda: int(os.getenv("VLLM_FP8_USE_BLADNN")),
    "VLLM_FUSED_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_FUSED_MOE_CHUNK_SIZE")),
    "VLLM_MASTER_ADDR":
    lambda: os.getenv("VLLM_MASTER_ADDR"),
    "VLLM_MLA_FP8_ATTENTION":
    lambda: int(os.getenv("VLLM_MLA_FP8_ATTENTION")),
    "VLLM_MOE_BALANCED_GATING":
    lambda: int(os.getenv("VLLM_MOE_BALANCED_GATING")),
    "VLLM_MOE_EXPERT_REMAPPING":
    lambda: os.getenv("VLLM_MOE_EXPERT_REMAPPING"),
    "VLLM_MOE_EXPERTS_OVERLAP":
    lambda: int(os.getenv("VLLM_MOE_EXPERTS_OVERLAP")),
    "VLLM_MOE_RANDOM_GATING":
    lambda: int(os.getenv("VLLM_MOE_RANDOM_GATING")),
    "VLLM_MOE_USE_BLADNN":
    lambda: int(os.getenv("VLLM_MOE_USE_BLADNN")),
    "VLLM_MOE_USE_DEEPEP":
    lambda: int(os.getenv("VLLM_MOE_USE_DEEPEP")),
    "VLLM_PD_TRANSFER_TIMEOUT_SECONDS":
    lambda: int(os.getenv("VLLM_PD_TRANSFER_TIMEOUT_SECONDS")),
    "VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS":
    lambda: int(os.getenv("VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS")),
    "VLLM_QUANTIZATION_LAYER_WISE":
    lambda: int(os.getenv("VLLM_QUANTIZATION_LAYER_WISE")),
    "VLLM_USE_DEEP_GEMM":
    lambda: int(os.getenv("VLLM_USE_DEEP_GEMM")),
    "VLLM_USE_FLASHINFER_SAMPLER":
    lambda: int(os.getenv("VLLM_USE_FLASHINFER_SAMPLER")),
    "VLLM_USE_V1":
    lambda: int(os.getenv("VLLM_USE_V1")),
}


# pylint: disable=invalid-name
def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# pylint: disable=invalid-name
def __dir__():
    return list(environment_variables.keys())
