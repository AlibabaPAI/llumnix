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
import hashlib

from blade_llm.service.args import ServingArgs
from blade_llm.model.config_utils import load_config


from llumnix.logger import init_logger
from llumnix.arg_utils import EngineManagerArgs

logger = init_logger(__name__)

def detect_unsupported_feature(engine_args: ServingArgs) -> None:
    unsupported_feature = None
    if engine_args.enable_lora:
        unsupported_feature = "multi-lora serving"
    elif not engine_args.disable_prompt_cache:
        unsupported_feature = "automatic prompt caching"
    elif engine_args.use_sps:
        unsupported_feature = "speculative decoding"
    if unsupported_feature:
        raise ValueError(f'Unsupported feature: Llumnix does not support "{unsupported_feature}" currently.')

def check_engine_args(engine_args: ServingArgs, engine_manager_args: EngineManagerArgs) -> None:
    migration_config = engine_manager_args.create_migration_config()
    if (engine_args.tensor_parallel_size > 1 or engine_args.tensor_parallel_size > 1) and migration_config.migration_backend == 'nccl':
        # TODO(s5u13b): fix logger
        print("Llumnix does not support TP or PP enabled model when the migration backend is nccl, change migration backend to gloo.")
        engine_manager_args.migration_backend = 'gloo'
    detect_unsupported_feature(engine_args)

# instance_id is string format in Llumnix while Bladellm only accepts int format.
def string_to_int(string: str) -> int:
    """
    Convert a string to an integer.
    """
    hash_object = hashlib.sha256(string.encode())
    hex_dig = hash_object.hexdigest()
    print(hex_dig)  # 输出十六进制表示的哈希值

    # 如果需要整数表示，可以将其转换为整数
    int_value = int(hex_dig, 16)
    return int_value

def get_model_conf(args: ServingArgs):
    model_conf = None
    try:
        model_conf = load_config(args.load_model_options.model)
        model_conf.verify_with_parallel_config(
            args.tensor_parallel_size, args.pipeline_parallel_size, args.enable_hybrid_dp
        )
    except Exception as e:
        raise type(e)("Failed to load model config when init AsyncLLMEngine: {}", str(e)) from e
    return model_conf