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
import torch
import numpy as np

from blade_llm.service.args import ServingArgs
from blade_llm.model.config_utils import load_config
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2


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
    if (engine_args.tensor_parallel_size > 1 or engine_args.tensor_parallel_size > 1) and \
        migration_config.migration_backend == 'nccl':
        logger.info("Llumnix does not support TP or PP enabled model when the migration backend is nccl, \
                    change migration backend to gloo.")
        engine_manager_args.migration_backend = 'gloo'
    detect_unsupported_feature(engine_args)

# instance_id is string format in Llumnix while Bladellm only accepts int format.
def string_to_int(string: str) -> int:
    """
    Convert a string to an integer.
    """
    hash_object = hashlib.sha256(string.encode())
    hex_dig = hash_object.hexdigest()
<<<<<<< HEAD
=======

>>>>>>> 2e29f13 (Your commit message)
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



def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor is not None:
        return tensor.numpy().tobytes()
    return None


def tensor_to_tensor_message(tensor: torch.Tensor) -> migration_worker_pb2.TensorMessage:
    if tensor is not None:
        tensor_message = migration_worker_pb2.TensorMessage(
            data=tensor.cpu().numpy().tobytes(), 
            shape=list(tensor.shape),            
            dtype=str(tensor.dtype),          
            device=str(tensor.device)    
        )
        return tensor_message
    else:
        return None


def tensor_message_to_tensor(message: migration_worker_pb2.TensorMessage) -> torch.Tensor:
    # print("crewve",message.data,message.dtype,message.shape)
    if message is None:
        return None
    dtype_str = message.dtype
    if dtype_str in dtype_mapping:
        np_dtype = dtype_mapping[dtype_str]
    np_array = np.frombuffer(message.data, dtype=np_dtype).reshape(tuple(message.shape))
    tensor = torch.from_numpy(np_array)
    
    if message.device.startswith("cuda"):
        tensor = tensor.to(torch.device(message.device))
    return tensor

dtype_mapping = {
    'torch.float32': 'float32',
    'torch.float64': 'float64',
    'torch.float16': 'float16',
    'torch.int8': 'int8',
    'torch.int16': 'int16',
    'torch.int32': 'int32',
    'torch.int64': 'int64',
    'torch.uint8': 'uint8',
}