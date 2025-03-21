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

from blade_llm.service.args import ServingArgs

from llumnix.arg_utils import EngineManagerArgs
from llumnix.logger import init_logger

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
    assert migration_config.migration_backend in ['grpc', 'kvtransfer'], \
        f'Unsupported migration backend for BladeLLM: {migration_config.migration_backend}'
    detect_unsupported_feature(engine_args)
