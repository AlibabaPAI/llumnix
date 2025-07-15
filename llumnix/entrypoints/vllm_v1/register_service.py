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

import argparse

from llumnix.arg_utils import RegisterServiceArgs, save_engine_args
from llumnix.entrypoints.vllm_v1.arg_utils import (
    add_engine_cli_args,
    get_engine_args,
    VLLMV1EngineArgs,
    LlumnixArgumentParser,
)


if __name__ == "__main__":
    from vllm.utils import FlexibleArgumentParser
    parser: FlexibleArgumentParser = FlexibleArgumentParser()
    RegisterServiceArgs.add_cli_args(parser)

    parser = add_engine_cli_args(parser)
    cli_args = parser.parse_args()
    engine_args = get_engine_args(cli_args)
    vllm_v1_engine_args = VLLMV1EngineArgs(engine_args)

    save_engine_args(cli_args.engine_type, cli_args.save_path, vllm_v1_engine_args, cli_args.save_key)
