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

from llumnix.arg_utils import RegisterServiceArgs, save_engine_args
from llumnix.entrypoints.bladellm.arg_utils import add_engine_cli_args, get_engine_args, BladeLLMEngineArgs
from llumnix.entrypoints.bladellm.serve import launch_job_on_gpu_node

def main():
    # pylint: disable=import-outside-toplevel
    from blade_llm.utils.argparse_helper import PatchedArgumentParser

    parser = PatchedArgumentParser()
    RegisterServiceArgs.add_cli_args(parser)

    parser = add_engine_cli_args(parser)
    cli_args = parser.parse_args()
    engine_args = get_engine_args(cli_args)
    bladellm_engine_args = BladeLLMEngineArgs(engine_args)

    save_engine_args(cli_args.engine_type, cli_args.save_path, bladellm_engine_args, cli_args.save_key)


if __name__ == "__main__":
    launch_job_on_gpu_node(module="llumnix.entrypoints.bladellm.register_service", main_func=main)
