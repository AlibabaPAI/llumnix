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

from llumnix.backends.bladellm.utils import check_engine_args
from llumnix.arg_utils import LlumnixEntrypointsArgs, EngineManagerArgs
from llumnix.logger import init_logger

logger = init_logger(__name__)

def get_args(llumnixCfg, llumnixParser, engine_args):
    llumnix_entrypoints_args = LlumnixEntrypointsArgs.from_llumnix_config(llumnixCfg)
    LlumnixEntrypointsArgs.check_args(llumnix_entrypoints_args, llumnixParser)
    engine_manager_args = EngineManagerArgs.from_llumnix_config(llumnixCfg)
    EngineManagerArgs.check_args(engine_manager_args, llumnixParser)
    check_engine_args(engine_args, engine_manager_args)

    logger.info("llumnix_entrypoints_args: {}".format(llumnix_entrypoints_args))
    logger.info("engine_manager_args: {}".format(engine_manager_args))
    logger.info("engine_args: {}".format(engine_args))

    return llumnix_entrypoints_args, engine_manager_args, engine_args
