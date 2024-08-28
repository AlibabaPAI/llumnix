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

import vllm
from vllm import *

from llumnix.server_info import ServerInfo
from llumnix.entrypoints.llumnix_utils import (launch_ray_cluster, connect_to_ray_cluster,
                                               init_manager, init_llumlets)
from llumnix.arg_utils import EngineManagerArgs
from llumnix.llm_engine_manager import LLMEngineManager
from llumnix.llumlet.llumlet import Llumlet

from .version import __version__

__all__ = [
    "__version__",
    "ServerInfo",
    "launch_ray_cluster",
    "connect_to_ray_cluster",
    "init_manager",
    "init_llumlets",
    "EngineManagerArgs",
    "LLMEngineManager",
    "Llumlet"
]

__all__.extend(getattr(vllm, "__all__", []))
