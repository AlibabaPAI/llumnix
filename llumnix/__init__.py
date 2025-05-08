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

from llumnix.manager import Manager
from llumnix.server_info import ServerInfo
from llumnix.entrypoints.setup import (launch_ray_cluster,
                                       connect_to_ray_cluster,
                                       init_manager)
from llumnix.arg_utils import ManagerArgs, InstanceArgs, LaunchArgs, EntrypointsArgs
from llumnix.llumlet.llumlet import Llumlet
from llumnix.queue.queue_type import QueueType
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode
from llumnix.version import __version__

__all__ = [
    "__version__",
    "Manager",
    "ServerInfo",
    "launch_ray_cluster",
    "connect_to_ray_cluster",
    "init_manager",
    "ManagerArgs",
    "InstanceArgs",
    "Llumlet",
    "QueueType",
    "BackendType",
    "LaunchArgs",
    "EntrypointsArgs",
    "LaunchMode"
]

try:
    import vllm
    from vllm import *
    __all__.extend(getattr(vllm, "__all__", []))
except ImportError:
    pass

# TODO(KuilongCui): import blade_llm after cuda is ready
# try:
#     import blade_llm
#     from blade_llm import *
#     __all__.extend(getattr(blade_llm, "__all__", []))
# except ImportError:
#     pass
