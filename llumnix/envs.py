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

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    LLUMNIX_CONFIGURE_LOGGING: int = 1
    LLUMNIX_LOGGING_CONFIG_PATH: Optional[str] = None
    LLUMNIX_LOGGING_LEVEL: str = "INFO"
    LLUMNIX_LOGGING_PREFIX: str = "Llumnix"
    LLUMNIX_LOG_STREAM: int = 1
    LLUMNIX_LOG_NODE_PATH: str = ""


environment_variables: Dict[str, Callable[[], Any]] = {
    # Logging configuration
    # If set to 0, llumnix will not configure logging
    # If set to 1, llumnix will configure logging using the default configuration
    #    or the configuration file specified by LLUMNIX_LOGGING_CONFIG_PATH
    "LLUMNIX_CONFIGURE_LOGGING":
    lambda: int(os.getenv("LLUMNIX_CONFIGURE_LOGGING", "1")),
    "LLUMNIX_LOGGING_CONFIG_PATH":
    lambda: os.getenv("LLUMNIX_LOGGING_CONFIG_PATH"),

    # this is used for configuring the default logging level
    "LLUMNIX_LOGGING_LEVEL":
    lambda: os.getenv("LLUMNIX_LOGGING_LEVEL", "INFO"),

    # if set, LLUMNIX_LOGGING_PREFIX will be prepended to all log messages
    "LLUMNIX_LOGGING_PREFIX":
    lambda: os.getenv("LLUMNIX_LOGGING_PREFIX", ""),

    # if set, llumnix will routing all logs to stream
    "LLUMNIX_LOG_STREAM":
    lambda: os.getenv("LLUMNIX_LOG_STREAM", "1"),
    # if set, llumnix will routing all node logs to this path
    "LLUMNIX_LOG_NODE_PATH":
    lambda: os.getenv("LLUMNIX_LOG_NODE_PATH", ""),
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
