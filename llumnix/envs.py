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
from llumnix.constants import (
    SERVER_READY_TIMEOUT,
    INSTANCE_READY_TIMEOUT,
    DATASET_PATH,
    MODEL_PATH,
    USAGERATIO_BUSY_THRESHOLD,
    REMAININGSTEPS_BUSY_THRESHOLD,
    DECODE_COMPUTE_BOUND_BATCH_SIZE
)

if TYPE_CHECKING:
    HEAD_NODE: Optional[str] = None
    HEAD_NODE_IP: Optional[str] = None
    LLUMNIX_CONFIGURE_LOGGING: int = 1
    LLUMNIX_LOGGING_CONFIG_PATH: Optional[str] = None
    LLUMNIX_LOGGING_LEVEL: str = "INFO"
    LLUMNIX_LOGGING_PREFIX: str = "Llumnix"
    LLUMNIX_LOG_STREAM: int = 1
    LLUMNIX_LOG_NODE_PATH: str = ""
    MODEL_PATH: str = ""
    DATASET_PATH: str = ""
    USAGERATIO_BUSY_THRESHOLD: float = 0.0
    REMAININGSTEPS_BUSY_THRESHOLD: float = 0.0
    DECODE_COMPUTE_BOUND_BATCH_SIZE: float = 0.0



environment_variables: Dict[str, Callable[[], Any]] = {
    # ================== Llumnix environment variables ==================

    # Ray cluster setup configuration
    "HEAD_NODE": lambda: os.getenv("HEAD_NODE"),
    "HEAD_NODE_IP": lambda: os.getenv("HEAD_NODE_IP"),

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
    lambda: os.getenv("LLUMNIX_LOGGING_LEVEL", "DEBUG"),

    # if set, LLUMNIX_LOGGING_PREFIX will be prepended to all log messages
    "LLUMNIX_LOGGING_PREFIX":
    lambda: os.getenv("LLUMNIX_LOGGING_PREFIX", ""),

    # if set, llumnix will routing all logs to stream
    "LLUMNIX_LOG_STREAM":
    lambda: os.getenv("LLUMNIX_LOG_STREAM", "1"),
    # if set, llumnix will routing all node logs to this path
    "LLUMNIX_LOG_NODE_PATH":
    lambda: os.getenv("LLUMNIX_LOG_NODE_PATH", ""),

    "MODEL_PATH":
    lambda: os.getenv("MODEL_PATH", MODEL_PATH),
    "DATASET_PATH":
    lambda: os.getenv("DATASET_PATH", DATASET_PATH),

    # used for scale up
    "SERVER_READY_TIMEOUT":
    lambda: os.getenv("LLUMNIX_SERVER_READY_TIMEOUT", str(SERVER_READY_TIMEOUT)),
    "INSTANCE_READY_TIMEOUT":
    lambda: os.getenv("LLUMNIX_INSTANCE_READY_TIMEOUT", str(INSTANCE_READY_TIMEOUT)),

    # used for load computation
    "USAGERATIO_BUSY_THRESHOLD":
    lambda: os.getenv("USAGERATIO_BUSY_THRESHOLD", str(USAGERATIO_BUSY_THRESHOLD)),
    "REMAININGSTEPS_BUSY_THRESHOLD":
    lambda: os.getenv("REMAININGSTEPS_BUSY_THRESHOLD", str(REMAININGSTEPS_BUSY_THRESHOLD)),
    "DECODE_COMPUTE_BOUND_BATCH_SIZE":
    lambda: os.getenv("DECODE_COMPUTE_BOUND_BATCH_SIZE", str(DECODE_COMPUTE_BOUND_BATCH_SIZE)),
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
