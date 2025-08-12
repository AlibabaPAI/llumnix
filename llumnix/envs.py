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
    KVBLOCKSRATIO_BUSY_THRESHOLD,
    REMAININGSTEPS_BUSY_THRESHOLD,
    DECODE_COMPUTE_BOUND_BATCH_SIZE,
    MISSWAITINGTOKENS_BUSY_THRESHOLD,
    DEFAULT_METRICS_EXPORT_INTERVAL_SEC,
    RAY_TASK_RETRY_DELAY_MS,
    CACHE_TRANSFER_THRESHOLD,
    CACHE_TRANSFER_PENALTY_FACTOR,
    UNIT_READY_TIMEOUT,
    SERVER_START_TIMEOUT,
    UNIT_FAILOVER_TIMEOUT,
    RAY_RPC_TIMEOUT,
    UTILITY_CALL_TIMEOUT
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

    AUTO_SCALE_UP_INTERVAL: float = 0.0
    HEARTBEAT_INTERVAL: float = 0.0
    WAIT_PLACEMENT_GROUP_TIMEOUT: float = 0.0
    CHECK_DEPLOYMENT_STATES_INTERVAL: float = 0.0
    WATCH_DEPLOYMENT_INTERVAL: float = 0.0

    SERVER_READY_TIMEOUT: float = 0.0
    INSTANCE_READY_TIMEOUT: float = 0.0
    UNIT_READY_TIMEOUT: float = 0.0
    SERVER_START_TIMEOUT: float = 0.0

    INIT_CACHED_CLUSTER_ACTORS_INTERVAL: float = 0.0
    UPDATE_CACHED_CLUSTER_ACTORS_INTERVAL: float = 0.0

    METRICS_OUTPUT_TARGET:  str = "logger,eas"
    # 'XXX_METRICS_SAMPLE_EVERY_N_RECORDS = n' means that for every n observe, only one will be actually executed
    MANAGER_METRICS_SAMPLE_EVERY_N_RECORDS:  str = "1"
    GLOBAL_SCHEDULER_METRICS_SAMPLE_EVERY_N_RECORDS:  str = "1"
    LLUMNIX_CLIENT_METRICS_SAMPLE_EVERY_N_RECORDS:  str = "1"
    LLUMLET_METRICS_SAMPLE_EVERY_N_RECORDS: str = "1"
    QUEUE_SERVER_METRICS_SAMPLE_EVERY_N_RECORDS: str = "1"
    LLUMNIX_METRICS_EXPORT_INTERVAL_SEC: str = "15"

    KVBLOCKSRATIO_BUSY_THRESHOLD: float = 0.0
    REMAININGSTEPS_BUSY_THRESHOLD: float = 0.0
    DECODE_COMPUTE_BOUND_BATCH_SIZE: float = 0.0
    MISSWAITINGTOKENS_BUSY_THRESHOLD: float = 0.0
    CACHE_TRANSFER_THRESHOLD: int = 0
    CACHE_TRANSFER_PENALTY_FACTOR: float = 0.0
    UNIT_FAILOVER_TIMEOUT: float = 0.0

    RAY_TASK_RETRY_DELAY_MS: int = 0

    RAY_RPC_TIMEOUT: float = 0.0

    UTILITY_CALL_TIMEOUT: float = 0.0


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
    lambda: os.getenv("LLUMNIX_MODEL_PATH", MODEL_PATH),
    "DATASET_PATH":
    lambda: os.getenv("LLUMNIX_DATASET_PATH", DATASET_PATH),

    "AUTO_SCALE_UP_INTERVAL":
    lambda: os.getenv("LLUMNIX_AUTO_SCALE_UP_INTERVAL", str(AUTO_SCALE_UP_INTERVAL)),
    "HEARTBEAT_INTERVAL":
    lambda: os.getenv("LLUMNIX_HEARTBEAT_INTERVAL", str(HEARTBEAT_INTERVAL)),
    "WAIT_PLACEMENT_GROUP_TIMEOUT":
    lambda: os.getenv("LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT", str(WAIT_PLACEMENT_GROUP_TIMEOUT)),
    "CHECK_DEPLOYMENT_STATES_INTERVAL":
    lambda: os.getenv("LLUMNIX_CHECK_DEPLOYMENT_STATES_INTERVAL", str(CHECK_DEPLOYMENT_STATES_INTERVAL)),
    "WATCH_DEPLOYMENT_INTERVAL":
    lambda: os.getenv("LLUMNIX_WATCH_DEPLOYMENT_INTERVAL", str(WATCH_DEPLOYMENT_INTERVAL)),

    "SERVER_READY_TIMEOUT":
    lambda: os.getenv("LLUMNIX_SERVER_READY_TIMEOUT", str(SERVER_READY_TIMEOUT)),
    "INSTANCE_READY_TIMEOUT":
    lambda: os.getenv("LLUMNIX_INSTANCE_READY_TIMEOUT", str(INSTANCE_READY_TIMEOUT)),
    "UNIT_READY_TIMEOUT":
    lambda: os.getenv("LLUMNIX_UNIT_READY_TIMEOUT", str(UNIT_READY_TIMEOUT)),
    "SERVER_START_TIMEOUT":
    lambda: os.getenv("LLUMNIX_SERVER_START_TIMEOUT", str(SERVER_START_TIMEOUT)),

    # service discorvery
    "INIT_CACHED_CLUSTER_ACTORS_INTERVAL":
    lambda: os.getenv("LLUMNIX_INIT_CACHED_CLUSTER_ACTORS_INTERVAL", str(INIT_CACHED_CLUSTER_ACTORS_INTERVAL)),
    "UPDATE_CACHED_CLUSTER_ACTORS_INTERVAL":
    lambda: os.getenv("LLUMNIX_UPDATE_CACHED_CLUSTER_ACTORS_INTERVAL", str(UPDATE_CACHED_CLUSTER_ACTORS_INTERVAL)),

    # metrics
    "METRICS_OUTPUT_TARGET":
    lambda: os.getenv("LLUMNIX_METRICS_OUTPUT_TARGET", "logger"),
    "MANAGER_METRICS_SAMPLE_EVERY_N_RECORDS":
    lambda: os.getenv("LLUMNIX_MANAGER_METRICS_SAMPLE_EVERY_N_RECORDS", "0"),
    "LLUMNIX_CLIENT_METRICS_SAMPLE_EVERY_N_RECORDS":
    lambda: os.getenv("LLUMNIX_CLIENT_METRICS_SAMPLE_EVERY_N_RECORDS", "0"),
    "LLUMLET_METRICS_SAMPLE_EVERY_N_RECORDS":
    lambda: os.getenv("LLUMNIX_LLUMLET_METRICS_SAMPLE_EVERY_N_RECORDS", "0"),
    "GLOBAL_SCHEDULER_METRICS_SAMPLE_EVERY_N_RECORDS":
    lambda: os.getenv("LLUMNIX_GLOBAL_SCHEDULER_METRICS_SAMPLE_EVERY_N_RECORDS", "0"),
    "QUEUE_SERVER_METRICS_SAMPLE_EVERY_N_RECORDS":
    lambda: os.getenv("LLUMNIX_QUEUE_SERVER_METRICS_SAMPLE_EVERY_N_RECORDS", "0"),
    "LLUMNIX_METRICS_EXPORT_INTERVAL_SEC":
    lambda: os.getenv("LLUMNIX_METRICS_EXPORT_INTERVAL_SEC", str(DEFAULT_METRICS_EXPORT_INTERVAL_SEC)),

    # request scheduling
    "KVBLOCKSRATIO_BUSY_THRESHOLD":
    lambda: os.getenv("LLUMNIX_KVBLOCKSRATIO_BUSY_THRESHOLD", str(KVBLOCKSRATIO_BUSY_THRESHOLD)),
    "REMAININGSTEPS_BUSY_THRESHOLD":
    lambda: os.getenv("LLUMNIX_REMAININGSTEPS_BUSY_THRESHOLD", str(REMAININGSTEPS_BUSY_THRESHOLD)),
    "DECODE_COMPUTE_BOUND_BATCH_SIZE":
    lambda: os.getenv("LLUMNIX_DECODE_COMPUTE_BOUND_BATCH_SIZE", str(DECODE_COMPUTE_BOUND_BATCH_SIZE)),
    "MISSWAITINGTOKENS_BUSY_THRESHOLD":
    lambda: os.getenv("LLUMNIX_MISSWAITINGTOKENS_BUSY_THRESHOLD", str(MISSWAITINGTOKENS_BUSY_THRESHOLD)),
    "CACHE_TRANSFER_THRESHOLD":
    lambda: os.getenv("LLUMNIX_CACHE_TRANSFER_THRESHOLD", str(CACHE_TRANSFER_THRESHOLD)),
    "CACHE_TRANSFER_PENALTY_FACTOR":
    lambda: os.getenv("LLUMNIX_CACHE_TRANSFER_PENALTY_FACTOR", str(CACHE_TRANSFER_PENALTY_FACTOR)),

    # used in unit failover
    "UNIT_FAILOVER_TIMEOUT":
    lambda: os.getenv("UNIT_FAILOVER_TIMEOUT", str(UNIT_FAILOVER_TIMEOUT)),

    "RAY_TASK_RETRY_DELAY_MS":
    lambda: os.getenv("LLUMNIX_RAY_TASK_RETRY_DELAY_MS", str(RAY_TASK_RETRY_DELAY_MS)),

    "RAY_RPC_TIMEOUT":
    lambda: os.getenv("LLUMNIX_RAY_RPC_TIMEOUT", str(RAY_RPC_TIMEOUT)),

    # used in vLLM-V1
    "UTILITY_CALL_TIMEOUT":
    lambda: os.getenv("LLUMNIX_UTILITY_CALL_TIMEOUT", str(UTILITY_CALL_TIMEOUT)),
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
