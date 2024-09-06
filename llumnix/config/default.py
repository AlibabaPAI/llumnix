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

from .llumnix_config import LlumnixConfig as LC

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = LC()

# -----------------------------------------------------------------------------
# API SERVER CONFIGURATION
# -----------------------------------------------------------------------------
_C.SERVER = LC()
# Hostname for the server
_C.SERVER.HOST = "localhost"
# Port number for the server
_C.SERVER.PORT = 8000
# Path to SSL key file for secure connections
_C.SERVER.SSL_KEYFILE = None
# Path to SSL certificate file for secure connections
_C.SERVER.SSL_CERTFILE = None
# Disable logging requests in server
_C.SERVER.DISABLE_LOG_REQUESTS_SERVER = False

# -----------------------------------------------------------------------------
# RAY CONFIGURATION
# -----------------------------------------------------------------------------
_C.RAY = LC()
# Port number for the Ray cluster
_C.RAY.RAY_CLUSTER_PORT = 30050
# If True, launch Ray cluster in API server
_C.RAY.LAUNCH_RAY_CLUSTER = False

# -----------------------------------------------------------------------------
# MANAGER CONFIGURATION
# -----------------------------------------------------------------------------
_C.MANAGER = LC()
# Disable logging requests in manager
_C.MANAGER.DISABLE_LOG_REQUESTS_MANAGER = False
# Enable logging instance info
_C.MANAGER.LOG_INSTANCE_INFO = False
# Log filename
_C.MANAGER.LOG_FILENAME = "server.log"
# Profiling result file path
_C.MANAGER.PROFILING_RESULT_FILE_PATH = ""
# GPU type specified when using simulator
_C.MANAGER.GPU_TYPE = "a10"

# Disable fixing the placement of instance to current node
_C.MANAGER.DISABLE_FIXED_NODE_INIT_INSTANCE = False
# Disable initializing instance by manager
_C.MANAGER.DISABLE_INIT_INSTANCE_BY_MANAGER = False
# Number of instances created at initialization
_C.MANAGER.INITIAL_INSTANCES = 1
# Time interval(s) to update instance info and pair migration
_C.MANAGER.POLLING_INTERVAL = 0.05

# -----------------------------------------------------------------------------
# DISPATCH CONFIGURATION
# -----------------------------------------------------------------------------
# Instance load metric with choices: 'remaining_steps', 'usage_ratio'
_C.MANAGER.LOAD_METRIC = 'remaining_steps'
# Request dispatch policy with choices: 'balanced', 'load', 'queue', 'flood'
_C.MANAGER.DISPATCH_POLICY = 'load'

# -----------------------------------------------------------------------------
# MIGRATION CONFIGURATION
# -----------------------------------------------------------------------------
# Enable migrate requests between instances
_C.MANAGER.ENABLE_MIGRATION = False
# Pair migration frequency
_C.MANAGER.PAIR_MIGRATION_FREQUENCY = 1
# Pair migration policy with choices: 'balanced', 'defrag_constrained', 'defrag_relaxed'
_C.MANAGER.PAIR_MIGRATION_POLICY = 'defrag_constrained'
# Migrate out instance load threshold
_C.MANAGER.MIGRATE_OUT_THRESHOLD = 3.0
# Request migration policy with choices: 'LCFS', 'SJF', 'LJF'
_C.MANAGER.REQUEST_MIGRATION_POLICY = 'SJF'
# Enable defragmentation through migration based on virtual usage
_C.MANAGER.ENABLE_DEFRAG = True
# Drop migration if the number of stages > max_stages
_C.MANAGER.MAX_STAGES = 3
# If the number of remain blocks < last_stage_max_blocks, do last stage migration
_C.MANAGER.LAST_STAGE_MAX_BLOCKS = 16

# Communication backend of migration with choices: 'gloo', 'nccl', 'rpc'
_C.MANAGER.MIGRATION_BACKEND = "rpc"
# Timeout(s) for initializing migration backend
_C.MANAGER.MIGRATION_BACKEND_INIT_TIMEOUT = 10.0
# Number of cache blocks in migration
_C.MANAGER.MIGRATION_CACHE_BLOCKS = 512
# Number of kv-cache layers to transfer in each round during migration
_C.MANAGER.MIGRATION_NUM_LAYERS = 1

# -----------------------------------------------------------------------------
# SCALING CONFIGURATION
# -----------------------------------------------------------------------------
# Enable scaling instances based on load
_C.MANAGER.ENABLE_SCALING = False
# Minimum number of instances
_C.MANAGER.MIN_INSTANCES = 1
# Maximum number of instances
_C.MANAGER.MAX_INSTANCES = 1
# Interval time to check scaling
_C.MANAGER.SCALING_INTERVAL = 10
# Scaling policy with choices: 'max_load', 'avg_load'
_C.MANAGER.SCALING_POLICY = 'avg_load'
# Scale up threshold
_C.MANAGER.SCALE_UP_THRESHOLD = 10
# Scale down threshold
_C.MANAGER.SCALE_DOWN_THRESHOLD = 60
