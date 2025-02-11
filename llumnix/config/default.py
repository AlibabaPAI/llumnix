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

from .config import LlumnixConfig as LC

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = LC()

# -----------------------------------------------------------------------------
# SERVER CONFIGURATION
# -----------------------------------------------------------------------------
_C.SERVER = LC()
# Hostname for the server
_C.SERVER.HOST = "localhost"
# Port number for the server
_C.SERVER.PORT = 8000
# Path to SSL key file
_C.SERVER.SSL_KEYFILE = None
# Path to SSL certificate file
_C.SERVER.SSL_CERTFILE = None
# Log level for the server
_C.SERVER.LOG_LEVEL = "info"
# Queue type for request output queue
_C.SERVER.REQUEST_OUTPUT_QUEUE_TYPE = "rayqueue"
# Port number for the zmq request output queue
_C.SERVER.REQUEST_OUTPUT_QUEUE_PORT = 1234
# Disable logging requests in server
_C.SERVER.DISABLE_LOG_REQUESTS_SERVER = False
# Enable logging request timestamp
_C.SERVER.LOG_REQUEST_TIMESTAMPS = False
# Path to config file of arguments
_C.SERVER.CONFIG_FILE = None
# Disable keep serve process alive
_C.SERVER.DISABLE_KEEP_SERVE_PROCESS_ALIVE = False

# ----------------------------- RAY CONFIGURATION -----------------------------
# If True, launch Ray cluster in API server
_C.SERVER.LAUNCH_RAY_CLUSTER = False
# Port number for the Ray cluster
_C.SERVER.RAY_CLUSTER_PORT = 6379
# Disable redirecting all worker logs to driver
_C.SERVER.DISABLE_LOG_TO_DRIVER = False

# -----------------------------------------------------------------------------
# MANAGER CONFIGURATION
# -----------------------------------------------------------------------------
_C.MANAGER = LC()
# Number of instances created at initialization
_C.MANAGER.INITIAL_INSTANCES = 1
# Time interval(s) to update instance info and pair migration
_C.MANAGER.POLLING_INTERVAL = 0.05
# Disable logging requests in manager
_C.MANAGER.DISABLE_LOG_REQUESTS_MANAGER = False
# Enable logging instance info
_C.MANAGER.LOG_INSTANCE_INFO = False
# Log filename
_C.MANAGER.LOG_FILENAME = "server.log"
# Enable port increment when deploying multiple servers
_C.MANAGER.ENABLE_PORT_INCREMENT = False
# Enable store port offset when deploying multiple servers
_C.MANAGER.ENABLE_PORT_OFFSET_STORE = False
# Enable prefill decoding disaggregation
_C.MANAGER.ENABLE_PD_DISAGG = False
# The p:d ratio used in gloabl launch model
_C.MANAGER.PD_RATIO = "1:1"

# -------------------------- DISPATCH CONFIGURATION ---------------------------
# Request dispatch policy
_C.MANAGER.DISPATCH_POLICY = 'load'
# Number of candidate instances for dispatch policy
_C.MANAGER.POWER_OF_K_CHOICE = 1

# -------------------------- MIGRATION CONFIGURATION --------------------------
# Enable migrate requests between instances
_C.MANAGER.ENABLE_MIGRATION = False
# Pair migration frequency
_C.MANAGER.PAIR_MIGRATION_FREQUENCY = 1
# Pair migration policy
_C.MANAGER.PAIR_MIGRATION_POLICY = 'defrag'
# Migrate out instance load threshold
_C.MANAGER.MIGRATE_OUT_THRESHOLD = -3.0

# --------------------------- SCALING CONFIGURATION ---------------------------
# Enable auto scaling
_C.MANAGER.ENABLE_SCALING = False
# Instance scaling load metric
_C.MANAGER.SCALING_LOAD_METRIC = 'remaining_steps'
# Minimum number of instances
_C.MANAGER.MIN_INSTANCES = -1
# Maximum number of instances
_C.MANAGER.MAX_INSTANCES = -1
# Interval time to check scaling
_C.MANAGER.SCALING_INTERVAL = 10
# Scaling policy
_C.MANAGER.SCALING_POLICY = 'avg_load'
# Scale up threshold
_C.MANAGER.SCALE_UP_THRESHOLD = -10
# Scale down threshold
_C.MANAGER.SCALE_DOWN_THRESHOLD = -60

# -----------------------------------------------------------------------------
# INSTANCE CONFIGURATION
# -----------------------------------------------------------------------------
_C.INSTANCE = LC()
# Engine types: prefill, decode, no_constraints
_C.INSTANCE.INSTANCE_TYPE = "no_constraints"
# Enable simulator mode
_C.INSTANCE.SIMULATOR_MODE = False
# Profiling result file path when using simulator
_C.INSTANCE.PROFILING_RESULT_FILE_PATH = None

# ------------------------- LOAD METRICS CONFIGURATION ------------------------
# Instance dispatch load metric
_C.INSTANCE.DISPATCH_LOAD_METRIC = 'remaining_steps'
# Instance migration load metric
_C.INSTANCE.MIGRATION_LOAD_METRIC = 'remaining_steps'

# -------------------------- MIGRATION CONFIGURATION --------------------------
# Enable defragmentation through migration based on virtual usage
_C.INSTANCE.ENABLE_DEFRAG = False
# Request migration policy
_C.INSTANCE.REQUEST_MIGRATION_POLICY = 'SR'
# Drop migration if the number of stages > migration_max_stages
_C.INSTANCE.MIGRATION_MAX_STAGES = 3
# If the number of remain blocks < migration_last_stage_max_blocks, do last stage migration
_C.INSTANCE.MIGRATION_LAST_STAGE_MAX_BLOCKS = 16
# Communication backend of migration
_C.INSTANCE.MIGRATION_BACKEND = "gloo"
# Number of cache blocks in migration
_C.INSTANCE.MIGRATION_BUFFER_BLOCKS = 512
# Number of kv-cache layers to transfer in each round during migration
_C.INSTANCE.MIGRATION_NUM_LAYERS = 1
# Timeout(s) for initializing migration backend
_C.INSTANCE.MIGRATION_BACKEND_INIT_TIMEOUT = 10.0
# Transfer type for migration backend kvTransfer
_C.INSTANCE.MIGRATION_BACKEND_TRANSFER_TYPE = "rdma"
# Address of grpc server for migration backend
_C.INSTANCE.GRPC_MIGRATION_BACKEND_SERVER_ADDRESS = "127.0.0.1:50051"
# URL of naming server for kvtransfer migration backend
_C.INSTANCE.KVTRANSFER_MIGRATION_BACKEND_NAMING_URL = "file:/tmp/llumnix/naming/"
