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

import torch


RAY_RPC_TIMEOUT: float = 10.0

# llumnix/llumlet.py, llumnix/backends/vllm/executor.py
# Llumlet and Worker share the same 1 gpu in the first bundle of PlacementGroup.
NUM_GPUS_VLLM_GPU_ACTOR = 0.5
# FIXME(zhaozhiyu): copy from NUM_GPUS_VLLM_GPU_ACTOR, modify this later
NUM_GPUS_VLLM_V1_GPU_ACTOR = 0.5
# llumnix/llumlet.py, llumnix/scaler.py, llumnix/backends/bladellm/worker.py
# Llumlet, ActorOutputMediator, APIServerActor and Worker share the same 1 gpu in the first bundle of PlacementGroup.
NUM_GPUS_BLADELLM_GPU_ACTOR = 0.25

# examples/offline_inference.py, tests/*
MODEL_PATH: str = '/mnt/model'
DATASET_PATH: str = '/mnt/dataset'

# llumnix/manager.py
NO_INSTANCE_RETRY_GENERATE_INTERVAL: float = 1.0
WAIT_ALL_MIGRATIONS_DONE_INTERVAL: float = 0.1
AUTO_SCALE_UP_INTERVAL: float = 1.0
WAIT_PLACEMENT_GROUP_TIMEOUT: float = 5.0
CHECK_DEPLOYMENT_STATES_INTERVAL: float = 30.0
WATCH_DEPLOYMENT_INTERVAL: float = 10.0
INSTANCE_READY_TIMEOUT: float = 300.0
SERVER_READY_TIMEOUT: float = 60.0

# llumnix/global_scheduler/dispatch_scheduler.py
DISPATCH_LOG_FREQUENCY: int = 100

# llumnix/entrypoints/setup.py
MAX_RAY_RESTART_TIMES: int = 10
RAY_RESTART_INTERVAL: float = 10.0
SUBPROCESS_RUN_TIMEOUT: float = 60.0

# llumnix/entrypoints/vllm/client.py, llumnix/entrypoints/bladellm/client.py
WAIT_MANAGER_INTERVAL: float = 1.0
INIT_GLOBAL_INSTANCES_INTERVAL: float = 600.0
UPDATE_GLOBAL_INSTANCES_INTERVAL: float = 1800.0

# llumnix/entrypoints/vllm/api_server.py
SERVER_TIMEOUT_KEEP_ALIVE: float = 5.0

# llumnix/llumlet/llumlet.py
CHECK_ENGINE_STATE_INTERVAL: float = 1.0

# llumnix/backends/vllm/llm_engine.py
NO_OUTPUTS_STEP_INTERVAL: float = 0.01

# llumnix/queue/ray_queue_client.py, llumnix/queue/ray_queue_server.py
RAY_QUEUE_RPC_TIMEOUT: int = 5000

# llumnix/queue/zmq_client.py, llumnix/queue/zmq_server.py
ZMQ_RPC_TIMEOUT: int = 5000

# llumnix/queue/zmq_server.py
RPC_SOCKET_LIMIT_CUTOFF: int = 2000
RPC_ZMQ_HWM: int = 0
RETRY_BIND_ADDRESS_INTERVAL: float = 10.0
MAX_BIND_ADDRESS_RETRY_TIMES: int = 10
ZMQ_IO_THREADS: int = 8
SERVER_QUEUE_TIMEOUT: float = 10.0

# llumnix/entrypoints/utils.py
MAX_ACTOR_METHOD_RETRIES: int = 10
RAY_TASK_RETRY_DELAY_MS: int = 1000
MAX_TASK_RETRIES: int = 10
RETRIES_INTERVAL: float = 5.0

# llumnix/backends/*/migration_backend.py, llumnix/backends/*/migration_worker.py
GRPC_MAX_MESSAGE_LENGTH = 1 << 31 - 1
NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION = [torch.float32, torch.float16]
GRPC_TIMEOUT = 10.0
KVTRANSFER_MIGRATION_TIMEOUT = 100.0
GRPC_MIGRATION_TIMEOUT = 100.0
RAYRPC_MIGRATION_TIMEOUT = 100.0

# llumnix/entrypoints/api_server_actor.py
SERVER_GRACEFUL_SHUTDOWN_TIMEOUT: float = 10.0
SERVER_START_TIMEOUT: float = 10.0
SERVER_STOP_TIMEOUT: float = 10.0

# llumnix/llumlet/migration_coordinator.py
PENDING_MIGRATE_IN_TIMEOUT = 10.0

# llumnix/metrics
DEFAULT_METRICS_EXPORT_INTERVAL_SEC: int = 15
EAS_FAILURE_LIMIT: int = 5
DEFAULT_EAS_EXPORTER_URL = "http://localhost:8080/api/builtin/realtime_metrics"

# llumnix/load_computation.py
BLOCKDEMANDFACTOR_BUSY_THRESHOLD: float = 1.0
REMAININGSTEPS_BUSY_THRESHOLD: float = 10.0
DECODE_COMPUTE_BOUND_BATCH_SIZE: float = 128
