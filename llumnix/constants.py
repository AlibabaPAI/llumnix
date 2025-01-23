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

# llumnix/manager.py
CLEAR_REQUEST_INSTANCE_INTERVAL: float = 1000.0
NO_INSTANCE_RETRY_GENERATE_INTERVAL: float = 1.0
WAIT_ALL_MIGRATIONS_DONE_INTERVAL: float = 0.1
AUTO_SCALE_UP_INTERVAL: float = 1.0
WAIT_PLACEMENT_GROUP_TIMEOUT: float = 5.0
CHECK_DEPLOYMENT_STATES_INTERVAL: float = 30.0
WATCH_DEPLOYMENT_INTERVAL: float = 10.0
WATCH_DEPLOYMENT_INTERVAL_PENDING_INSTANCE: float = 120.0

# llumnix/entrypoints/setup.py
MAX_RAY_RESTART_TIMES: int = 10
RAY_RESTART_INTERVAL: float = 10.0

# llumnix/entrypoints/vllm/client.py, llumnix/entrypoints/bladellm/client.py
WAIT_MANAGER_INTERVAL: float = 1.0

# llumnix/entrypoints/vllm/api_server.py
SERVER_TIMEOUT_KEEP_ALIVE: float = 5.0

# llumnix/llumlet/llumlet.py
CHECK_ENGINE_STATE_INTERVAL: float = 1.0

# llumnix/backends/vllm/llm_engine.py
NO_OUTPUTS_STEP_INTERVAL: float = 0.01

# llumnix/queue/zmq_client.py
RPC_GET_DATA_TIMEOUT_MS: int = 5000

# llumnix/queue/zmq_server.py
RPC_SOCKET_LIMIT_CUTOFF: int = 2000
RPC_ZMQ_HWM: int = 0
RETRY_BIND_ADDRESS_INTERVAL: float = 10.0
MAX_BIND_ADDRESS_RETRY_TIMES: int = 10

# llumnix/entrypoints/utils.py
MAX_MANAGER_RETRY_TIMES: int = 10
RETRY_MANAGER_INTERVAL: float = 5.0
