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
import pytest
import ray

from vllm.engine.arg_utils import EngineArgs

from llumnix.arg_utils import ManagerArgs, InstanceArgs, EntrypointsArgs, LaunchArgs
from llumnix.entrypoints.setup import launch_ray_cluster
from llumnix.utils import get_ip_address
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.ray_utils import get_manager_name, execute_actor_method_sync_with_retries, execute_actor_method_async_with_retries
from llumnix.manager import Manager
from llumnix.scaler import Scaler

# pylint: disable=unused-import
from tests.conftest import ray_env


@pytest.fixture
def manager():
    engine_args = EngineArgs(model="facebook/opt-125m", download_dir="/mnt/model", worker_use_ray=True, enforce_eager=True)
    scaler: Scaler = Scaler.from_args(
        EntrypointsArgs(), ManagerArgs(), InstanceArgs(), engine_args, LaunchArgs())
    ray.get(scaler.is_ready.remote())
    manager: Manager = ray.get_actor(get_manager_name(), namespace='llumnix')
    ray.get(manager.is_ready.remote())
    yield manager

def test_launch_ray_cluster():
    ip_address = get_ip_address()
    os.environ['HEAD_NODE'] = '1'
    os.environ['HEAD_NODE_IP'] = ip_address
    result = launch_ray_cluster(6379)
    assert result.returncode == 0

def test_init_manager(ray_env, manager):
    assert manager is not None
    manager_actor_handle = ray.get_actor(get_manager_name(), namespace='llumnix')
    assert manager_actor_handle is not None
    assert manager == manager_actor_handle

def test_init_zmq(ray_env):
    ip = '127.0.0.1'
    request_output_queue = init_request_output_queue_server(ip, 'zmq')
    assert request_output_queue is not None

def test_retry_manager_method_sync(ray_env, manager):
    ret = execute_actor_method_sync_with_retries(manager.is_ready.remote, "Manager", 'is_ready')
    assert ret is True

@pytest.mark.asyncio
async def test_retry_manager_method_async(ray_env, manager):
    ret = await execute_actor_method_async_with_retries(manager.is_ready.remote, "Manager", 'is_ready')
    assert ret is True
