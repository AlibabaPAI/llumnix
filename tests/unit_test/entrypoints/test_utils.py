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

from llumnix.arg_utils import EngineManagerArgs
from llumnix.entrypoints.setup import (get_ip_address,
                                       launch_ray_cluster,
                                       init_manager,
                                       retry_manager_method_sync,
                                       retry_manager_method_async)
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.utils import MANAGER_NAME

# pylint: disable=unused-import
from tests.conftest import ray_env


def test_launch_ray_cluster():
    ip_address = get_ip_address()
    os.environ['HEAD_NODE'] = '1'
    os.environ['HEAD_NODE_IP'] = ip_address
    result = launch_ray_cluster(6379)
    assert result.returncode == 0

def test_init_manager(ray_env):
    engine_manager_args = EngineManagerArgs()
    manager = init_manager(engine_manager_args)
    assert manager is not None
    manager_actor_handle = ray.get_actor(MANAGER_NAME, namespace='llumnix')
    assert manager_actor_handle is not None
    assert manager == manager_actor_handle

def test_init_zmq(ray_env):
    ip = '127.0.0.1'
    port = 1234
    request_output_queue = init_request_output_queue_server(ip, port, 'zmq')
    assert request_output_queue is not None

def test_retry_manager_method_sync(ray_env):
    engine_manager_args = EngineManagerArgs()
    manager = init_manager(engine_manager_args)
    ret = retry_manager_method_sync(manager.is_ready.remote, 'is_ready')
    assert ret is True

@pytest.mark.asyncio
async def test_retry_manager_method_async(ray_env):
    engine_manager_args = EngineManagerArgs()
    manager = init_manager(engine_manager_args)
    ret = await retry_manager_method_async(manager.is_ready.remote, 'is_ready')
    assert ret is True
