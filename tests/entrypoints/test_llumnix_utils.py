import os
import pytest
import ray

from llumnix.arg_utils import EngineManagerArgs
from llumnix.entrypoints.llumnix_utils import (get_ip_address,
                                               launch_ray_cluster,
                                               init_manager,
                                               init_request_output_queue,
                                               retry_manager_method_sync,
                                               retry_manager_method_async)
from llumnix.llm_engine_manager import MANAGER_ACTOR_NAME


def test_launch_ray_cluster():
    ip_address = get_ip_address()
    os.environ['HEAD_NODE'] = '1'
    os.environ['HEAD_NODE_IP'] = ip_address
    result = launch_ray_cluster(30050)
    assert result.returncode == 0

def test_init_manager():
    engine_manager_args = EngineManagerArgs()
    engine_manager = init_manager(engine_manager_args)
    assert engine_manager is not None
    engine_manager_actor_handle = ray.get_actor(MANAGER_ACTOR_NAME, namespace='llumnix')
    assert engine_manager_actor_handle is not None
    assert engine_manager == engine_manager_actor_handle
    ray.kill(engine_manager)
    ray.shutdown()

def test_init_request_output_queue():
    request_output_queue = init_request_output_queue()
    assert request_output_queue is not None
    ray.shutdown()

def test_retry_manager_method_sync():
    engine_manager_args = EngineManagerArgs()
    engine_manager = init_manager(engine_manager_args)
    ret = retry_manager_method_sync(engine_manager.is_ready.remote, 'is_ready')
    assert ret is True
    ray.kill(engine_manager)
    ray.shutdown()

@pytest.mark.asyncio
async def test_retry_manager_method_async():
    engine_manager_args = EngineManagerArgs()
    engine_manager = init_manager(engine_manager_args)
    ret = await retry_manager_method_async(engine_manager.is_ready.remote, 'is_ready')
    assert ret is True
    ray.kill(engine_manager)
    ray.shutdown()
