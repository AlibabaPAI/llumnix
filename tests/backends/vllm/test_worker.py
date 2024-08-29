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

import pytest
import torch
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from vllm.engine.arg_utils import EngineArgs
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.cache_engine import CacheEngine
from vllm.config import EngineConfig
from vllm.executor.ray_gpu_executor import RayWorkerWrapper

from llumnix.arg_utils import EngineManagerArgs
from llumnix.utils import random_uuid


def create_worker(rank: int, local_rank: int, engine_config: EngineConfig,
                  worker_module_name="llumnix.backends.vllm.worker",
                  worker_class_name="MigrationWorker"):
    scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )

    worker = ray.remote(
        num_cpus=0,
        num_gpus=1,
        scheduling_strategy=scheduling_strategy
    )(RayWorkerWrapper).remote(
        worker_module_name=worker_module_name,
        worker_class_name=worker_class_name,
        trust_remote_code=True
    )

    worker.init_worker.remote(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
        cache_config=engine_config.cache_config,
        load_config=engine_config.load_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
        lora_config=engine_config.lora_config,
        vision_language_config=engine_config.vision_language_config,
        is_driver_worker = False
    )

    return worker

@pytest.mark.parametrize("backend", ['rpc', 'gloo', 'nccl'])
def test_reserve_memory_for_migration(backend):
    ray.init(namespace="llumnix", ignore_reinit_error=True)

    engine_config = EngineArgs(model='facebook/opt-125m', max_model_len=8, enforce_eager=True).create_engine_config()
    migraiton_config = EngineManagerArgs(migration_cache_blocks=1).create_migration_config()
    migraiton_config.migration_backend = backend
    worker = create_worker(rank=0, local_rank=0, engine_config=engine_config)
    ray.get(worker.execute_method.remote('init_device'))

    block_size = CacheEngine.get_cache_block_size(engine_config.cache_config, engine_config.model_config,
                                                  engine_config.parallel_config)
    num_layers = engine_config.model_config.get_num_layers(engine_config.parallel_config)
    occupy_memory = migraiton_config.migration_cache_blocks * block_size * migraiton_config.migration_num_layers // num_layers

    migration_cache_size = ray.get(worker.execute_method.remote('reserve_memory_for_migration',
                                                                migration_config=migraiton_config,
                                                                model_config=engine_config.model_config,
                                                                cache_config=engine_config.cache_config,
                                                                parallel_config=engine_config.parallel_config))
    assert migration_cache_size == occupy_memory

    ray.shutdown()

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU to run the test.")
@pytest.mark.parametrize("backend", ['rpc', 'gloo', 'nccl'])
def test_rebuild_migration_backend(backend):
    ray.init(namespace="llumnix", ignore_reinit_error=True)

    engine_config = EngineArgs(model='facebook/opt-125m', max_model_len=8, enforce_eager=True).create_engine_config()
    migraiton_config = EngineManagerArgs(migration_cache_blocks=1).create_migration_config()
    migraiton_config.migration_backend = backend

    worker0 = create_worker(rank=0, local_rank=0, engine_config=engine_config)
    worker0_id = random_uuid()
    ray.get(worker0.execute_method.remote('init_device'))
    ray.get(worker0.execute_method.remote('initialize_cache', num_gpu_blocks=8, num_cpu_blocks=0))
    ray.get(worker0.execute_method.remote(
        'init_migration',
        instance_id=worker0_id,
        migration_config=migraiton_config,
        src_worker_handle_list=[worker0],
        node_id=ray.get_runtime_context().get_node_id()))
    instance_rank = {worker0_id: 0}
    assert ray.get(worker0.execute_method.remote('rebuild_migration_backend', instance_rank=instance_rank,
                                                 group_name=random_uuid()))
    assert ray.get(worker0.execute_method.remote('warmup'))

    worker1 = create_worker(rank=0, local_rank=0, engine_config=engine_config)
    worker1_id = random_uuid()
    ray.get(worker1.execute_method.remote('init_device'))
    ray.get(worker1.execute_method.remote('initialize_cache', num_gpu_blocks=8, num_cpu_blocks=0))
    ray.get(worker1.execute_method.remote(
        'init_migration',
        instance_id=worker1_id,
        migration_config=migraiton_config,
        src_worker_handle_list=[worker1],
        node_id=ray.get_runtime_context().get_node_id()))

    instance_rank = {worker1_id: 1, worker0_id: 0}
    group_name = random_uuid()
    assert all(ray.get([worker0.execute_method.remote('rebuild_migration_backend',
                                                      instance_rank=instance_rank, group_name=group_name),
                        worker1.execute_method.remote('rebuild_migration_backend',
                                                      instance_rank=instance_rank, group_name=group_name)]))
    assert all(ray.get([worker0.execute_method.remote('warmup'),
                        worker1.execute_method.remote('warmup')]))

    ray.kill(worker1)

    instance_rank = {worker0_id: 0}
    assert ray.get(worker0.execute_method.remote('rebuild_migration_backend', instance_rank=instance_rank,
                                                group_name=random_uuid()))
    assert ray.get(worker0.execute_method.remote('warmup'))

    ray.shutdown()
