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

import time

import pytest
import torch
import ray

from vllm.engine.arg_utils import EngineArgs
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.cache_engine import CacheEngine
from vllm.config import EngineConfig
from vllm.executor.ray_gpu_executor import RayWorkerWrapper

from llumnix.arg_utils import InstanceArgs
from llumnix.utils import random_uuid
from llumnix.ray_utils import initialize_placement_group, get_placement_group_name
from llumnix.backends.vllm.worker import MigrationWorker

# pylint: disable=unused-import
from tests.conftest import ray_env


class MockMigrationWorker(MigrationWorker):
    def __init__(self, *args, **kwargs):
        self.do_a_started = False
        self.do_a_finished = False
        super().__init__(*args, **kwargs)

    def do_a(self):
        self.do_a_started = True
        self.do_a_finished = False
        time.sleep(3.0)
        self.do_a_finished = True

    def do_b(self):
        return self.do_a_started, self.do_a_finished


def create_worker(rank: int, local_rank: int, engine_config: EngineConfig,
                  worker_module_name="llumnix.backends.vllm.worker",
                  worker_class_name="MigrationWorker",
                  max_concurrency=1):
    worker = ray.remote(
        num_cpus=0,
        num_gpus=1,
        max_concurrency=max_concurrency,
        name=f"unit_test_worker_{random_uuid()}"
    )(RayWorkerWrapper).remote(
        worker_module_name=worker_module_name,
        worker_class_name=worker_class_name,
        trust_remote_code=True
    )

    ray.get(worker.init_worker.remote(
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
        is_driver_worker = False
    ))

    return worker

@pytest.mark.parametrize("backend", ['rayrpc', 'gloo', 'nccl'])
def test_reserve_memory_for_migration(ray_env, backend):
    engine_config = EngineArgs(model='facebook/opt-125m', download_dir="/mnt/model", max_model_len=8, enforce_eager=True).create_engine_config()
    migration_config = InstanceArgs(migration_buffer_blocks=1).create_migration_config()
    migration_config.migration_backend = backend
    worker = create_worker(rank=0, local_rank=0, engine_config=engine_config)
    ray.get(worker.execute_method.remote('init_device'))

    block_size = CacheEngine.get_cache_block_size(engine_config.cache_config, engine_config.model_config,
                                                  engine_config.parallel_config)
    num_layers = engine_config.model_config.get_num_layers(engine_config.parallel_config)
    occupy_memory = migration_config.migration_buffer_blocks * block_size * migration_config.migration_num_layers // num_layers

    migration_cache_size = ray.get(worker.execute_method.remote('reserve_memory_for_migration',
                                                                migration_config=migration_config,
                                                                model_config=engine_config.model_config,
                                                                cache_config=engine_config.cache_config,
                                                                parallel_config=engine_config.parallel_config))
    assert migration_cache_size == occupy_memory

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU to run the test.")
@pytest.mark.parametrize("backend", ['rayrpc', 'gloo', 'nccl'])
def test_rebuild_migration_backend(ray_env, backend):
    engine_config = EngineArgs(model='facebook/opt-125m', download_dir="/mnt/model", max_model_len=8, enforce_eager=True).create_engine_config()
    migration_config = InstanceArgs(migration_buffer_blocks=1).create_migration_config()
    migration_config.migration_backend = backend

    worker0 = create_worker(rank=0, local_rank=0, engine_config=engine_config)
    worker0_id = random_uuid()
    placement_group0 = initialize_placement_group(get_placement_group_name(worker0_id), num_cpus=1, num_gpus=1, detached=True)
    ray.get(worker0.execute_method.remote('init_device'))
    ray.get(worker0.execute_method.remote('initialize_cache', num_gpu_blocks=8, num_cpu_blocks=0))
    ray.get(worker0.execute_method.remote(
        'init_migration',
        instance_id=worker0_id,
        migration_config=migration_config,
        src_worker_handle_list=[worker0],
        placement_group=placement_group0))
    instance_rank = {worker0_id: 0}
    assert ray.get(worker0.execute_method.remote('rebuild_migration_backend', instance_rank=instance_rank,
                                                 group_name=random_uuid()))
    assert ray.get(worker0.execute_method.remote('warmup'))

    worker1 = create_worker(rank=0, local_rank=0, engine_config=engine_config)
    worker1_id = random_uuid()
    placement_group1 = initialize_placement_group(get_placement_group_name(worker1_id), num_cpus=1, num_gpus=1, detached=True)
    ray.get(worker1.execute_method.remote('init_device'))
    ray.get(worker1.execute_method.remote('initialize_cache', num_gpu_blocks=8, num_cpu_blocks=0))
    ray.get(worker1.execute_method.remote(
        'init_migration',
        instance_id=worker1_id,
        migration_config=migration_config,
        src_worker_handle_list=[worker1],
        placement_group=placement_group1))

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

def test_max_concurrency(ray_env):
    engine_config = EngineArgs(model='facebook/opt-125m', download_dir="/mnt/model", max_model_len=8, enforce_eager=True).create_engine_config()
    worker_no_concurrency = create_worker(rank=0, local_rank=0, engine_config=engine_config,
                                          worker_module_name="tests.unit_test.backends.vllm.test_worker",
                                          worker_class_name="MockMigrationWorker",
                                          max_concurrency=1)

    worker_no_concurrency.execute_method.remote('do_a')
    do_a_started, do_a_finished = ray.get(worker_no_concurrency.execute_method.remote('do_b'))
    assert do_a_started and do_a_finished

    worker_with_concurrency = create_worker(rank=0, local_rank=0, engine_config=engine_config,
                                            worker_module_name="tests.unit_test.backends.vllm.test_worker",
                                            worker_class_name="MockMigrationWorker",
                                            max_concurrency=2)

    worker_with_concurrency.execute_method.remote('do_a')
    time.sleep(1.0)
    do_a_started, do_a_finished = ray.get(worker_with_concurrency.execute_method.remote('do_b'))
    assert do_a_started and not do_a_finished
