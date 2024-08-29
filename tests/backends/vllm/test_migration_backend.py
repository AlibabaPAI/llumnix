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

import random
import pytest
import torch
import ray

from vllm.engine.arg_utils import EngineArgs

from llumnix.backends.vllm.worker import MigrationWorker
from llumnix.arg_utils import EngineManagerArgs
from llumnix.utils import random_uuid

from tests.backends.vllm.test_worker import create_worker

class MockMigrationWorker(MigrationWorker):
    def set_gpu_cache(self, data):
        for layer_idx in range(self.cache_engine.num_layers):
            self.gpu_cache[layer_idx].copy_(data[layer_idx])
        torch.cuda.synchronize()

    def get_gpu_cache(self):
        torch.cuda.synchronize()
        return self.gpu_cache

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU to run the test.")
@pytest.mark.parametrize("backend", ['rpc', 'gloo', 'nccl'])
def test_migrate_cache(backend):
    ray.init(namespace="llumnix", ignore_reinit_error=True)

    engine_config = EngineArgs(model='facebook/opt-125m', max_model_len=8, enforce_eager=True).create_engine_config()
    migraiton_config = EngineManagerArgs(migration_cache_blocks=3, migration_num_layers=5).create_migration_config()
    migraiton_config.migration_backend = backend

    worker0 = create_worker(rank=0, local_rank=0, engine_config=engine_config,
                            worker_module_name="tests.backends.vllm.test_migration_backend",
                            worker_class_name="MockMigrationWorker")
    worker1 = create_worker(rank=0, local_rank=0, engine_config=engine_config,
                            worker_module_name="tests.backends.vllm.test_migration_backend",
                            worker_class_name="MockMigrationWorker")

    ray.get(worker0.execute_method.remote('init_device'))
    ray.get(worker1.execute_method.remote('init_device'))

    num_gpu_blocks = 8
    ray.get(worker0.execute_method.remote('initialize_cache', num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=0))
    ray.get(worker1.execute_method.remote('initialize_cache', num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=0))

    worker0_id = random_uuid()
    ray.get(worker0.execute_method.remote(
        'init_migration',
        instance_id=worker0_id,
        migration_config=migraiton_config,
        src_worker_handle_list=[worker0],
        node_id=ray.get_runtime_context().get_node_id()))

    worker1_id = random_uuid()
    ray.get(worker1.execute_method.remote(
        'init_migration',
        instance_id=worker1_id,
        migration_config=migraiton_config,
        src_worker_handle_list=[worker1],
        node_id=ray.get_runtime_context().get_node_id()))

    instance_rank = {worker0_id: 0, worker1_id: 1}
    group_name = random_uuid()
    assert all(ray.get([worker0.execute_method.remote('rebuild_migration_backend',
                                        instance_rank=instance_rank, group_name=group_name),
                        worker1.execute_method.remote('rebuild_migration_backend',
                                        instance_rank=instance_rank, group_name=group_name)]))
    assert all(ray.get([worker0.execute_method.remote('warmup'),
                        worker1.execute_method.remote('warmup')]))

    num_layers = engine_config.model_config.get_num_layers(engine_config.parallel_config)
    head_size = engine_config.model_config.get_head_size()
    num_heads = engine_config.model_config.get_num_kv_heads(engine_config.parallel_config)
    block_size = engine_config.cache_config.block_size

    dummy_data = torch.randn(size=(num_layers, 2, num_gpu_blocks, block_size*num_heads*head_size))
    ray.get(worker0.execute_method.remote('set_gpu_cache', data=dummy_data))
    worker0_data = ray.get(worker0.execute_method.remote('get_gpu_cache'))

    dst_blocks = list(range(num_gpu_blocks))
    random.shuffle(dst_blocks)
    src_to_dst = dict(enumerate(dst_blocks))
    ray.get(worker1.execute_method.remote(
        'migrate_cache',
        src_worker_handle_list=[worker0],
        src_blocks=list(src_to_dst.keys()),
        dst_blocks=list(src_to_dst.values())))

    worker1_data = ray.get(worker1.execute_method.remote('get_gpu_cache'))

    for layer_idx in range(num_layers):
        for src_idx, dst_idx in src_to_dst.items():
            assert torch.allclose(worker0_data[layer_idx][0][src_idx], worker1_data[layer_idx][0][dst_idx])
            assert torch.allclose(worker0_data[layer_idx][1][src_idx], worker1_data[layer_idx][1][dst_idx])

    ray.shutdown()
