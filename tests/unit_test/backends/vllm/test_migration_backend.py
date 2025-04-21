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
import os

import pytest
import torch
import ray

from vllm.engine.arg_utils import EngineArgs

from llumnix.backends.vllm.worker import MigrationWorker
from llumnix.arg_utils import InstanceArgs
from llumnix.utils import random_uuid, try_convert_to_local_path
from llumnix.ray_utils import initialize_placement_group, get_placement_group_name

# pylint: disable=unused-import
from tests.conftest import ray_env
from .test_worker import create_worker


class MockMigrationWorker(MigrationWorker):
    def __init__(self, *args, **kwargs):
        # Set 'VLLM_USE_RAY_SPMD_WORKER' to True and therefore, once is_last_stage of migrate_cache is True,
        # send_worker_metadata of do_send will be True.
        os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
        super().__init__(*args, **kwargs)

    def set_gpu_cache(self, data):
        for layer_idx in range(self.cache_engine[0].num_attention_layers):
            self.gpu_cache[0][layer_idx].copy_(data[layer_idx])
        torch.cuda.synchronize()

    def get_gpu_cache(self):
        torch.cuda.synchronize()
        return self.gpu_cache[0]

    def set_worker_metadata(self, request_id):
        self._seq_group_metadata_cache[request_id] = "worker_metadata"

    def get_worker_metadata(self, request_id):
        return self._seq_group_metadata_cache[request_id]

    def get_worker_stage_metadata(self, request_id):
        return self.migrating_in_seq_group_metadata[request_id]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU to run the test.")
@pytest.mark.parametrize("backend", ['rayrpc', 'gloo', 'nccl'])
@pytest.mark.parametrize("send_worker_metadata", [True, False])
def test_migrate_cache(ray_env, backend, send_worker_metadata):
    engine_config = EngineArgs(model=try_convert_to_local_path("facebook/opt-125m"), download_dir="/mnt/model",
                               max_model_len=8, enforce_eager=True).create_engine_config()
    migraiton_config = InstanceArgs(migration_buffer_blocks=3, migration_num_layers=5).create_migration_config()
    migraiton_config.migration_backend = backend

    worker0 = create_worker(rank=0, local_rank=0, engine_config=engine_config,
                            worker_module_name="tests.unit_test.backends.vllm.test_migration_backend",
                            worker_class_name="MockMigrationWorker")
    worker1 = create_worker(rank=0, local_rank=0, engine_config=engine_config,
                            worker_module_name="tests.unit_test.backends.vllm.test_migration_backend",
                            worker_class_name="MockMigrationWorker")

    ray.get(worker0.execute_method.remote('init_device'))
    ray.get(worker1.execute_method.remote('init_device'))

    num_gpu_blocks = 8
    ray.get(worker0.execute_method.remote('initialize_cache', num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=0))
    ray.get(worker1.execute_method.remote('initialize_cache', num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=0))

    worker0_id = random_uuid()
    placement_group0 = initialize_placement_group(get_placement_group_name(worker0_id), num_cpus=1, num_gpus=1, detached=True)
    ray.get(worker0.execute_method.remote(
        'init_migration',
        instance_id=worker0_id,
        migration_config=migraiton_config,
        src_worker_handle_list=[worker0],
        placement_group=placement_group0))

    worker1_id = random_uuid()
    placement_group1 = initialize_placement_group(get_placement_group_name(worker1_id), num_cpus=1, num_gpus=1, detached=True)
    ray.get(worker1.execute_method.remote(
        'init_migration',
        instance_id=worker1_id,
        migration_config=migraiton_config,
        src_worker_handle_list=[worker1],
        placement_group=placement_group1))

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

    dummy_data = torch.randn(size=(num_layers, 2, num_gpu_blocks, block_size, num_heads, head_size))
    ray.get(worker0.execute_method.remote('set_gpu_cache', data=dummy_data))
    worker0_cache = ray.get(worker0.execute_method.remote('get_gpu_cache'))

    request_id = random_uuid()
    if send_worker_metadata:
        ray.get(worker0.execute_method.remote('set_worker_metadata', request_id=request_id))
        worker0_data = ray.get(worker0.execute_method.remote('get_worker_metadata', request_id=request_id))

    dst_blocks = list(range(num_gpu_blocks))
    random.shuffle(dst_blocks)
    src_to_dst = dict(enumerate(dst_blocks))
    ray.get(worker1.execute_method.remote(
        'migrate_cache',
        src_worker_handle_list=[worker0],
        src_blocks=list(src_to_dst.keys()),
        dst_blocks=list(src_to_dst.values()),
        request_id=request_id,
        is_last_stage=send_worker_metadata))

    worker1_cache = ray.get(worker1.execute_method.remote('get_gpu_cache'))

    for layer_idx in range(num_layers):
        for src_idx, dst_idx in src_to_dst.items():
            assert torch.allclose(worker0_cache[layer_idx][0][src_idx], worker1_cache[layer_idx][0][dst_idx])
            assert torch.allclose(worker0_cache[layer_idx][1][src_idx], worker1_cache[layer_idx][1][dst_idx])

    if send_worker_metadata:
        worker1_data = ray.get(worker1.execute_method.remote('get_worker_stage_metadata', request_id=request_id))
        assert worker0_data == worker1_data
