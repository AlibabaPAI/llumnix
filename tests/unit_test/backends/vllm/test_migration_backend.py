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
from llumnix.arg_utils import InstanceArgs
from llumnix.utils import random_uuid, initialize_placement_group, get_placement_group_name

# pylint: disable=unused-import
from tests.conftest import ray_env
from .test_worker import create_worker

def get_ready_workers(num_workers, num_gpu_blocks, engine_config, migraiton_config):
    workers = []
    worker_ids = []
    for _ in range(num_workers):
        worker_id = random_uuid()
        worker = create_worker(rank=0, local_rank=0, engine_config=engine_config,
                               worker_module_name="tests.unit_test.backends.vllm.test_migration_backend",
                               worker_class_name="MockMigrationWorker",
                               max_concurrency=8)
        ray.get(worker.execute_method.remote('initialize_cache', num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=0))
        placement_group = initialize_placement_group(get_placement_group_name(worker_id), num_cpus=1, num_gpus=0, detached=True)
        ray.get(worker.execute_method.remote(
            'init_migration',
            instance_id=worker_id,
            migration_config=migraiton_config,
            src_worker_handle_list=[worker],
            placement_group=placement_group))
        workers.append(worker)
        worker_ids.append(worker_id)

    instance_rank = {}
    for idx, worker_id in enumerate(worker_ids):
        instance_rank[worker_id] = idx
    group_name = random_uuid()
    init_group_tasks =[]
    for worker in workers:
        init_group_tasks.append(worker.execute_method.remote('rebuild_migration_backend',
            instance_rank=instance_rank, group_name=group_name))
    assert all(ray.get(init_group_tasks))

    warmup_tasks = []
    for worker in workers:
        warmup_tasks.append(worker.execute_method.remote('warmup'))
    assert all(ray.get(warmup_tasks))

    return workers, worker_ids


class MockMigrationWorker(MigrationWorker):
    def set_gpu_cache(self, data):
        for layer_idx in range(self.cache_engine[0].num_attention_layers):
            self.gpu_cache[0][layer_idx].copy_(data[layer_idx])
        torch.cuda.synchronize()

    def get_gpu_cache(self):
        torch.cuda.synchronize()
        gpu_data = []
        for layer_idx in range(self.cache_engine[0].num_attention_layers):
            gpu_data.append(self.gpu_cache[0][layer_idx].clone().cpu())
        return gpu_data


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Need at least 4 GPUs to run the test.")
@pytest.mark.parametrize("backend", ['rayrpc', 'gloo'])
def test_one_to_many_migrate_cache(ray_env, backend, migration_num_buffers):
    engine_config = EngineArgs(model='facebook/opt-125m', max_model_len=8, enforce_eager=True).create_engine_config()
    migraiton_config = InstanceArgs(migration_buffer_blocks=3, migration_num_layers=5,
                                    migration_num_buffers=3).create_migration_config()
    migraiton_config.migration_backend = backend

    num_workers = 4
    num_gpu_blocks = 6000
    workers, _ = get_ready_workers(num_workers, num_gpu_blocks, engine_config, migraiton_config)

    num_layers = engine_config.model_config.get_num_layers(engine_config.parallel_config)
    head_size = engine_config.model_config.get_head_size()
    num_heads = engine_config.model_config.get_num_kv_heads(engine_config.parallel_config)
    block_size = engine_config.cache_config.block_size

    dummy_data = torch.randn(size=(num_layers, 2, num_gpu_blocks, block_size, num_heads, head_size))
    ray.get(workers[0].execute_method.remote('set_gpu_cache', data=dummy_data))
    worker0_data = ray.get(workers[0].execute_method.remote('get_gpu_cache'))

    dst_blocks = list(range(num_gpu_blocks))
    random.shuffle(dst_blocks)

    single_worker_num_blocks = len(dst_blocks)//(num_workers-1)
    migration_tasks = []
    worker_idx = 1
    per_step_blocks = 500
    for offset in range(0, len(dst_blocks), single_worker_num_blocks):
        src_to_dst = dict(enumerate(dst_blocks[offset:offset+single_worker_num_blocks]))
        src_blocks = list(src_to_dst.keys())
        dst_blocks = list(src_to_dst.values())
        for idx in range(0, len(src_blocks), per_step_blocks):
            cur_src_blocks = src_blocks[idx:idx+per_step_blocks]
            cur_dst_blocks = dst_blocks[idx:idx+per_step_blocks]
            migration_tasks.append(workers[worker_idx].execute_method.remote(
                'migrate_cache',
                src_worker_handle_list=[workers[0]],
                src_blocks=cur_src_blocks,
                dst_blocks=cur_dst_blocks)
            )
        worker_idx += 1
    ray.get(migration_tasks)

    worker_idx = 1
    for offset in range(0, len(dst_blocks), single_worker_num_blocks):
        src_to_dst = dict(enumerate(dst_blocks[offset:offset+single_worker_num_blocks]))
        dst_worker_data = ray.get(workers[worker_idx].execute_method.remote('get_gpu_cache'))
        for layer_idx in range(num_layers):
            for src_idx, dst_idx in src_to_dst.items():
                assert torch.allclose(worker0_data[layer_idx][0][src_idx], dst_worker_data[layer_idx][0][dst_idx])
                assert torch.allclose(worker0_data[layer_idx][1][src_idx], dst_worker_data[layer_idx][1][dst_idx])
        worker_idx += 1

@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Need at least 4 GPU to run the test.")
@pytest.mark.parametrize("backend", ['rayrpc'])
def test_many_to_one_migrate_cache(ray_env, backend):
    engine_config = EngineArgs(model='facebook/opt-125m', max_model_len=8, enforce_eager=True).create_engine_config()
    migraiton_config = InstanceArgs(migration_buffer_blocks=3, migration_num_layers=5,
                                    migration_num_buffers=3).create_migration_config()
    migraiton_config.migration_backend = backend

    num_workers = 4
    num_gpu_blocks = 1000
    workers, _ = get_ready_workers(num_workers, num_gpu_blocks, engine_config, migraiton_config)

    num_layers = engine_config.model_config.get_num_layers(engine_config.parallel_config)
    head_size = engine_config.model_config.get_head_size()
    num_heads = engine_config.model_config.get_num_kv_heads(engine_config.parallel_config)
    block_size = engine_config.cache_config.block_size
    dummy_data = torch.randn(size=(num_layers, 2, num_gpu_blocks, block_size, num_heads, head_size))

    worker_datas = [0]
    for idx in range(1, num_workers):
        ray.get(workers[idx].execute_method.remote('set_gpu_cache', data=dummy_data))
        worker_datas.append(ray.get(workers[idx].execute_method.remote('get_gpu_cache')))

    dst_blocks = list(range(num_gpu_blocks))
    random.shuffle(dst_blocks)

    single_worker_num_blocks = len(dst_blocks)//(num_workers-1)
    migration_tasks = []
    worker_idx = 1
    per_step_blocks = 500
    for offset in range(0, len(dst_blocks), single_worker_num_blocks):
        src_to_dst = dict(enumerate(dst_blocks[offset:offset+single_worker_num_blocks]))
        src_blocks = list(src_to_dst.keys())
        dst_blocks = list(src_to_dst.values())
        for idx in range(0, len(src_blocks), per_step_blocks):
            cur_src_blocks = src_blocks[idx:idx+per_step_blocks]
            cur_dst_blocks = dst_blocks[idx:idx+per_step_blocks]
            migration_tasks.append(workers[0].execute_method.remote(
                'migrate_cache',
                src_worker_handle_list=[workers[worker_idx]],
                src_blocks=cur_src_blocks,
                dst_blocks=cur_dst_blocks)
            )
        worker_idx += 1
    ray.get(migration_tasks)

    dst_worker_data = ray.get(workers[0].execute_method.remote('get_gpu_cache'))

    worker_idx = 1
    for offset in range(0, len(dst_blocks), single_worker_num_blocks):
        src_to_dst = dict(enumerate(dst_blocks[offset:offset+single_worker_num_blocks]))

        for layer_idx in range(num_layers):
            for src_idx, dst_idx in src_to_dst.items():
                assert torch.allclose(worker_datas[worker_idx][layer_idx][0][src_idx], dst_worker_data[layer_idx][0][dst_idx])
                assert torch.allclose(worker_datas[worker_idx][layer_idx][1][src_idx], dst_worker_data[layer_idx][1][dst_idx])
        worker_idx += 1

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU to run the test.")
@pytest.mark.parametrize("backend", ['rayrpc'])
def test_many_one_to_one_migrate_cache(ray_env, backend):
    engine_config = EngineArgs(model='facebook/opt-125m', max_model_len=8, enforce_eager=True).create_engine_config()
    migraiton_config = InstanceArgs(migration_buffer_blocks=3, migration_num_layers=5,
                                    migration_num_buffers=3).create_migration_config()
    migraiton_config.migration_backend = backend

    num_workers = 2
    num_gpu_blocks = 1000
    workers, _ = get_ready_workers(num_workers, num_gpu_blocks, engine_config, migraiton_config)

    num_layers = engine_config.model_config.get_num_layers(engine_config.parallel_config)
    head_size = engine_config.model_config.get_head_size()
    num_heads = engine_config.model_config.get_num_kv_heads(engine_config.parallel_config)
    block_size = engine_config.cache_config.block_size
    dummy_data = torch.randn(size=(num_layers, 2, num_gpu_blocks, block_size, num_heads, head_size))

    worker_datas = [0]
    for idx in range(1, num_workers):
        ray.get(workers[idx].execute_method.remote('set_gpu_cache', data=dummy_data))
        worker_datas.append(ray.get(workers[idx].execute_method.remote('get_gpu_cache')))

    dst_blocks = list(range(num_gpu_blocks))
    random.shuffle(dst_blocks)

    num_migrations = 3
    single_migration_num_blocks = len(dst_blocks)//(num_migrations-1)
    migration_tasks = []
    per_step_blocks = 500
    for offset in range(0, len(dst_blocks), single_migration_num_blocks):
        src_to_dst = dict(enumerate(dst_blocks[offset:offset+single_migration_num_blocks]))
        src_blocks = list(src_to_dst.keys())
        dst_blocks = list(src_to_dst.values())
        for idx in range(0, len(src_blocks), per_step_blocks):
            cur_src_blocks = src_blocks[idx:idx+per_step_blocks]
            cur_dst_blocks = dst_blocks[idx:idx+per_step_blocks]
            migration_tasks.append(workers[0].execute_method.remote(
                'migrate_cache',
                src_worker_handle_list=[workers[1]],
                src_blocks=cur_src_blocks,
                dst_blocks=cur_dst_blocks)
            )
    ray.get(migration_tasks)

    dst_worker_data = ray.get(workers[0].execute_method.remote('get_gpu_cache'))

    for offset in range(0, len(dst_blocks), single_migration_num_blocks):
        src_to_dst = dict(enumerate(dst_blocks[offset:offset+single_migration_num_blocks]))

        for layer_idx in range(num_layers):
            for src_idx, dst_idx in src_to_dst.items():
                assert torch.allclose(worker_datas[1][layer_idx][0][src_idx], dst_worker_data[layer_idx][0][dst_idx])
                assert torch.allclose(worker_datas[1][layer_idx][1][src_idx], dst_worker_data[layer_idx][1][dst_idx])

@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Need at least 4 GPU to run the test.")
@pytest.mark.parametrize("backend", ['rayrpc'])
def test_many_to_many_migrate_cache(ray_env, backend):
    engine_config = EngineArgs(model='facebook/opt-125m', max_model_len=8, enforce_eager=True).create_engine_config()
    migraiton_config = InstanceArgs(migration_buffer_blocks=3, migration_num_layers=5,
                                    migration_num_buffers=3).create_migration_config()
    migraiton_config.migration_backend = backend

    num_workers = 4
    num_gpu_blocks = 1000
    workers, _ = get_ready_workers(num_workers, num_gpu_blocks, engine_config, migraiton_config)

    num_layers = engine_config.model_config.get_num_layers(engine_config.parallel_config)
    head_size = engine_config.model_config.get_head_size()
    num_heads = engine_config.model_config.get_num_kv_heads(engine_config.parallel_config)
    block_size = engine_config.cache_config.block_size

    worker_datas = []
    for worker_idx in range(num_workers):
        torch.manual_seed(worker_idx)
        dummy_data = torch.randn(size=(num_layers, 2, num_gpu_blocks, block_size, num_heads, head_size))
        ray.get(workers[worker_idx].execute_method.remote('set_gpu_cache', data=dummy_data))
        worker_datas.append(ray.get(workers[worker_idx].execute_method.remote('get_gpu_cache')))

    migration_pairs = [
        (0, 2),
        (1, 3),
        (2, 0),
        (3, 0),
    ]

    blocks_per_migration = num_gpu_blocks // 4
    migration_ranges = [
        (0, blocks_per_migration),
        (blocks_per_migration, 2 * blocks_per_migration),
        (2 * blocks_per_migration, 3 * blocks_per_migration),
        (3 * blocks_per_migration, 4 * blocks_per_migration),
    ]

    migration_tasks = []
    per_step_blocks = 500

    for (src_idx, dst_idx), (start_block, end_block) in zip(migration_pairs, migration_ranges):
        src_blocks = list(range(start_block, end_block))
        dst_blocks = list(range(start_block, end_block))

        for idx in range(0, len(src_blocks), per_step_blocks):
            cur_src_blocks = src_blocks[idx:idx+per_step_blocks]
            cur_dst_blocks = dst_blocks[idx:idx+per_step_blocks]

            migration_tasks.append(workers[dst_idx].execute_method.remote(
                'migrate_cache',
                src_worker_handle_list=[workers[src_idx]],
                src_blocks=cur_src_blocks,
                dst_blocks=cur_dst_blocks
            ))

    ray.get(migration_tasks)

    final_worker_datas = []
    for idx in range(num_workers):
        final_worker_datas.append(ray.get(workers[idx].execute_method.remote('get_gpu_cache')))

    for (src_idx, dst_idx), (start_block, end_block) in zip(migration_pairs, migration_ranges):
        for layer_idx in range(num_layers):
            for block_offset in range(start_block, end_block):
                assert torch.allclose(
                    worker_datas[src_idx][layer_idx][0][block_offset],
                    final_worker_datas[dst_idx][layer_idx][0][block_offset]
                ), f"Mismatch at layer {layer_idx}, block {block_offset} from worker {src_idx} to {dst_idx}"
                assert torch.allclose(
                    worker_datas[src_idx][layer_idx][1][block_offset],
                    final_worker_datas[dst_idx][layer_idx][1][block_offset]
                ), f"Mismatch at layer {layer_idx}, block {block_offset} from worker {src_idx} to {dst_idx}"

    for worker_idx in range(num_workers):
        for (_, dst_idx), (start_block, end_block) in zip(migration_pairs, migration_ranges):
            if worker_idx == dst_idx:
                continue
            for layer_idx in range(num_layers):
                for block_offset in range(start_block, end_block):
                    assert torch.allclose(
                        worker_datas[worker_idx][layer_idx][0][block_offset],
                        final_worker_datas[worker_idx][layer_idx][0][block_offset]
                    ), f"Unexpected change at worker {worker_idx}, layer {layer_idx}, block {block_offset}"
