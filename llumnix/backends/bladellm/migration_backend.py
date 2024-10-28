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

from typing import List
import torch
import grpc
import numpy as np

from google.protobuf import empty_pb2

from blade_llm.generation.statemanagers.base_state_manager import StateManagerBase
from blade_llm.generation.kvcache.kv_cache_arena import PagedKVCacheArena

from llumnix.internal_config import MigrationConfig
from llumnix.backends.migration_backend_interface import MigrationBackendBase
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.logger import init_logger


logger = init_logger(__name__)

NUMPY_SUPPORTED_DTYPES = [torch.float32, torch.float16]

class GrpcMigrationBackend(MigrationBackendBase):
    def __init__(self, worker_address: str, migration_config: MigrationConfig, state_manager: StateManagerBase):
        self.worker_address = worker_address
        self.state_manager = state_manager
        self.num_migration_cache_blocks = migration_config.migration_cache_blocks
        migration_cache_key_shape = list([len(state_manager._kv_cache[0])]) + list(state_manager._kv_cache[0][0].shape)
        migration_cache_key_shape[1] = migration_config.migration_cache_blocks
        migration_cache_value_shape = list([len(state_manager._kv_cache[1])]) + list(state_manager._kv_cache[1][0].shape)
        migration_cache_value_shape[1] = migration_config.migration_cache_blocks

        if state_manager.dtype in NUMPY_SUPPORTED_DTYPES:
            self.migration_cache_dtype = state_manager.dtype
        else:
            self.migration_cache_dtype = torch.float32
            logger.warning("Detecting numpy unsupported dtype: {}, using torch.float32."
                           .format(state_manager.dtype))
        self.np_migration_cache_dtype = torch.tensor([], dtype=self.migration_cache_dtype).numpy().dtype

        self.dummy_key_cache = torch.empty(
            size=migration_cache_key_shape,
            dtype=self.migration_cache_dtype,
            pin_memory=True
        )
        self.dummy_value_cache = torch.empty(
            size=migration_cache_value_shape,
            dtype=self.migration_cache_dtype,
            pin_memory=True
        )

        engine_kv_cache_arean: PagedKVCacheArena = state_manager.kv_cache_arena
        kv_cache_mem_dict = {}
        kv_cache_mem_dict["gpu_k_cache"] = engine_kv_cache_arean.gpu_k_cache
        kv_cache_mem_dict["gpu_v_cache"] = engine_kv_cache_arean.gpu_v_cache
        kv_cache_mem_dict["cpu_k_cache"] = self.dummy_key_cache
        kv_cache_mem_dict["cpu_v_cache"] = self.dummy_value_cache
        self.kv_cache_arena = PagedKVCacheArena(engine_kv_cache_arean.num_layers, kv_cache_mem_dict)

    def init_backend(self, group_name, world_size, rank) -> bool:
        logger.info("create grpc migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        pass

    def warmup(self) -> bool:
        with grpc.insecure_channel(self.worker_address) as channel:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)
            responce = stub.is_ready(empty_pb2.Empty())
            return responce.is_ready

    def migrate_cache(self, src_handle: str, src_blocks: List[int], dst_blocks: List[int]) -> None:
        with grpc.insecure_channel(src_handle) as channel:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)

            tot_blocks = len(src_blocks)
            for start_idx in range(0, tot_blocks, self.num_migration_cache_blocks):
                offset = min(self.num_migration_cache_blocks, tot_blocks - start_idx)
                cur_src_blocks = src_blocks[start_idx:start_idx+offset]
                cur_dst_blocks = dst_blocks[start_idx:start_idx+offset]
                response = stub.do_send(migration_worker_pb2.SendKvCacheRequest(blocks=cur_src_blocks))
                # TODO(KuilongCui): overlap stub.do_send and do_recv
                self.do_recv(response, cur_dst_blocks)
            
            for layer_idx in range(self.kv_cache_arena.num_layers):
                self.kv_cache_arena.events[layer_idx].wait()

    # pylint: disable=unused-argument
    def do_send(self, request, context):
        blocks = list(request.blocks)

        num_blocks = len(blocks)
        send_key_cache = self.dummy_key_cache[:, :num_blocks]
        send_value_cache = self.dummy_value_cache[:, :num_blocks]

        mapping = [None] * (2 * num_blocks)
        mapping[::2] = blocks
        mapping[1::2] = range(num_blocks)

        self.kv_cache_arena.swap_blocks(None, mapping)

        for layer_idx in range(self.kv_cache_arena.num_layers):
            self.kv_cache_arena.events[layer_idx].wait()

        responce = migration_worker_pb2.SendKvCacheResponse(key=send_key_cache.numpy().tobytes(),
                                                            value=send_value_cache.numpy().tobytes())
        return responce

    def do_recv(self, src_handle, blocks: List[int]):
        # use pin memory dummy_cache to speed up data transfer
        num_blocks = len(blocks)
        recv_key_cache = self.dummy_key_cache[:, :num_blocks]
        recv_value_cache = self.dummy_value_cache[:, :num_blocks]
        key, value = src_handle.key, src_handle.value

        recv_key_cache.copy_(torch.from_numpy(np.frombuffer(
            key, dtype=self.np_migration_cache_dtype).reshape(recv_key_cache.shape)))
        recv_value_cache.copy_(torch.from_numpy(np.frombuffer(
            value, dtype=self.np_migration_cache_dtype).reshape(recv_value_cache.shape)))

        mapping = [None] * (2 * num_blocks)
        mapping[::2] = range(num_blocks)
        mapping[1::2] = blocks
        self.kv_cache_arena.swap_blocks(mapping, None)

def get_migration_backend(worker_address: str, migration_config: MigrationConfig, state_manager: StateManagerBase) -> MigrationBackendBase:
    target_migration_backend = None

    backend = migration_config.migration_backend
    assert backend in ['grpc', 'kvtransfer']
    if backend == 'grpc':
        target_migration_backend = GrpcMigrationBackend(worker_address, migration_config, state_manager)
    else:
        assert False, "Not Support, now."
    return target_migration_backend
