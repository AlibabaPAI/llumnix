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
from typing import List
import torch
import grpc
import numpy as np

from blade_llm.generation.statemanagers.base_state_manager import StateManagerBase
from blade_llm.generation.statemanagers.ragged_flash_state_manager import RaggedFlashStateManager
from blade_llm.generation.statemanagers.paged_state_manager import PagedStateManager
from blade_llm.generation.kvcache.kv_cache_arena import PagedKVCacheArena, RaggedFlashKVCacheArena
from blade_llm.generation.kvcache.kv_transfer import KVTransferClient, KVTransferServer, TransferType
from blade_llm.service.args import ServingArgs

from llumnix.internal_config import MigrationConfig
from llumnix.backends.migration_backend_interface import MigrationBackendBase
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.logger import init_logger
from llumnix.utils import random_uuid


logger = init_logger(__name__)

NUMPY_SUPPORTED_DTYPES = [torch.float32, torch.float16]

class AddressHandle:
    ip_address: None
    def __init__(self, ip_address, instance_id=None, worker_id=None):
        self.ip_address = ip_address
        self.instance_id = instance_id
        self.worker_id = worker_id

class GrpcMigrationBackend(MigrationBackendBase):
    def __init__(self, rank: int, migration_config: MigrationConfig, state_manager: StateManagerBase):
        self.worker_address = migration_config.migration_backend_server_address.split(",")[rank]
        self.state_manager = state_manager
        self.num_migration_cache_blocks = migration_config.migration_cache_blocks

        # pylint: disable=invalid-name
        # TODO(KuilongCui): check the best value for MAX_MESSAGE_LENGHT
        MAX_MESSAGE_LENGHT = 1024 * 1024 * 1024
        self.channel_options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGHT),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGHT),
        ]

        if state_manager.dtype in NUMPY_SUPPORTED_DTYPES:
            self.migration_cache_dtype = state_manager.dtype
        else:
            self.migration_cache_dtype = torch.float32
            logger.warning("Detecting numpy unsupported dtype: {}, using torch.float32."
                           .format(state_manager.dtype))
        self.np_migration_cache_dtype = torch.tensor([], dtype=self.migration_cache_dtype).numpy().dtype

        if isinstance(state_manager, PagedStateManager):
            num_layer = len(state_manager._kv_cache[0])
            migration_cache_key_shape = list([num_layer]) + list(state_manager._kv_cache[0][0].shape)
            migration_cache_value_shape = list([num_layer]) + list(state_manager._kv_cache[1][0].shape)
            migration_cache_key_shape[1] = migration_config.migration_cache_blocks
            migration_cache_value_shape[1] = migration_config.migration_cache_blocks

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
        elif isinstance(state_manager, RaggedFlashStateManager):
            num_layer = len(state_manager.kv_cache_arena.gpu_kv_cache)
            gpu_kv_cache = state_manager.kv_cache_arena.gpu_kv_cache
            migration_cache_kv_shape = list([num_layer]) + list(gpu_kv_cache[0].shape)
            migration_cache_kv_shape[1] = migration_config.migration_cache_blocks

            self.dummy_kv_cache = torch.empty(
                size=migration_cache_kv_shape,
                dtype=self.migration_cache_dtype,
                pin_memory=True
            )

            kv_cache_mem_dict = {}
            kv_cache_mem_dict["gpu_kv_cache"] = state_manager.kv_cache_arena.gpu_kv_cache
            kv_cache_mem_dict["cpu_kv_cache"] = self.dummy_kv_cache
            self.kv_cache_arena = RaggedFlashKVCacheArena(num_layer, kv_cache_mem_dict)
        else:
            raise RuntimeError("Unsupported state manager type: {}".format(type(state_manager)))

    def init_backend(self, group_name, world_size, rank) -> bool:
        logger.info("create grpc migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        pass

    def warmup(self) -> bool:
        self.migrate_cache(AddressHandle(ip_address=self.worker_address), [0], [1])
        return True

    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        with grpc.insecure_channel(src_handle.ip_address, options=self.channel_options) as channel:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)

            tot_blocks = len(src_blocks)
            for start_idx in range(0, tot_blocks, self.num_migration_cache_blocks):
                offset = min(self.num_migration_cache_blocks, tot_blocks - start_idx)
                cur_src_blocks = src_blocks[start_idx:start_idx+offset]
                cur_dst_blocks = dst_blocks[start_idx:start_idx+offset]
                response = stub.do_send(migration_worker_pb2.SendKvCacheRequest(src_blocks=cur_src_blocks))
                # TODO(KuilongCui): overlap stub.do_send and do_recv
                self.do_recv(response, cur_dst_blocks)

            for layer_idx in range(self.kv_cache_arena.num_layers):
                self.kv_cache_arena.events[layer_idx].wait()

    # pylint: disable=unused-argument
    def do_send(self, request, context):
        blocks = list(request.src_blocks)

        num_blocks = len(blocks)
        mapping = [None] * (2 * num_blocks)
        mapping[::2] = blocks
        mapping[1::2] = range(num_blocks)

        self.kv_cache_arena.swap_blocks(None, mapping)

        for layer_idx in range(self.kv_cache_arena.num_layers):
            self.kv_cache_arena.events[layer_idx].wait()

        if isinstance(self.state_manager, PagedStateManager):
            send_key_cache = self.dummy_key_cache[:, :num_blocks]
            send_value_cache = self.dummy_value_cache[:, :num_blocks]
            responce = migration_worker_pb2.SendKvCacheResponse(
                key=send_key_cache.numpy().tobytes(),
                value=send_value_cache.numpy().tobytes()
            )
        elif isinstance(self.state_manager, RaggedFlashStateManager):
            send_kv_cache = self.dummy_kv_cache[:, :num_blocks]
            responce = migration_worker_pb2.SendKvCacheResponse(
                kv=send_kv_cache.numpy().tobytes()
            )
        else:
            raise RuntimeError("Unsupported state manager type: {}".format(type(self.state_manager)))

        return responce

    def do_recv(self, src_handle, blocks: List[int]):
        # use pin memory dummy_cache to speed up data transfer
        num_blocks = len(blocks)

        if isinstance(self.state_manager, PagedStateManager):
            recv_key_cache = self.dummy_key_cache[:, :num_blocks]
            recv_value_cache = self.dummy_value_cache[:, :num_blocks]
            recv_key_cache.copy_(torch.from_numpy(np.frombuffer(
                src_handle.key, dtype=self.np_migration_cache_dtype).reshape(recv_key_cache.shape)))
            recv_value_cache.copy_(torch.from_numpy(np.frombuffer(
                src_handle.value, dtype=self.np_migration_cache_dtype).reshape(recv_value_cache.shape)))
        elif isinstance(self.state_manager, RaggedFlashStateManager):
            recv_kv_cache = self.dummy_kv_cache[:, :num_blocks]
            recv_kv_cache.copy_(torch.from_numpy(np.frombuffer(
                src_handle.kv, dtype=self.np_migration_cache_dtype).reshape(recv_kv_cache.shape)))
        else:
            raise RuntimeError("Unsupported state manager type: {}".format(type(self.state_manager)))

        mapping = [None] * (2 * num_blocks)
        mapping[::2] = range(num_blocks)
        mapping[1::2] = blocks
        self.kv_cache_arena.swap_blocks(mapping, None)

transfer_methods = {"cuda_ipc", "rdma"}

def string_to_enum(transfer_str):
    if transfer_str == "cuda_ipc":
        return TransferType.CUDA_IPC_DIRECT
    elif transfer_str == "rdma":
        return TransferType.RDMA_DIRECT
    else:
        return None

class KvTransferMigrationBackend(MigrationBackendBase):
    def __init__(self, rank: int, instance_id: int, worker_id: int, migration_config: MigrationConfig,
                 serving_args: ServingArgs, state_manager: StateManagerBase):
        self.instance_id = instance_id
        self.worker_id = worker_id
        self.worker_address = migration_config.migration_backend_server_address.split(",")[rank]
        self.serving_args = serving_args
        self.state_manager = state_manager

        self.tranfer_type = string_to_enum(migration_config.migration_backend_transfer_type)
        assert self.tranfer_type in [TransferType.RDMA_DIRECT, TransferType.CUDA_IPC_DIRECT]

        # TODO(KuilongCui): support PagedStateManager
        assert isinstance(state_manager, RaggedFlashStateManager)

        self.kv_cache = state_manager.kv_cache_arena.gpu_kv_cache
        token_tensor = self.kv_cache[0][0][0]
        token_bytes = token_tensor.element_size() * token_tensor.numel()
        block_bytes = self.state_manager.block_size * token_bytes

        naming_url = migration_config.migration_backend_kvtransfer_naming_url
        self.client_kv = KVTransferClient(instance_id, serving_args.tensor_parallel_size, worker_id, rank,
                                       block_bytes, token_bytes, self.tranfer_type, naming_url,
                                       self.state_manager._kv_cache)

        self.server_kv = KVTransferServer(instance_id, serving_args.tensor_parallel_size, worker_id, rank,
                                       block_bytes, token_bytes, self.tranfer_type, naming_url,
                                       self.state_manager._kv_cache)

    def init_backend(self, group_name, world_size, rank) -> bool:
        logger.info("create kvtransfer migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        pass

    def warmup(self) -> bool:
        # CUDA_IPC_DIRECT does not support communication with itself; used only for RDMA_DIRECT warmup
        if self.tranfer_type == TransferType.RDMA_DIRECT:
            self.migrate_cache(AddressHandle(ip_address=self.worker_address, instance_id=self.instance_id,
                worker_id=self.worker_id), [0], [1])

        return True

    def migrate_cache(self, src_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        with grpc.insecure_channel(src_handle.ip_address) as channel:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)
            migration_kv_uuid = random_uuid()

            self.server_kv.submit_req_recv(src_handle.instance_id, src_handle.worker_id, migration_kv_uuid, dst_blocks)

            stub.do_send(migration_worker_pb2.SendKvCacheRequest(
                dst_instance_id=self.instance_id,
                dst_worker_id=self.worker_id,
                src_blocks=src_blocks,
                dst_blocks=dst_blocks,
                kv_request_id=migration_kv_uuid,
            ))
            self.check_recv_done(src_handle.instance_id, src_handle.worker_id, migration_kv_uuid,
                                 src_blocks, dst_blocks)

    # pylint: disable=unused-argument
    def do_send(self, request, context):
        dst_instance_id, dst_worker_id = request.dst_instance_id, request.dst_worker_id
        self.client_kv.add_worker(dst_instance_id, dst_worker_id, 0, len(self.kv_cache))
        num_token = self.state_manager.block_size * len(request.src_blocks)

        self.client_kv.submit_req_send(dst_instance_id, dst_worker_id, request.kv_request_id, num_token,
                                      True, request.src_blocks, request.dst_blocks)

        self.client_kv.start_send_step()

        self.client_kv.flush_send_step()

        return migration_worker_pb2.SendKvCacheResponse()

    def check_recv_done(self, src_instance_id, src_worker_id, kv_request_id: str,
                        src_blocks: List[int], dst_blocks: List[int]):
        timeout_threshold_ms = 30
        escape_time = 0
        while escape_time < timeout_threshold_ms:
            kv_transfer_done =  self.server_kv.check_req_transfer_done(kv_request_id)
            if kv_transfer_done:
                self.server_kv.clear_done_reqs([kv_request_id])
                return
            time.sleep(0.01)
            escape_time += 0.01

        raise RuntimeError("Kvtransfer migrate cache req [kv:{}] timeout: src instance id {}, \
            src worker id {}, dst instance id {}, dst worker id {}, src blocks: {}, dst blocks: {}"
            .format(kv_request_id, src_instance_id, src_worker_id, self.instance_id,
            self.worker_id, src_blocks, dst_blocks))

def get_migration_backend(instance_id: int, worker_id: int, rank: int,
                          migration_config: MigrationConfig, state_manager: StateManagerBase,
                          serving_args: ServingArgs) -> MigrationBackendBase:
    target_migration_backend = None

    backend = migration_config.migration_backend
    assert backend in ['grpc', 'kvtransfer']
    if backend == 'grpc':
        target_migration_backend = GrpcMigrationBackend(rank, migration_config, state_manager)
    else:
        target_migration_backend = KvTransferMigrationBackend(rank, instance_id, worker_id, migration_config,
                                                              serving_args, state_manager)
    return target_migration_backend