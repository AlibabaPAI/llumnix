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

# pylint: disable=protected-access

import threading
import time
from typing import List, Tuple

import pickle
import torch
import grpc
import numpy as np

from blade_kvt import nic_affinity
from blade_kvt.kv_transfer import (
    KVTransferClient, KVTransferServer, KVTransferProtocolType, connect_naming)
from blade_llm.generation.statemanagers.base_state_manager import StateManagerBase
from blade_llm.generation.statemanagers.ragged_flash_state_manager import RaggedFlashStateManager
from blade_llm.generation.statemanagers.disagg_ragged_flash_state_manager import (
    DisaggPrefillStateManager, DisaggDecodeStateManager)
from blade_llm.generation.statemanagers.paged_state_manager import PagedStateManager
from blade_llm.generation.kvcache.kv_cache_arena import PagedKVCacheArena, RaggedFlashKVCacheArena
from blade_llm.service.args import ServingArgs
from blade_llm.service.tracing import ReqTracker
from blade_llm.service.workers.base_worker import BaseWorker
from blade_llm.service.proto.bladellm_pb2 import (SamplingParams, LogitsProcessorParams,
                                                  DetokenParams, StoppingCriteria, WorkerRequest)

from llumnix.utils import get_ip_address
from llumnix.internal_config import MigrationConfig
from llumnix.backends.migration_backend_interface import MigrationBackendBase
from llumnix.backends.bladellm.proto import migration_worker_pb2_grpc, migration_worker_pb2
from llumnix.backends.bladellm.proto.migration_worker_pb2 import WorkerInfo, RecvCacheRequest
from llumnix.logging.logger import init_logger
from llumnix.constants import (
    GRPC_MAX_MESSAGE_LENGTH,
    NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION,
    GRPC_TIMEOUT,
    KVTRANSFER_MIGRATION_TIMEOUT,
    GRPC_MIGRATION_TIMEOUT,
)

logger = init_logger(__name__)


# TODO(KuilongCui): Refactor this code, as there is a lot of duplicate code.
# TODO(KuilongCui): Add more type hint for BladeLLM.

class WorkerRequestSyncGroup:
    def __init__(self, request_group, request_tracker):
        # TODO(KuilongCui): use rw lock to improve performance
        self.lock = threading.Lock()
        self.request_group = request_group # state_manager.request_group
        self.request_tracker: ReqTracker = request_tracker # TODO(KuilongCui): handle the concurrency issue

        # Backup migrated out request worker meta to avoid concurrent access to state_manager.request_group and
        # request_tracker
        self._backup_state_manager_groups = {}
        self._backup_request_tracker = {}
        # The worker meta of a migrated-in request is first added to _new_request_groups and _new_request_tracker
        # to prevent concurrent access to state_manager.request_group, and will be added to state_manager in the
        # next handle_rpc_call
        self._new_request_groups = {}
        self._new_request_tracker = {}

    def update_migrated_in_request(self):
        with self.lock:
            self.request_group.update(self._new_request_groups)
            self.request_tracker.req_metrics_map.update(self._new_request_tracker)

            self._new_request_groups.clear()
            self._new_request_tracker.clear()

    def add_new_request(self, request_group_id, request_group_data, request_tracker_data):
        with self.lock:
            self._new_request_groups[request_group_id] = request_group_data
            self._new_request_tracker[request_group_id] = request_tracker_data

    def backup_request_group(self, request_group_ids):
        with self.lock:
            for request_group_id in request_group_ids:
                self._backup_state_manager_groups[request_group_id] = self.request_group[request_group_id]
                self._backup_request_tracker[request_group_id] = self.request_tracker.get_req_metrics(request_group_id)

    def remove_backup_request_group(self, request_group_ids):
        with self.lock:
            for request_group_id in request_group_ids:
                self._backup_state_manager_groups.pop(request_group_id, None)
                self._backup_request_tracker.pop(request_group_id, None)

    def get_request_meta(self, request_group_id):
        with self.lock:
            assert request_group_id in self._backup_state_manager_groups
            assert request_group_id in self._backup_request_tracker
            state_manager_data = self._backup_state_manager_groups[request_group_id]
            request_tracker_data = self._backup_request_tracker[request_group_id]
            return state_manager_data, request_tracker_data


class GrpcMigrationBackend(MigrationBackendBase):
    def __init__(self,
                 migration_config: MigrationConfig,
                 request_sync_group: WorkerRequestSyncGroup,
                 state_manager: StateManagerBase):
        self.request_sync_group: WorkerRequestSyncGroup = request_sync_group
        self.worker_migration_ip_addr = get_ip_address() + ":" + str(migration_config.grpc_migration_server_port)
        self.state_manager = state_manager
        self.num_migration_buffer_blocks = migration_config.migration_buffer_blocks

        # pylint: disable=invalid-name
        self.channel_options=[
            ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH),
        ]

        if state_manager.dtype in NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION:
            self.migration_cache_dtype = state_manager.dtype
        else:
            self.migration_cache_dtype = torch.float32
            logger.warning("Detecting numpy unsupported dtype: {}, using torch.float32.".format(state_manager.dtype))
        self.np_migration_cache_dtype = torch.tensor([], dtype=self.migration_cache_dtype).numpy().dtype

        if isinstance(state_manager, PagedStateManager):
            num_layer = len(state_manager._kv_cache[0])
            migration_cache_key_shape = list([num_layer]) + list(state_manager._kv_cache[0][0].shape)
            migration_cache_value_shape = list([num_layer]) + list(state_manager._kv_cache[1][0].shape)
            migration_cache_key_shape[1] = migration_config.migration_buffer_blocks
            migration_cache_value_shape[1] = migration_config.migration_buffer_blocks

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
            migration_cache_kv_shape[1] = migration_config.migration_buffer_blocks

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

    def init_backend(self, group_name: str, world_size: int, rank: int) -> bool:
        logger.info("create grpc migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        pass

    def warmup(self) -> bool:
        self.state_manager.add_new_request(self._create_dummy_worker_request())
        self.recv_cache(
            RecvCacheRequest(
                src_worker_handle_list=[WorkerInfo(ip_address=self.worker_migration_ip_addr)],
                request_id=0,
                is_last_stage=True,
                src_blocks=[0],
                dst_blocks=[1],
            ),
            src_blocks=[0],
            dst_blocks=[1],
        )
        self.state_manager._request_groups.pop(0, None)
        return True

    # pylint: disable=arguments-differ
    def recv_cache(self, src_worker_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        ip_address = src_worker_handle.src_worker_handle_list[self.state_manager.rank].ip_address
        src_blocks = src_worker_handle.src_blocks
        dst_blocks = src_worker_handle.dst_blocks
        with grpc.insecure_channel(ip_address, options=self.channel_options) as channel:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)
            tot_blocks = len(src_blocks)
            for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
                offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
                cur_src_blocks = src_blocks[start_idx:start_idx+offset]
                cur_dst_blocks = dst_blocks[start_idx:start_idx+offset]
                response = stub.do_send(
                    migration_worker_pb2.SendKVCacheRequest(
                        request_id=src_worker_handle.request_id,
                        src_blocks=cur_src_blocks,
                        is_last_stage=(src_worker_handle.is_last_stage and start_idx+offset==tot_blocks)
                    ),
                    timeout=GRPC_MIGRATION_TIMEOUT,
                )
                # TODO(KuilongCui): overlap stub.do_send and do_recv
                self.do_recv(response, cur_dst_blocks)

            if src_worker_handle.is_last_stage:
                assert response.state_manager_data and len(response.state_manager_data) > 0, "Invalid state manager meta"
                state_manager_data = pickle.loads(response.state_manager_data)
                request_tracker_data = pickle.loads(response.request_tracker_data)
                self.request_sync_group.add_new_request(
                    response.request_id, state_manager_data, request_tracker_data)

            for layer_idx in range(self.kv_cache_arena.num_layers):
                self.kv_cache_arena.events[layer_idx].wait()

    # pylint: disable=unused-argument,arguments-differ
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
            response = migration_worker_pb2.SendKvCacheResponse(
                request_id = request.request_id,
                key=send_key_cache.numpy().tobytes(),
                value=send_value_cache.numpy().tobytes()
            )
        elif isinstance(self.state_manager, RaggedFlashStateManager):
            send_kv_cache = self.dummy_kv_cache[:, :num_blocks]
            response = migration_worker_pb2.SendKvCacheResponse(
                request_id = request.request_id,
                kv=send_kv_cache.numpy().tobytes()
            )
        else:
            raise RuntimeError("Unsupported state manager type: {}".format(type(self.state_manager)))

        # set request state manager meta
        if request.is_last_stage:
            state_manager_data, request_tracker_data = self.request_sync_group.get_request_meta(request.request_id)
            # some pb repeated field is not allowed in pickle, so we use list to replace it
            for request_state in state_manager_data.request_states:
                request_state.vlm_prompt = list(request_state.vlm_prompt)
                request_state.prompt_tokens = list(request_state.prompt_tokens)

            response.state_manager_data = pickle.dumps(state_manager_data)
            response.request_tracker_data = pickle.dumps(request_tracker_data)
            logger.debug("Pickle state manager meta for request id {}, data length: {}".format(
                request.request_id, len(response.state_manager_data)))

        return response

    # pylint: disable=unused-argument,arguments-differ
    def do_recv(self, src_worker_handle, blocks: List[int]) -> None:
        # use pin memory dummy_cache to speed up data transfer
        num_blocks = len(blocks)
        if isinstance(self.state_manager, PagedStateManager):
            recv_key_cache = self.dummy_key_cache[:, :num_blocks]
            recv_value_cache = self.dummy_value_cache[:, :num_blocks]
            recv_key_cache.copy_(torch.from_numpy(np.frombuffer(
                src_worker_handle.key, dtype=self.np_migration_cache_dtype).reshape(recv_key_cache.shape)))
            recv_value_cache.copy_(torch.from_numpy(np.frombuffer(
                src_worker_handle.value, dtype=self.np_migration_cache_dtype).reshape(recv_value_cache.shape)))
        elif isinstance(self.state_manager, RaggedFlashStateManager):
            recv_kv_cache = self.dummy_kv_cache[:, :num_blocks]
            recv_kv_cache.copy_(torch.from_numpy(np.frombuffer(
                src_worker_handle.kv, dtype=self.np_migration_cache_dtype).reshape(recv_kv_cache.shape)))
        else:
            raise RuntimeError("Unsupported state manager type: {}".format(type(self.state_manager)))

        mapping = [None] * (2 * num_blocks)
        mapping[::2] = range(num_blocks)
        mapping[1::2] = blocks
        self.kv_cache_arena.swap_blocks(mapping, None)

    def _create_dummy_worker_request(self):
        return WorkerRequest(
            id=0,
            prompt="hello",
            prompt_tokens=[373],
            in_flight_tokens=1,
            seen_tokens=0,
            sampling_params=SamplingParams(temperature=0.7, top_p=0.85, top_k=30),
            stopping_criterial=StoppingCriteria(max_new_tokens=0),
            logits_processors_params=LogitsProcessorParams(repetition_penalty=1.0),
            detoken_params=DetokenParams(cat_prompt=True),
        )


def get_kv_tranfer_context(statemanager: RaggedFlashStateManager) -> Tuple[int, int]:
    assert isinstance(statemanager, RaggedFlashStateManager)
    kv_tranfer_unit = (
        statemanager.model_conf.num_attention_heads // statemanager.attn_head_tp_size
        if statemanager.model_conf.num_query_group is None
        else max(1, statemanager.model_conf.num_query_group // statemanager.attn_head_tp_size)
    )
    kv_transfer_token_bytes = 2 * statemanager.dtype.itemsize * statemanager.model_conf.head_dim * kv_tranfer_unit
    kv_transfer_block_bytes = statemanager.block_size * kv_transfer_token_bytes
    return kv_transfer_token_bytes, kv_transfer_block_bytes


class KvTransferMigrationBackend(MigrationBackendBase):
    def __init__(self,
                 rank: int,
                 instance_id: str,
                 worker_id: int,
                 migration_config: MigrationConfig,
                 request_sync_group: WorkerRequestSyncGroup,
                 base_worker: BaseWorker,
                 serving_args: ServingArgs,
                 state_manager: StateManagerBase):
        nic_affinity.generate()
        self.instance_id = instance_id
        self.worker_id = worker_id
        self.worker_migration_ip_addr = get_ip_address() + ":" + str(migration_config.grpc_migration_server_port)
        self.serving_args = serving_args
        self.state_manager = state_manager
        self.request_sync_group = request_sync_group

        self.tranfer_type = KVTransferProtocolType.to_protocol_type(migration_config.kvtransfer_migration_backend_transfer_type)

        # TODO(KuilongCui): support PagedStateManager
        assert isinstance(state_manager, RaggedFlashStateManager)
        self.kv_cache = state_manager.kv_cache_arena.gpu_kv_cache
        token_bytes, block_bytes = get_kv_tranfer_context(state_manager)
        naming_url = migration_config.kvtransfer_migration_backend_naming_url
        if isinstance(state_manager, (DisaggPrefillStateManager, DisaggDecodeStateManager)):
            naming_url = serving_args.naming_url

        self.client_kv = getattr(self.state_manager, "_kv_client", None)
        self.server_kv = getattr(self.state_manager, "_kv_server", None)

        self.kv_transfer_instance_id = self.instance_id
        if serving_args.enable_disagg and serving_args.disagg_options is not None:
            self.kv_transfer_instance_id = serving_args.disagg_options.inst_id
        if self.client_kv is None:
            self.client_kv = KVTransferClient(self.kv_transfer_instance_id, serving_args.tensor_parallel_size, worker_id, rank,
                                              block_bytes, token_bytes, naming_url, self.state_manager._kv_cache,
                                              [self.tranfer_type])
        if self.server_kv is None:
            self.server_kv = KVTransferServer(self.kv_transfer_instance_id, serving_args.tensor_parallel_size, worker_id, rank,
                                              block_bytes, token_bytes, naming_url, self.state_manager._kv_cache,
                                              [self.tranfer_type])

            kvt_info = BaseWorker.info(base_worker, kvt_worker_kind="server").kvt_info
            self._naming_client = connect_naming(self.instance_id, migration_config.kvtransfer_migration_backend_naming_url)
            self._naming_client.store(f"worker_{rank}", kvt_info)
            logger.info("store worker info to naming server, instance_id: {}, rank: {}, kvt_info: {}".format(
                self.instance_id, rank, kvt_info))

    def init_backend(self, group_name, world_size, rank) -> bool:
        logger.info("create kvtransfer migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        pass

    def warmup(self) -> bool:
        # CUDA_IPC does not support communication with itself, used only for RDMA_DIRECT warmup
        if self.tranfer_type == KVTransferProtocolType.RDMA_DIRECT:
            self.recv_cache(
                WorkerInfo(
                    ip_address=self.worker_migration_ip_addr,
                    instance_id=self.instance_id,
                    worker_id=self.worker_id
                ),
                [0],
                [1],
            )
        return True

    # pylint: disable=arguments-differ
    def recv_cache(self, src_worker_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        ip_address = src_worker_handle.src_worker_handle_list[self.state_manager.rank].ip_address
        kv_transfer_instance_id = src_worker_handle.src_worker_handle_list[self.state_manager.rank].kv_transfer_instance_id
        worker_id = src_worker_handle.src_worker_handle_list[self.state_manager.rank].worker_id
        with grpc.insecure_channel(ip_address) as channel:
            stub = migration_worker_pb2_grpc.MigrationWorkerStub(channel)
            self.server_kv.submit_req_recv(kv_transfer_instance_id, worker_id, str(src_worker_handle.request_id), dst_blocks)
            # Note: dst_instance_id must be set to kv_transfer_instance_id, not instance_id
            response = stub.do_send(
                migration_worker_pb2.SendKVCacheRequest(
                    request_id=src_worker_handle.request_id,
                    dst_kv_transfer_instance_id=self.kv_transfer_instance_id,
                    dst_worker_id=self.worker_id,
                    src_blocks=src_blocks,
                    dst_blocks=dst_blocks,
                    is_last_stage=src_worker_handle.is_last_stage
                ),
                timeout=GRPC_TIMEOUT
            )
            if src_worker_handle.is_last_stage:
                assert response.state_manager_data and len(response.state_manager_data) > 0, "Invalid state manager meta"
                state_manager_data = pickle.loads(response.state_manager_data)
                request_tracker_data = pickle.loads(response.request_tracker_data)
                self.request_sync_group.add_new_request(
                    response.request_id, state_manager_data, request_tracker_data)
            self.do_recv(kv_transfer_instance_id, worker_id, str(src_worker_handle.request_id),
                         src_blocks, dst_blocks)

    # pylint: disable=unused-argument,arguments-differ
    def do_send(self, request, context):
        dst_kv_transfer_instance_id, dst_worker_id = request.dst_kv_transfer_instance_id, request.dst_worker_id
        self.client_kv.add_worker(dst_kv_transfer_instance_id, dst_worker_id, 0, len(self.kv_cache), self.tranfer_type)
        num_token = self.state_manager.block_size * len(request.src_blocks)
        self.client_kv.submit_req_send(dst_kv_transfer_instance_id, dst_worker_id, str(request.request_id), num_token,
                                      True, request.src_blocks, request.dst_blocks)

        self.client_kv.start_send_step()

        # async op
        self.client_kv.flush_send_step()

        # set request state manager meta
        response = migration_worker_pb2.SendKvCacheResponse(request_id=request.request_id)
        if request.is_last_stage:
            state_manager_data, request_tracker_data = self.request_sync_group.get_request_meta(request.request_id)
            # some pb repeated field is not allowed in pickle, so we use list to replace it
            for request_state in state_manager_data.request_states:
                request_state.vlm_prompt = list(request_state.vlm_prompt)
                request_state.prompt_tokens = list(request_state.prompt_tokens)

            response.state_manager_data = pickle.dumps(state_manager_data)
            response.request_tracker_data = pickle.dumps(request_tracker_data)
            logger.debug("Pickle state manager meta for request id {}, data length: {}".format(
                request.request_id, len(response.state_manager_data)))

        self.client_kv.check_req_transfer_done(str(request.request_id))

        return response

    # pylint: disable=arguments-differ
    def do_recv(self,
                src_instance_id: str,
                src_worker_id: int,
                kv_request_id: str,
                src_blocks: List[int],
                dst_blocks: List[int]) -> None:
        elapsed_time = 0
        while elapsed_time < KVTRANSFER_MIGRATION_TIMEOUT:
            kv_transfer_done = self.server_kv.check_req_transfer_done(kv_request_id)
            if kv_transfer_done:
                self.server_kv.clear_done_reqs([kv_request_id])
                return
            time.sleep(0.01)
            elapsed_time += 0.01

        raise TimeoutError(
            "KvTransfer migrate kv cache timeout after {} seconds (request_id: {}, src_instance_id: {}, "
            "src_worker_id: {}, dst_instance_id: {}, dst_worker_id: {}, src_blocks: {}, dst_blocks: {})"
            .format(kv_request_id, KVTRANSFER_MIGRATION_TIMEOUT, src_instance_id, src_worker_id, self.instance_id,
                self.worker_id, src_blocks, dst_blocks)
        )


def get_migration_backend(instance_id: str,
                          worker_id: int,
                          rank: int,
                          migration_config: MigrationConfig,
                          request_sync_group: WorkerRequestSyncGroup,
                          base_worker: BaseWorker,
                          state_manager: StateManagerBase,
                          serving_args: ServingArgs) -> MigrationBackendBase:
    assert migration_config.migration_backend in ['grpc', 'kvtransfer'], \
        "Only support grpc and kvtransfer migration backend for BladeLLM."

    target_migration_backend = None
    backend = migration_config.migration_backend
    if backend == 'grpc':
        target_migration_backend = GrpcMigrationBackend(migration_config, request_sync_group, state_manager)
    else:
        target_migration_backend = KvTransferMigrationBackend(rank, instance_id, worker_id, migration_config,
                                                              request_sync_group, base_worker, serving_args,
                                                              state_manager)

    return target_migration_backend
