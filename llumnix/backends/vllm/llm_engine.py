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
from typing import Any, List, Optional, Dict, Union, Iterable, Tuple
from collections import defaultdict
import threading
import ray
from ray.util.placement_group import PlacementGroup

from vllm.engine.llm_engine import LLMEngine
from vllm.core.scheduler import ScheduledSequenceGroup
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup, SequenceStatus, SamplerOutput, SequenceGroupMetadata
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import Counter
from vllm.usage.usage_lib import UsageContext

from llumnix.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface
from llumnix.backends.vllm.scheduler import SchedulerLlumnix
from llumnix.backends.vllm.sequence import SequenceGroupLlumnix
from llumnix.backends.profiling import LatencyMemData
from llumnix.server_info import ServerInfo
from llumnix.config import MigrationConfig
from llumnix.rpc.queue_client import QueueClient


logger = init_logger(__name__)

@ray.remote(num_cpus=0)
class AsyncActor:
    def __init__(self, instance_id):
        logger.info("Before create QueueClient")
        self.request_output_queue_client = QueueClient()
        logger.info("After create QueueClient")
        self.instance_id = instance_id
        self.engine_actor_handle = None

    async def put_nowait_batch_to_server(self, 
                                         server_request_outputs: Dict[str, List[RequestOutput]], 
                                         server_info_dict: Dict[str, ServerInfo]) -> None:
        if self.engine_actor_handle is None:
            self.engine_actor_handle = ray.get_actor("instance_{}".format(self.instance_id), namespace="llumnix")
        for server_id, req_outputs in server_request_outputs.items():
            try:
                server_info = server_info_dict[server_id]
                await self.request_output_queue_client.put_nowait_batch(req_outputs, server_info)
            except TimeoutError:
                logger.info("Server {} is dead".format(server_id))
                request_ids = [req_output.request_id for req_output in req_outputs]
                self.engine_actor_handle.abort_request.remote(request_ids)

class LLMEngineLlumnix(LLMEngine):
    def __init__(self, instance_id: str, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        self.async_actor = AsyncActor.remote(instance_id)
        self.request_server_info: Dict[str, ServerInfo] = {}

    # pylint: disable=W0221
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        migration_config: MigrationConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        instance_id: str = None,
        placement_group: Optional["PlacementGroup"] = None,
        node_id: str = None,
        latency_mem: Optional[LatencyMemData] = None
    ) -> "LLMEngineLlumnix":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        engine_config.parallel_config.placement_group = placement_group
        # Initialize the cluster and specify the executor class.
        # pylint: disable=import-outside-toplevel
        if latency_mem is not None:
            from llumnix.backends.vllm.executor import SimGPUExecutor
            executor_class = SimGPUExecutor
            executor_class.latency_mem = latency_mem
        elif engine_config.parallel_config.worker_use_ray:
            from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor
            executor_class = LlumnixRayGPUExecutor
            executor_class.migration_config = migration_config
        else:
            raise ValueError('Unsupported executor backend')
        # Hack to pass node_id to _init_workers_ray function.
        executor_class.node_id = node_id
        # Create the LLM engine.
        engine = cls(
            instance_id=instance_id,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )
        return engine

    def _process_model_outputs(
        self,
        output: List[SamplerOutput],
        scheduled_seq_groups: List[ScheduledSequenceGroup],
        ignored_seq_groups: List[SequenceGroup],
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[RequestOutput], List[ServerInfo]]:
        # ensure scheduled_seq_groups matching output
        with self.scheduler.scheduler_lock:
            server_info_list = []
            if output:
                new_output = []
                new_scheduled_seq_groups = []
                new_seq_group_metadata_list = []
                for scheduled_seq_group, seq_group_meta, seq_group_output in zip(scheduled_seq_groups, seq_group_metadata_list, output[0].outputs):
                    seq_group = scheduled_seq_group.seq_group
                    if seq_group.get_seqs(SequenceStatus.RUNNING):
                        new_scheduled_seq_groups.append(scheduled_seq_group)
                        new_seq_group_metadata_list.append(seq_group_meta)
                        new_output.append(seq_group_output)
                        server_info_list.append(seq_group.server_info)
                scheduled_seq_groups = new_scheduled_seq_groups
                output[0].outputs = new_output
                seq_group_metadata_list = new_seq_group_metadata_list
            request_outputs = super()._process_model_outputs(output, scheduled_seq_groups, ignored_seq_groups, seq_group_metadata_list)
            # TODO(ZeldaHuang) Use LlumnixRequestOutput to store llumnix output args.
            return request_outputs, server_info_list

    def step(self) -> None:
        output_list, server_info_list = super().step()

        instance_info: InstanceInfo = self.instance_info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.latency = self.model_executor.last_inference_latency
        seq_groups = self.scheduler.running
        if seq_groups:
            tot_blocks = []
            for seq in seq_groups[-1].get_seqs(SequenceStatus.RUNNING):
                blocks = self.scheduler.block_manager.get_block_table(seq)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_blocks_last_running_request = len(tot_blocks)

        if output_list:
            self._put_request_outputs_to_server(output_list, server_info_list)
        self.instance_info = instance_info

    def update_instance_info(self, instance_info: InstanceInfo) -> None:
        # These fields are updated after step.
        if self.instance_info is not None:
            instance_info.instance_id = self.instance_info.instance_id
            instance_info.step_id = self.instance_info.step_id
            instance_info.timestamp = self.instance_info.timestamp
            instance_info.latency = self.instance_info.latency
            instance_info.num_blocks_last_running_request = self.instance_info.num_blocks_last_running_request
        self.instance_info = instance_info

    def add_request(self, request_id: str, server_info: ServerInfo, *args, **kwargs):
        super().add_request(request_id, *args, **kwargs)
        seq_group = self.scheduler.waiting[-1]
        self.scheduler.waiting[-1] = SequenceGroupLlumnix(request_id, server_info, [seq_group.get_seqs()[0]], seq_group.sampling_params,
                                  seq_group.metrics.arrival_time, seq_group.lora_request, seq_group.multi_modal_data)

    def _put_request_outputs_to_server(self, request_outputs, server_infos: List[ServerInfo]) -> None:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}
        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output, server_info in zip(request_outputs, server_infos):
            server_id = server_info.server_id
            server_request_outputs[server_id].append(request_output)
            if server_id not in server_info_dict:
                server_info_dict[server_id] = server_info
        self.async_actor.put_nowait_batch_to_server.remote(server_request_outputs, server_info_dict)

class BackendVLLM(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        migration_config: MigrationConfig,
        engine_args: EngineArgs,
        placement_group: "PlacementGroup" = None,
        node_id: str = None
    ) -> None:
        self.engine: LLMEngineLlumnix = LLMEngineLlumnix.from_engine_args(engine_args=engine_args,
                                                                          migration_config=migration_config,
                                                                          instance_id=instance_id,
                                                                          placement_group=placement_group,
                                                                          node_id=node_id)
        # multi-instance args
        self.engine.scheduler = SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config)
        self.engine.scheduler.add_update_instance_info_callback(self.engine.update_instance_info)
        self.engine.output_processor.scheduler = self.engine.scheduler
        self.instance_id = instance_id
        self.worker_handle_list = self.engine.model_executor.workers.copy()
        if len(self.worker_handle_list) + 1 == self.engine.parallel_config.world_size:
            self.worker_handle_list.insert(0, ray.get_actor(f"instance_{self.instance_id}", namespace="llumnix"))
        self._run_workers("init_migration", instance_id=instance_id, migration_config=migration_config,\
                                                      src_worker_handle_list=self.worker_handle_list,
                                                      placement_group=placement_group, node_id=node_id)
        self._thread = threading.Thread(
            target=self._start_engine_loop, args=(), daemon=True, name="engine_loop"
        )
        self._thread.start()

    def _start_engine_loop(self) -> None:
        while True:
            self.engine.step()

    def execute_worker_method(self, method, *args, **kwargs):
        return self.engine.model_executor.driver_worker.execute_method(method, *args, **kwargs)

    def add_request(self,
                    request_id: str,
                    server_info: ServerInfo,
                    *args,
                    **kwargs) -> None:
        # Store the server information of each request to put the request outputs back to the corresponding api server correctly.
        self.engine.request_server_info[request_id] = server_info
        self.engine.add_request(request_id, *args, **kwargs)

    def commit_dst_request(self, backend_request: SequenceGroupLlumnix) -> None:
        seq = backend_request.get_seqs()[0]
        seq.seq_id = next(self.engine.seq_counter)
        logger.info("add seq {} to block table".format(seq.seq_id))
        pre_alloc_blocks = self.engine.scheduler.pre_alloc_cache_dict.pop(backend_request.request_id)
        self.engine.scheduler.block_manager.add_block_table(pre_alloc_blocks, seq.seq_id)
        self.add_running_request(backend_request)
        backend_request.reset_migration_args()

    def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]) -> None:
        ray.get(dst_ray_actor.execute_engine_method.remote("_run_workers",
                                                           "migrate_cache",
                                                           dst_blocks=dst_blocks,
                                                           src_blocks=src_blocks,
                                                           src_worker_handle_list=self.worker_handle_list))

    def _run_workers(self, *args, **kwargs):
        # pylint: disable=protected-access
        return self.engine.model_executor._run_workers(*args, **kwargs)

    def is_ready(self):
        return True

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        return self.engine.abort_request(request_ids)

    def get_running_queue(self ) -> List[SequenceGroupLlumnix]:
        return self.engine.scheduler.get_running_queue()

    def get_request_incremental_blocks(self, *args, **kwargs) -> List[int]:
        return self.engine.scheduler.get_request_incremental_blocks(*args, **kwargs)

    def remove_running_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler.remove_running_request(*args, **kwargs)

    def add_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler.add_migrating_out_request_last_stage(*args, **kwargs)

    def remove_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler.remove_migrating_out_request_last_stage(*args, **kwargs)

    def pop_migrating_out_requests_last_stage(self, *args, **kwargs) -> List[Any]:
        return self.engine.scheduler.pop_migrating_out_requests_last_stage(*args, **kwargs)

    def pre_alloc(self, *args, **kwargs) -> List[int]:
        return self.engine.scheduler.pre_alloc(*args, **kwargs)

    def should_abort_migration(self, *args, **kwargs) -> bool:
        return self.engine.scheduler.should_abort_migration(*args, **kwargs)

    def add_running_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler.add_running_request(*args, **kwargs)

    def is_request_running(self, *args, **kwargs) -> bool:
        return self.engine.scheduler.is_request_running(*args, **kwargs)

    def free_dst_pre_alloc_cache(self, *args, **kwargs) -> None:
        return self.engine.scheduler.free_dst_pre_alloc_cache(*args, **kwargs)

    def free_src_request(self, backend_request: SequenceGroup) -> None:
        return self.engine.scheduler.free_src_request(backend_request)

    def get_all_request_ids(self) -> List[str]:
        return self.engine.scheduler.get_all_request_ids()
