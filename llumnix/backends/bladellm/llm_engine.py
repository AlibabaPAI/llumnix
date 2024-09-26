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
import asyncio
import ray
from ray.util.placement_group import PlacementGroup

from blade_llm.service.engine import AsyncLLMEngine
from blade_llm.service.args import ServingArgs
from blade_llm.utils.counter import Counter

from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface
from llumnix.backends.bladellm.scheduler import SchedulerLlumnix
from llumnix.backends.bladellm.sequence import SequenceGroupLlumnix
from llumnix.llumlet.request import LlumnixRequest
from llumnix.internal_config import MigrationConfig
from llumnix.server_info import ServerInfo
from llumnix.rpc.queue_client import QueueClient

logger = init_logger(__name__)

class AsyncPutQueueThread(threading.Thread):
    def __init__(self, instance_id):
        super().__init__()
        self.instance_id = instance_id
        self.request_output_queue_client = QueueClient()
        self.engine_actor_handle = None
        self.loop = asyncio.new_event_loop()
        self.daemon = True

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

class LLMEngineLlumnix(AsyncLLMEngine):
    def __init__(self, instance_id: str, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        # TODO(s5u13b): Reduce the overhead.
        self.async_put_queue_thread = AsyncPutQueueThread(instance_id)
        self.async_put_queue_thread.start()
    
    async def _init(self):
        self._worker_processes.start()

        self._stop_event = asyncio.Event()
        self._req_buffer = asyncio.Queue()
        client_args = (
            (self._args, self._worker_processes.worker_addrs)
            if self._args.enable_remote_worker
            else (self._args, [], self._inst_id)
        )
        logger.info("client_args {}".format(client_args))
        # pylint: disable=protected-access
        self._workers = LlumnixAioWorkerClient(*client_args)
        await self._workers.wait_backend_ready()
        token_capacity, block_size, model_max_len, cpu_blocks = (
            await self._workers.estimate_token_capacity()
        )
        logger.info(
            "Workers estimate token capacity to: {}, cpu_blocks: {}, block_size: {}".format(
                token_capacity,
                cpu_blocks,
                block_size,
            )
        )
        self._scheduler = SchedulerLlumnix(
            self._args,
            SchedulerInitInfo(
                token_capacity=token_capacity,
                block_size=block_size,
                model_max_len=model_max_len,
                cpu_blocks=cpu_blocks,
            ),
        )

        # profiling-related fields
        self._sch_status_helper = scheduler_status_helper(self._scheduler)
        # this will take control of exporters if step-wise tracing will trace from the beginning
        self.engine_pre_step_metrics()
    
    def add_request(self, request_id: str, server_info: ServerInfo, *args, **kwargs):
        super().add_request(*args, **kwargs)
        seq_group = self.scheduler.waiting[-1]
        self.scheduler.waiting[-1] = SequenceGroupLlumnix(request_id, server_info, seq_group.length, seq_group.is_streaming, seq_group.num_generation,
                                                          seq_group.best_of, seq_group.num_finished, seq_group.finished_gen, seq_group.paged_reqs,
                                                          seq_group.receive_time, seq_group.join_time, seq_group.prompt_len_priority_scale, seq_group.last_swap_time,
                                                          seq_group.lora_path)
        self.scheduler.scheduler_lock.release()
    
    async def step(self):
        await super().step()

        instance_info: InstanceInfo = self.instance_info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        # TODO(xinyi)
        # instance_info.latency = self.model_executor.last_inference_latency

        seq_groups = self.scheduler.running
        if seq_groups:
            tot_blocks = []
            for seq in seq_groups[-1].paged_reqs:
                blocks = self.scheduler.block_manager.get_block_table(seq)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_blocks_last_running_request = len(tot_blocks)
        
        self.instance_info=instance_info


class BackendBladeLLM(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        migration_config: MigrationConfig,
        engine_args: ServingArgs,
        placement_group: PlacementGroup = None,
        node_id: str = None
    ) -> None:
        self.engine: LLMEngineLlumnix = LLMEngineLlumnix(instance_id, engine_args)
        self.instance_id = instance_id
        loop = asyncio.get_event_loop()
        self.engine.start(loop)
        # self.engine.scheduler = SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config)
    
    def add_request(self,
                    *args,
                    **kwargs) -> None:
        # Store the server information of each request to put the request outputs back to the corresponding api server correctly.
        req = args[0]
        self.engine.add_request(req)

    def send_blocks(self, dst_ray_actor: "ray.actor.ActorHandle", src_blocks: List[int], dst_blocks: List[int]) -> None:
        ray.get(dst_ray_actor.execute_engine_method.remote("_run_workers",
                                                           "migrate_cache",
                                                           dst_blocks=dst_blocks,
                                                           src_blocks=src_blocks,
                                                           src_worker_handle_list=self.worker_handle_list))

    def is_ready(self):
        return True
    
    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for req_id in request_ids:
            self.engine.drop_request(req_id) # TODO
    
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

    def free_src_request(self, backend_request: LlumnixRequest) -> None:
        return self.engine.scheduler.free_src_request(backend_request)

    def get_all_request_ids(self) -> List[str]:
        return self.engine.scheduler.get_all_request_ids()
    
    