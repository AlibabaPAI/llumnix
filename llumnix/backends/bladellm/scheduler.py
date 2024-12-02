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

from asyncio.log import logger
import time
from typing import Dict, List, Set, Union

from blade_llm.service.paged_utils import PreemptionMode
from blade_llm.service.block_space_manager import BlockSpaceManager
from blade_llm.service.proto.bladellm_pb2 import BlockTable, GenerationGroup, WorkerStepRequest
from blade_llm.service.schedulers import PagedScheduler
from blade_llm.service.scheduler_types import SchedulerStepOutput, GenerationGroupState
from blade_llm.service.paged_utils import PreemptionMode
from blade_llm.service.args import ServingArgs
from blade_llm.service.scheduler_types import GenerationGroupState
from blade_llm.protocol import GenerateStreamResponse

from llumnix.instance_info import InstanceInfo
from llumnix.logger import init_logger
from llumnix.llumlet.request import RequestInferenceType
from llumnix.backends.bladellm.sequence import GenerationGroupStateLlumnix
from llumnix.backends.bladellm.sequence import GenerateStreamResponseLlumnix
from loguru import logger as my_logger

logger = init_logger(__name__)

class BlockManagerLlumnix(BlockSpaceManager):
    def get_free_blocks(self, num_required_blocks: int) -> BlockTable:
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        if num_free_gpu_blocks - num_required_blocks < self.reserved_blocks:
            return []
        blocks = []
        for _ in range(num_required_blocks):
            block = self.gpu_allocator.allocate()
            blocks.append(block)

        return blocks

    def add_block_table(self, block_table: BlockTable, block_table_id: int) -> None:
        self.block_tables[block_table_id] = block_table.copy()

class PagedSchedulerLlumnix(PagedScheduler):
    def __init__(self, serving_args: ServingArgs, *args, **kwargs) -> None:
        super().__init__(serving_args, *args, **kwargs)
        self.block_manager = BlockManagerLlumnix(
            block_size=self.block_size,
            num_gpu_blocks=self._max_processing_units,
            num_cpu_blocks=self.cpu_blocks,
            block_reserved_percent=self.block_reserved_percent,
            gamma=self.gamma_blank, # gamma step for speculative decoding, lookahead etc
            disable_prompt_cache=serving_args.disable_prompt_cache,
            prompt_cache_enable_swap=serving_args.prompt_cache_enable_swap,
        )
        self.pre_alloc_cache_dict: Dict[int, BlockTable] = {}
        self.migrating_out_request_last_stage: List[GenerationGroupStateLlumnix] = []

    def _handle_single_response(
        self,
        gen_group: GenerationGroup,
        req_group,
        client_resp: Dict[int, Union[GenerateStreamResponse, GenerateStreamResponseLlumnix]],
        running_request_groups_map: Dict[int, GenerationGroupStateLlumnix],
    ):
        my_logger.info(gen_group)
        my_logger.info(running_request_groups_map)

        gen_group_id = gen_group['request_group_id']
        request_id, server_info = "", None
        if gen_group_id in running_request_groups_map:
            request_id = running_request_groups_map[gen_group_id].request_id
            server_info = running_request_groups_map[gen_group_id].server_info

        super()._handle_single_response(gen_group, req_group, client_resp, running_request_groups_map)

        for generation in gen_group['generations']:
            client_resp[generation['request_id']] = GenerateStreamResponseLlumnix(
                request_id=request_id,
                server_info=server_info,
                resp=client_resp[generation['request_id']],
            )

        my_logger.info(client_resp)

    def add_update_instance_info_callback(self, update_instance_info_callback):
        self.update_instance_info_callback = update_instance_info_callback
        self.update_instance_info_callback(self._get_instance_info([]))
    
    def _preempt(self, seq_group: GenerationGroupState, insert=True) -> PreemptionMode:
        seq_group.last_preemption_time = time.time()
        return super()._preempt(seq_group, insert)
    
    def _get_num_killed_requests(self) -> int:
        cnt = len(self.swapped)
        for seq_group in self.waiting:
            if seq_group.last_preemption_time:
                cnt += 1
        return cnt

    def get_running_queue(self):
        return self.running

    def get_all_request_ids(self) -> List[str]:
        request_ids : List[str] = []
        for state_queue in [self.waiting, self.running, self.swapped, self.hanging]:
            for seq_group in state_queue:
                request_ids.append(seq_group.request_group_id)
        return request_ids
    
    def get_request_incremental_blocks(self, backend_request: GenerationGroupStateLlumnix, pre_stage_num_blocks: int) -> List[int]:
        seq = backend_request.paged_reqs[0]
        blocks = self.block_manager.get_block_table(seq)
        return blocks[pre_stage_num_blocks:]

    def remove_running_request(self, request_id: int) -> None:
        for seq_group in self.running:
            if seq_group.request_group_id == request_id:
                self.running.remove(seq_group)
                break

    def add_migrating_out_request_last_stage(self, backend_request: GenerationGroupStateLlumnix) -> None:
        self.migrating_out_request_last_stage.append(backend_request)

    def remove_migrating_out_request_last_stage(self, backend_request: GenerationGroupStateLlumnix) -> None:
        self.migrating_out_request_last_stage.remove(backend_request)

    def pop_migrating_out_requests_last_stage(self) -> List[GenerationGroupStateLlumnix]:
        migrating_out_request_last_stage = self.migrating_out_request_last_stage.copy()
        self.migrating_out_request_last_stage.clear()
        return migrating_out_request_last_stage

    def pre_alloc(self, request_id: int, block_num: int) -> List[int]:
        blocks = self.block_manager.get_free_blocks(block_num)
        pre_blocks = self.pre_alloc_cache_dict.get(request_id, [])
        pre_blocks.extend(blocks)
        self.pre_alloc_cache_dict[request_id] = pre_blocks
        blocks = [block.block_number for block in blocks]
        return blocks

    def add_running_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        self.running.append(backend_request)

    def is_request_running(self, backend_request: GenerationGroupStateLlumnix) -> bool:
        return backend_request in self.running

    def free_dst_pre_alloc_cache(self, request_id: str = None) -> None:
        if request_id:
            blocks = self.pre_alloc_cache_dict.pop(request_id, [])
            # pylint: disable=protected-access
            self.block_manager._free_block_table(blocks)
        else:
            # Clear all pre-allocated cache of dst instance when src instance encounters exception.
            request_ids = list(self.pre_alloc_cache_dict.keys())
            for req_id in request_ids:
                blocks = self.pre_alloc_cache_dict.pop(req_id, [])
                # pylint: disable=protected-access
                self.block_manager._free_block_table(blocks)

    def free_src_request(self, backend_request: GenerationGroupStateLlumnix) -> None:
        seq = backend_request.paged_reqs[0]
        logger.info("free seq {}".format(seq.request_id))
        self._free_req(backend_request)

    def _get_instance_info(self, steps: List[WorkerStepRequest]) -> InstanceInfo:
        num_total_gpu_blocks = self._max_processing_units
        num_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
        num_used_gpu_blocks = num_total_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / num_total_gpu_blocks
        if self.waiting:
            num_blocks_waiting_requests = []
            waiting_time_waiting_requests = []
            for seq_group in self.waiting:
                num_prompt_tokens = len(seq_group.paged_reqs[0].token_ids)
                num_blocks = num_prompt_tokens / self.block_size
                waiting_time = time.monotonic() - seq_group.receive_time
                num_blocks_waiting_requests.append(num_blocks)
                waiting_time_waiting_requests.append(waiting_time)
            num_blocks_first_waiting_request = num_blocks_waiting_requests[0]
            waiting_time_first_waiting_request = waiting_time_waiting_requests[0]
            num_blocks_all_waiting_requests = sum(num_blocks_waiting_requests)
        else:
            num_blocks_first_waiting_request = 0
            waiting_time_first_waiting_request = 0
            num_blocks_all_waiting_requests = 0
        instance_info = InstanceInfo(
            num_total_gpu_blocks=num_total_gpu_blocks,
            num_watermark_blocks=self.block_manager.reserved_blocks,
            num_used_gpu_blocks=num_used_gpu_blocks,
            num_free_gpu_blocks=num_free_gpu_blocks,
            gpu_cache_usage=gpu_cache_usage,
            num_running_requests=len(self.running),
            num_waiting_requests=len(self.waiting),
            num_killed_requests=self._get_num_killed_requests(),
            num_blocks_first_waiting_request=num_blocks_first_waiting_request,
            waiting_time_first_waiting_request=waiting_time_first_waiting_request,
            num_blocks_all_waiting_requests=num_blocks_all_waiting_requests,
        )

        for gen_group in self.running:
            instance_info.running_seq_lens.extend([len(req_state.token_ids) for req_state in gen_group.paged_reqs])
            instance_info.num_seqs = len(instance_info.running_seq_lens)

        instance_info.inference_type = RequestInferenceType.generate_inference_type(
            exist_prefill=any(len(step.prefill) > 0 for step in steps),
            exist_decode=any(len(step.decode) > 0 for step in steps))
        instance_info.num_batched_tokens = sum([
            len(step.decode) + sum([len(prefill.prompt_tokens) for prefill in step.prefill]) for step in steps
        ])
        instance_info.finished_request_ids = len(self._finished_req_to_remove)
        logger.info("update in scheduler {}".format(instance_info.num_running_requests))
        return instance_info
        
    def step(self) -> SchedulerStepOutput:
        step_out = super().step()
        my_logger.info(step_out)

        if step_out.step:
            request_groups_map = self.get_request_groups_map()
            for single_step in step_out.step:
                step_ids = list(single_step.decode) + [r.id for r in single_step.prefill]
                for r_id in step_ids:
                    num_new_tokens = request_groups_map[r_id].get_num_new_tokens() 
                    request_groups_map[r_id].token_chunk_size = num_new_tokens
            self.update_instance_info_callback(self._get_instance_info(step_out.step))
        return step_out

    def add_gen_group(self, gen_group, server_req):
        gen_group_llumnix = GenerationGroupStateLlumnix(
            gen_group, server_req.request_id, server_req.server_info, server_req.expected_steps)
        if hasattr(gen_group_llumnix.server_info, 'request_timestamps'):
            gen_group_llumnix.server_info.request_timestamps.engine_add_request_timestamp = time.time()
        super().add_gen_group(gen_group_llumnix, server_req)

    def get_request_groups_map(self) -> Dict[str, GenerationGroupStateLlumnix]:
        request_groups_map = {}
        for state_queue in [self.waiting, self.running, self.swapped, self.hanging]:
            for gen_group in state_queue:
                request_groups_map[gen_group.request_group_id] = gen_group 
        return request_groups_map
    
    def _preempt_by_recompute(self, gen_group: GenerationGroupStateLlumnix, *args, **kwargs):
        super()._preempt_by_recompute(gen_group, *args, **kwargs)
        gen_group._num_computed_tokens = 0
        gen_group.is_prefill = True
