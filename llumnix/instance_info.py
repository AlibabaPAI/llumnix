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

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Tuple

from llumnix.logging.logger import init_logger
from llumnix.load_computation import (
    LoadCalculatorFactory,
    BaseLoad,
    DummyLoad,
    KvBlocksRatioLoad,
    AdaptiveDecodeBatchLoad,
    RemainingStepsLoad,
    MissWaitingTokensLoad,
)
from llumnix.llumlet.request import RequestInferenceType
from llumnix.arg_utils import InstanceArgs

logger = init_logger(__name__)


class InstanceType(str, Enum):
    NO_CONSTRAINTS = "no_constraints"
    PREFILL = "prefill"
    DECODE = "decode"
    PREFILL_AS_DECODE = "prefill_as_decode"
    DECODE_AS_PREFILL = "decode_as_prefill"
    

INSTANCE_TYPE_TO_METRIC_FIELD: Dict[InstanceType, str] = {
    InstanceType.NO_CONSTRAINTS: 'dispatch_load_metric',
    InstanceType.PREFILL: 'dispatch_prefill_load_metric',
    InstanceType.DECODE: 'dispatch_decode_load_metric',
    InstanceType.PREFILL_AS_DECODE: 'dispatch_prefill_as_decode_load_metric',
    InstanceType.DECODE_AS_PREFILL: 'dispatch_decode_as_prefill_load_metric'
}


@dataclass
class InstanceInfo:
    instance_id: str = ""
    instance_type: InstanceType = None

    step_id: int = None
    timestamp: float = None
    num_batched_tokens: int = None
    num_seqs: int = None
    running_seq_lens: List[int] = field(default_factory=list)
    last_inference_latency: float = None
    inference_type: RequestInferenceType = None
    decode_batch_size: int = 0

    num_total_gpu_blocks: int = 0
    num_watermark_blocks: int = 0
    num_used_gpu_blocks: int = 0
    num_free_gpu_blocks: int = 0
    gpu_cache_usage: float = 0.0
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    num_killed_requests: int = 0
    num_blocks_first_waiting_request: int = 0
    waiting_time_first_waiting_request: int = 0
    num_blocks_all_waiting_requests: int = 0
    num_blocks_last_running_request: int = 0
    num_miss_tokens_all_waiting_requests: int = 0

    # cache state
    num_cached_request_ids: int = 0
    num_wait_update_request_ids: int = 0
    num_trans_wrapper_cached_request: int = 0

    # load metrics
    kv_blocks_ratio: KvBlocksRatioLoad
    remaining_steps: RemainingStepsLoad
    adaptive_decode: AdaptiveDecodeBatchLoad
    miss_waiting_tokens: MissWaitingTokensLoad

    # on-demand init infos
    migration_load_metric: BaseLoad = DummyLoad()
    migration_load_metric_after_migrate_in: BaseLoad = DummyLoad()
    migration_load_metric_after_migrate_out: BaseLoad = DummyLoad()
    has_migration_slot: bool = True
    is_migrating: bool = False

    # lazy init infos
    profiling_data: Tuple[str, int, int, float] = None

    # instance level info for load computation
    enable_defrag: bool = False

    def __post_init__(self) -> None:
        self.num_available_gpu_blocks = self.num_total_gpu_blocks - self.num_used_gpu_blocks
        self.num_available_gpu_blocks_waiting = self.num_available_gpu_blocks - self.num_blocks_all_waiting_requests

    def __hash__(self):
        return hash(self.instance_id)

    def __repr__(self):
        return f"InstanceInfo(instance_id={self.instance_id}, instance_type={self.instance_type})"

def sort_instance_infos(available_instance_infos: Iterable[InstanceInfo],
                        key_attr: str,
                        descending: bool = False) -> List[InstanceInfo]:
    return sorted(
        available_instance_infos,
        key=lambda instance_info: (
            getattr(instance_info, key_attr),
            instance_info.instance_id
        ),
        reverse=descending
    )


class InstanceLoadCalculator:
    def __init__(self, instance_args: InstanceArgs) -> None:
        self.migration_load_calculator = LoadCalculatorFactory.get_load_calculator(instance_args.migration_load_metric)
            
        self.adaptive_decode_calculator = LoadCalculatorFactory.get_load_calculator('adaptive_decode')
        self.kv_blocks_ratio_calculator = LoadCalculatorFactory.get_load_calculator('kv_blocks_ratio')
        self.remaining_steps_calculator = LoadCalculatorFactory.get_load_calculator('remaining_steps')
        self.miss_waiting_tokens_calculator = LoadCalculatorFactory.get_load_calculator('miss_waiting_tokens')

    def compute_instance_load(self, instance_info: InstanceInfo):
        instance_info.migration_load_metric = self.migration_load_calculator.compute_instance_load(instance_info)

        instance_info.migration_load_metric_after_migrate_out = \
            self._compute_load_after_migrate(instance_info, is_migrate_in=False)
        instance_info.migration_load_metric_after_migrate_in = \
            self._compute_load_after_migrate(instance_info, is_migrate_in=True)
        
        instance_info.adaptive_decode = self.adaptive_decode_calculator.compute_instance_load(instance_info)
        instance_info.kv_blocks_ratio = self.kv_blocks_ratio_calculator.compute_instance_load(instance_info)
        instance_info.remaining_steps = self.remaining_steps_calculator.compute_instance_load(instance_info)
        instance_info.miss_waiting_tokens = self.miss_waiting_tokens_calculator.compute_instance_load(instance_info)

    def _compute_load_after_migrate(self, instance_info: InstanceInfo, is_migrate_in: bool) -> BaseLoad:
        instance_info_after_migrate = copy.deepcopy(instance_info)
        num_blocks_last_running_request = instance_info_after_migrate.num_blocks_last_running_request

        if is_migrate_in:
            instance_info_after_migrate.num_running_requests += 1
            instance_info_after_migrate.num_available_gpu_blocks -= num_blocks_last_running_request
        else:
            instance_info_after_migrate.num_running_requests -= 1
            instance_info_after_migrate.num_available_gpu_blocks += num_blocks_last_running_request

        return self.migration_load_calculator.compute_instance_load(instance_info_after_migrate)


class ScalingLoadComputation:
    def __init__(self, load_metric):
        self.load_calculator = LoadCalculatorFactory.get_load_calculator(load_metric)

    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        return self.load_calculator.compute_instance_load(instance_info)
