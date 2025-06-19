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
from collections import defaultdict
from itertools import islice, repeat
from typing import Callable, Dict, List, Optional, Tuple, Type, Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# pylint: disable=unused-import
from ray.util.placement_group import PlacementGroup

from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, make_async ,envs)
from vllm.worker.worker_base import WorkerBase
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest
from vllm.model_executor.layers.sampler import SamplerOutput

from llumnix.internal_config import MigrationConfig
from llumnix.logging.logger import init_logger
from llumnix.utils import random_uuid, ray_get_with_timeout
from llumnix.constants import NUM_GPUS_VLLM_GPU_ACTOR
import llumnix.envs as llumnix_envs

logger = init_logger(__name__)


class LlumnixRayDistributedExecutor(RayDistributedExecutor):
    instance_id: str = None
    migration_config: MigrationConfig = None
    last_inference_latency: int = 0