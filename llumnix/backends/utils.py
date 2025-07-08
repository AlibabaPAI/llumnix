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

from typing import Any
import os
from enum import Enum

import ray
from ray.util.placement_group import PlacementGroup

from llumnix.arg_utils import LlumnixEngineArgs, InstanceArgs
from llumnix.backends.backend_interface import BackendInterface
from llumnix.queue.queue_type import QueueType
from llumnix.logging.logger import init_logger
from llumnix.utils import BackendType

logger = init_logger(__name__)

class EngineState(str, Enum):
    INIT = "INIT"
    CRASHED = "CRASHED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


# Once worker died, proxy actor will not restart.
@ray.remote(num_cpus=0, max_concurrency=2, max_restarts=-1)
class ProxyActor:
    def __init__(self, is_driver_worker: bool, use_ray_spmd_worker: bool):
        self.is_driver_worker = is_driver_worker
        self.use_ray_spmd_worker = use_ray_spmd_worker

    def exec_method(self, handle: ray.actor.ActorHandle, *args, **kwargs) -> Any:
        if self.is_driver_worker and not self.use_ray_spmd_worker:
            ret = ray_get_with_timeout(
                handle.execute_engine_method_async.remote(
                    "execute_driver_worker_method_async", *args, **kwargs
                )
            )
        else:
            ret = ray_get_with_timeout(
                handle.execute_method.remote(*args, **kwargs)
            )

        return ret


def init_backend_engine(instance_id: str,
                        placement_group: PlacementGroup,
                        request_output_queue_type: QueueType,
                        instance_args: InstanceArgs,
                        llumnix_engine_args: LlumnixEngineArgs) -> BackendInterface:
    backend_type = llumnix_engine_args.backend_type
    if backend_type == BackendType.VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.llm_engine import BackendVLLM
        backend_engine = BackendVLLM(instance_id,
                                     placement_group,
                                     request_output_queue_type,
                                     instance_args,
                                     llumnix_engine_args)
    elif backend_type == BackendType.VLLM_V1:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm_v1.core import BackendVLLMV1
        backend_engine = BackendVLLMV1(instance_id,
                                     placement_group,
                                     request_output_queue_type,
                                     instance_args,
                                     llumnix_engine_args)
    elif backend_type == BackendType.BLADELLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.bladellm.llm_engine import BackendBladeLLM
        backend_engine = BackendBladeLLM(instance_id,
                                         placement_group,
                                         request_output_queue_type,
                                         instance_args,
                                         llumnix_engine_args)
    elif backend_type == BackendType.SIM_VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.sim_llm_engine import BackendSimVLLM
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        backend_engine = BackendSimVLLM(instance_id,
                                        placement_group,
                                        request_output_queue_type,
                                        instance_args,
                                        llumnix_engine_args)
    else:
        raise ValueError(f'Unsupported backend: {backend_type}')
    return backend_engine
