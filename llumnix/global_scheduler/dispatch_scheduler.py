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

from typing import Dict

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceType, InstanceInfo
from llumnix.constants import DISPATCH_LOG_FREQUENCY
from llumnix.global_scheduler.dispatch_policy import DispatchPolicyFactory, DispatchPolicy
from llumnix.utils import RequestIDType

logger = init_logger(__name__)


class DispatchScheduler:
    def __init__(self,
                 dispatch_policy: str,
                 topk_random_dispatch: int,
                 enable_pd_disagg: bool,
                 enable_engine_pd_disagg: bool,
                 enable_dynamic_pd_disagg: bool,
                 dispatch_load_metric: str,
                 dispatch_prefill_load_metric: str,
                 dispatch_decode_load_metric: str,
                 dispatch_prefill_as_decode_load_metric: str,
                 dispatch_decode_as_prefill_load_metric: str) -> None:
        self.enable_pd_disagg = enable_pd_disagg
        self.enable_engine_pd_disagg = enable_engine_pd_disagg
        self.enable_dynamic_pd_disagg = enable_dynamic_pd_disagg
        self.dispatch_policy: DispatchPolicy = DispatchPolicyFactory.get_policy(
            dispatch_policy,
            topk_random_dispatch=topk_random_dispatch,
            dispatch_load_metric=dispatch_load_metric,
            dispatch_prefill_load_metric=dispatch_prefill_load_metric,
            dispatch_decode_load_metric=dispatch_decode_load_metric,
            dispatch_prefill_as_decode_load_metric=dispatch_prefill_as_decode_load_metric,
            dispatch_decode_as_prefill_load_metric=dispatch_decode_as_prefill_load_metric)

        # statistics
        self.instance_type_num_requests: Dict[str, int] = {}

    def dispatch(self,
                 request_id: RequestIDType,
                 instance_info: Dict[str, InstanceInfo],
                 instance_num_requests: Dict[str, int],
                 prefill_instance_info: Dict[str, InstanceInfo],
                 prefill_instance_num_requests: Dict[str, int],
                 decode_instance_info: Dict[str, InstanceInfo],
                 decode_instance_num_requests: Dict[str, int]):
        if not self.enable_dynamic_pd_disagg:
            no_constrains_instance_id = self.general_dispatch()
            instance_num_requests[no_constrains_instance_id] += 1
            prefill_instance_id, decode_instance_id = None, None
        else:
            instance_id = self.general_dispatch()
        logger.info("dispath request {} to instance {}.".format(request_id, instance_id))
        return no_constrains_instance_id, prefill_instance_id, decode_instance_id

    def general_dispatch(self,
                 request_id: RequestIDType,
                 instance_num_requests: Dict[str, int],
                 prefill_instance_info: Dict[str, InstanceInfo],
                 prefill_instance_num_requests: Dict[str, int],
                 decode_instance_info: Dict[str, InstanceInfo],
                 decode_instance_num_requests: Dict[str, int]):
        instance_type = InstanceType.NO_CONSTRAINTS
        if self.enable_pd_disagg or self.enable_engine_pd_disagg:
            instance_type = InstanceType.PREFILL

        instance_id = self.dispatch_policy.dispatch(
            instance_type=instance_type,
            instance_info=prefill_instance_info,
            instance_num_requests=prefill_instance_num_requests
        )

        if instance_type == InstanceType.NO_CONSTRAINTS or self.global_scheduler_config.enable_pd_disagg:
            logger.info("dispath request {} to instance {}.".format(request_id, instance_id))
            return instance_id

        if self.global_scheduler_config.enable_engine_pd_disagg:
            instance_type = InstanceType.DECODE
            decode_instance_id = self.dispatch_scheduler.dispatch(
                instance_type=instance_type,
                instance_info=self.decode_instance_info,
                instance_num_requests=self.decode_instance_num_requests
            )

    def general_dispatch(
        self,
        instance_type: InstanceType,
        instance_info: Dict[str, InstanceInfo],
        instance_num_requests: Dict[str, int],
    ) -> str:
        num_requests = self.instance_type_num_requests.get(instance_type, 0) + 1
        self.instance_type_num_requests[instance_type] = num_requests
        dispatch_instance_id = self.dispatch_policy.dispatch(instance_type, instance_num_requests, instance_info)
        assert instance_info[dispatch_instance_id].instance_type in [instance_type, InstanceType.NO_CONSTRAINTS]

        if num_requests % DISPATCH_LOG_FREQUENCY == 0:
            logger.info("dispatch scheduler total_dispatched_requests: {}".format(num_requests))
            for instance_id, num_requests in instance_num_requests.items():
                logger.info("instance {} num_dispatched_requests: {}".format(instance_id, num_requests))
        return dispatch_instance_id
    
    def soft_pd_dispatch(
        self,
        instance_type: InstanceType,
        prefill_instance_infos: Dict[str, InstanceInfo],
        prefill_instance_num_requests: Dict[str, int],
        decode_instance_infos: Dict[str, InstanceInfo],
        decode_instance_num_requests: Dict[str, int]
    ) -> str:
        dispatch_instance_id = self.dispatch_policy.soft_dispatch(
            instance_type, prefill_instance_infos, prefill_instance_num_requests,
            decode_instance_infos, decode_instance_num_requests
        )
        num_requests = self.instance_type_num_requests.get(instance_type, 0) + 1
        self.instance_type_num_requests[instance_type] = num_requests
        if num_requests % DISPATCH_LOG_FREQUENCY == 0:
            logger.info("dispatch scheduler total_dispatched_requests for {}: {}".format(instance_type, num_requests))
            if instance_type == InstanceType.PREFILL:
                for instance_id, num_requests in prefill_instance_num_requests.items():
                    logger.info("instance {} num_dispatched_requests: {}".format(instance_id, num_requests))
            elif instance_type == InstanceType.DECODE:
                for instance_id, num_requests in decode_instance_num_requests.items():
                    logger.info("instance {} num_dispatched_requests: {}".format(instance_id, num_requests))

        return dispatch_instance_id
