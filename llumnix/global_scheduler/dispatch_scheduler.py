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

from typing import Dict, Tuple

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceType, InstanceInfo, all_instances_busy
from llumnix.constants import DISPATCH_LOG_FREQUENCY
from llumnix.global_scheduler.dispatch_policy import DispatchPolicyFactory, DispatchPolicy

logger = init_logger(__name__)


class DispatchScheduler:
    def __init__(self,
                 dispatch_policy: str,
                 topk_random_dispatch: int,
                 enable_pd_disagg: bool,
                 enable_engine_pd_disagg: bool,
                 enable_dynamic_pd_disagg: bool) -> None:
        self.enable_pd_disagg = enable_pd_disagg
        self.enable_engine_pd_disagg = enable_engine_pd_disagg
        self.enable_dynamic_pd_disagg = enable_dynamic_pd_disagg

        self.dispatch_policy: DispatchPolicy = DispatchPolicyFactory.get_policy(
            dispatch_policy,
            topk_random_dispatch=topk_random_dispatch)

        # statistics
        self.instance_type_num_requests: Dict[InstanceType, int] = {
            InstanceType.NO_CONSTRAINTS: 0,
            InstanceType.DECODE: 0,
            InstanceType.PREFILL: 0
        }

    def _log_instance_dispatch_info(self, instance_type: InstanceType, instance_num_requests: Dict[str, int]):
        self.instance_type_num_requests[instance_type] += 1
        num_total_requests = self.instance_type_num_requests[instance_type]
        if num_total_requests % DISPATCH_LOG_FREQUENCY == 0:
            logger.info("dispatch scheduler total_dispatched_requests: {}.".format(num_total_requests))
            for instance_id, num_requests in instance_num_requests.items():
                logger.info("{} instance {} num_dispatched_requests: {}.".format(
                    instance_type, instance_id, num_requests))

    def dispatch_no_constrains(self,
                 instance_infos: Dict[str, InstanceInfo],
                 instance_num_requests: Dict[str, int]):
        instance_type = InstanceType.NO_CONSTRAINTS
        targer_instance_id = self.dispatch_policy.dispatch(
            instance_type=instance_type,
            instance_num_requests=instance_num_requests,
            available_instance_infos=instance_infos,
        )
        instance_num_requests[targer_instance_id] += 1
        self._log_instance_dispatch_info(instance_type, instance_num_requests)

        return targer_instance_id

    # pylint: disable=unused-argument
    def _dispatch_normal_pd(self,
                            instance_infos: Dict[str, InstanceInfo],
                            instance_num_requests: Dict[str, int],
                            prefill_instance_infos: Dict[str, InstanceInfo],
                            prefill_instance_num_requests: Dict[str, int],
                            decode_instance_infos: Dict[str, InstanceInfo],
                            decode_instance_num_requests: Dict[str, int]) -> Tuple[str, str]:
        prefill_instance_id = self.dispatch_policy.dispatch(
            instance_type=InstanceType.PREFILL,
            instance_num_requests=prefill_instance_num_requests,
            available_instance_infos=prefill_instance_infos
        )
        prefill_instance_num_requests[prefill_instance_id] += 1
        instance_num_requests[prefill_instance_id] += 1

        decode_instance_id = None
        if self.enable_engine_pd_disagg:
            decode_instance_id = self.dispatch_policy.dispatch(
                instance_type=InstanceType.DECODE,
                instance_num_requests=decode_instance_num_requests,
                available_instance_infos=decode_instance_infos
            )
            decode_instance_num_requests[decode_instance_id] += 1
            instance_num_requests[decode_instance_id] += 1

        return prefill_instance_id, decode_instance_id

    # pylint: disable=unused-argument
    def _dispatch_dynamic_pd(self,
                            instance_infos: Dict[str, InstanceInfo],
                            instance_num_requests: Dict[str, int],
                            prefill_instance_infos: Dict[str, InstanceInfo],
                            prefill_instance_num_requests: Dict[str, int],
                            decode_instance_infos: Dict[str, InstanceInfo],
                            decode_instance_num_requests: Dict[str, int]) -> Tuple[str, str]:
        # choose prefill instance first
        if not all_instances_busy(prefill_instance_infos.values()):
            prefill_instance_id = self.dispatch_policy.dispatch(
                instance_type=InstanceType.PREFILL,
                instance_num_requests=prefill_instance_num_requests,
                available_instance_infos=prefill_instance_infos)
            prefill_instance_num_requests[prefill_instance_id] += 1
        else:
            # all prefills are busy, dispatch to free decode instances
            available_decode_instance_infos = {}
            available_decode_instance_num_requests = {}
            for ins_id, ins_info in decode_instance_infos.items():
                if not ins_info.dispatch_load_metric.is_busy():
                    available_decode_instance_infos[ins_id] = ins_info
                    available_decode_instance_num_requests[ins_id] = decode_instance_num_requests[ins_id]

            if len(available_decode_instance_infos) > 0:
                prefill_instance_id = self.dispatch_policy.dispatch(
                    instance_type=InstanceType.DECODE_AS_PREFILL,
                    instance_num_requests=available_decode_instance_num_requests,
                    available_instance_infos=decode_instance_infos)
                decode_instance_num_requests[prefill_instance_id] += 1
            else: # fallback to prefill instances
                prefill_instance_id = self.dispatch_policy.dispatch(
                    instance_type=InstanceType.PREFILL,
                    instance_num_requests=prefill_instance_num_requests,
                    available_instance_infos=prefill_instance_infos)
                prefill_instance_num_requests[prefill_instance_id] += 1
        instance_num_requests[prefill_instance_id] += 1

        # choose decode instance
        decode_instance_id = None
        if self.enable_engine_pd_disagg:
            if not all_instances_busy(decode_instance_infos.values()):
                decode_instance_id = self.dispatch_policy.dispatch(
                    instance_type=InstanceType.DECODE,
                    instance_num_requests=decode_instance_num_requests,
                    available_instance_infos=decode_instance_infos)
                decode_instance_num_requests[decode_instance_id] += 1
            else:
                # all decodes are busy, dispatch to free prefill instances
                available_prefill_instance_infos = {}
                available_prefill_instance_num_requests = {}
                for ins_id, ins_info in prefill_instance_infos.items():
                    if not ins_info.dispatch_load_metric.is_busy():
                        available_prefill_instance_infos[ins_id] = ins_info
                        available_prefill_instance_num_requests[ins_id] = prefill_instance_num_requests[ins_id]

                if len(available_prefill_instance_infos) > 0:
                    decode_instance_id = self.dispatch_policy.dispatch(
                        instance_type=InstanceType.PREFILL_AS_DECODE,
                        instance_num_requests=available_prefill_instance_num_requests,
                        available_instance_infos=available_prefill_instance_infos)
                    prefill_instance_num_requests[decode_instance_id] += 1
                else: # fallback to decode instances
                    decode_instance_id = self.dispatch_policy.dispatch(
                        instance_type=InstanceType.DECODE,
                        instance_num_requests=decode_instance_num_requests,
                        available_instance_infos=decode_instance_infos)
                    decode_instance_num_requests[decode_instance_id] += 1
            instance_num_requests[decode_instance_id] += 1

        return prefill_instance_id, decode_instance_id

    # For llumnix-based pd (enable_pd_disagg), the decode instance is not selected in the dispatch scheduler,
    # but rather based on the pair-migration-policy. For engine-based pd (enable_engine_pd_disagg), the decode
    # instance is selected in the dispatch scheduler.
    # pylint: disable=unused-argument
    def dispatch_pd(self,
                    instance_infos: Dict[str, InstanceInfo],
                    instance_num_requests: Dict[str, int],
                    prefill_instance_infos: Dict[str, InstanceInfo],
                    prefill_instance_num_requests: Dict[str, int],
                    decode_instance_infos: Dict[str, InstanceInfo],
                    decode_instance_num_requests: Dict[str, int]) -> Tuple[str, str]:
        if self.enable_dynamic_pd_disagg:
            prefill_instance_id, decode_instance_id = self._dispatch_dynamic_pd(
                instance_infos,
                instance_num_requests,
                prefill_instance_infos,
                prefill_instance_num_requests,
                decode_instance_infos,
                decode_instance_num_requests
            )
        else:
            prefill_instance_id, decode_instance_id = self._dispatch_normal_pd(
                instance_infos,
                instance_num_requests,
                prefill_instance_infos,
                prefill_instance_num_requests,
                decode_instance_infos,
                decode_instance_num_requests
            )

        if prefill_instance_id:
            self._log_instance_dispatch_info(InstanceType.PREFILL, prefill_instance_num_requests)

        if decode_instance_id:
            self._log_instance_dispatch_info(InstanceType.DECODE, decode_instance_num_requests)
        else:
            assert self.enable_pd_disagg, "decode instance should be selected in the dispatch scheduler, " \
                "except for llumnix-based pd."

        return prefill_instance_id, decode_instance_id
