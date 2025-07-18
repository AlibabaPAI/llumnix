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

from typing import Dict, Tuple, Optional

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.constants import DISPATCH_LOG_FREQUENCY
from llumnix.global_scheduler.dispatch_policy import DispatchPolicyFactory, DispatchPolicy, DispatchLoadMetricConfig
from llumnix.metrics.metrics_types import Summary
from llumnix.utils import InstanceType

logger = init_logger(__name__)

class DispatchScheduler:
    def __init__(self,
                 dispatch_policy: str,
                 topk_random_dispatch: int,
                 enable_pd_disagg: bool,
                 enable_engine_pd_disagg: bool,
                 enable_engine_semi_pd_disagg: bool,
                 enable_adaptive_pd: bool,
                 dispatch_load_metric_config: DispatchLoadMetricConfig,
                 dispatch_latency_metric: Summary = Summary("dummy"),
                 cache_meta_client_config_path: str = None) -> None:
        self.enable_pd_disagg = enable_pd_disagg
        self.enable_engine_pd_disagg = enable_engine_pd_disagg
        self.enable_engine_semi_pd_disagg = enable_engine_semi_pd_disagg
        self.enable_adaptive_pd = enable_adaptive_pd

        self.dispatch_policy: DispatchPolicy = DispatchPolicyFactory.get_policy(
            dispatch_policy,
            cache_meta_client_config_path=cache_meta_client_config_path,
            topk_random_dispatch=topk_random_dispatch,
            dispatch_load_metric_config=dispatch_load_metric_config
        )

        self.dispatch_latency_metric = dispatch_latency_metric

        # statistics
        self.instance_type_num_requests: Dict[InstanceType, int] = {
            InstanceType.NEUTRAL: 0,
            InstanceType.DECODE: 0,
            InstanceType.PREFILL: 0
        }

        # enable_early_reject is only a tag — nothing is actually implemented for it.
        self.enable_early_reject = False

    def _log_instance_dispatch_info(self, instance_type: InstanceType, instance_num_requests: Dict[str, int]):
        self.instance_type_num_requests[instance_type] += 1
        num_total_requests = self.instance_type_num_requests[instance_type]
        if num_total_requests % DISPATCH_LOG_FREQUENCY == 0:
            logger.info("dispatch scheduler total_dispatched_requests: {}.".format(num_total_requests))
            for instance_id, num_requests in instance_num_requests.items():
                logger.info("{} instance {} num_dispatched_requests: {}.".format(
                    instance_type, instance_id, num_requests))

    def _dispatch(self,
                  instance_type: InstanceType,
                  primary_instance_infos: Dict[str, InstanceInfo],
                  primary_instance_num_requests: Dict[str, int],
                  secondary_instance_infos: Optional[Dict[str, InstanceInfo]],
                  secondary_instance_num_requests: Optional[Dict[str, int]],
                  dispatch_context: Dict,
                  ) -> str:
        # Filter primary instances based on dispatch policy
        candidate_instance_infos, candidate_instance_num_requests = self.dispatch_policy.filter(
            instance_type, primary_instance_infos, primary_instance_num_requests)
        if instance_type == InstanceType.DECODE:
            print(f"candidate_instance_infos: {candidate_instance_infos}\n")

        is_fallback_to_secondary = False

        # Adaptive PD fallback: try secondary instance type if primary unavailable
        if not candidate_instance_infos and self.enable_adaptive_pd:
            fallback_type = InstanceType.DECODE if instance_type == InstanceType.PREFILL else InstanceType.PREFILL
            candidate_instance_infos, candidate_instance_num_requests = self.dispatch_policy.filter(
                fallback_type, secondary_instance_infos, secondary_instance_num_requests)

            if candidate_instance_infos:
                instance_type = (InstanceType.DECODE_AS_PREFILL if instance_type == InstanceType.PREFILL
                               else InstanceType.PREFILL_AS_DECODE)
                is_fallback_to_secondary = True

        # Early reject or fallback to primary instances if no candidates found
        if not candidate_instance_infos:
            if self.enable_early_reject:
                return None
            candidate_instance_infos, candidate_instance_num_requests = primary_instance_infos, primary_instance_num_requests

        # Select target instance and update request count
        target_instance_id = self.dispatch_policy.select(
            instance_type, candidate_instance_num_requests, candidate_instance_infos, dispatch_context)

        if not is_fallback_to_secondary:
            primary_instance_num_requests[target_instance_id] += 1
        else:
            secondary_instance_num_requests[target_instance_id] += 1

        assert target_instance_id is not None, "No available instance for dispatch."
        return target_instance_id

    def dispatch_no_constrains(self,
                               instance_infos: Dict[str, InstanceInfo],
                               instance_num_requests: Dict[str, int],
                               dispatch_context: Dict,
                               ):
        instance_type = InstanceType.NEUTRAL

        with self.dispatch_latency_metric.observe_time(
            labels={"instance_type": InstanceType.NEUTRAL.value}
        ):
            target_instance_id = self._dispatch(instance_type, instance_infos, instance_num_requests, None, None, dispatch_context)

        self._log_instance_dispatch_info(instance_type, instance_num_requests)
        return target_instance_id

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
                    decode_instance_num_requests: Dict[str, int],
                    dispatch_context: Dict,
                    ) -> Tuple[str, str]:
        with self.dispatch_latency_metric.observe_time(
            labels={"instance_type": InstanceType.PREFILL.value}
        ):
            prefill_instance_id = self._dispatch(
                InstanceType.PREFILL,
                prefill_instance_infos,
                prefill_instance_num_requests,
                decode_instance_infos,
                decode_instance_num_requests,
                dispatch_context
            )
        instance_num_requests[prefill_instance_id] += 1

        self._log_instance_dispatch_info(InstanceType.PREFILL, prefill_instance_num_requests)

        decode_instance_id = None
        if not self.enable_pd_disagg:
            with self.dispatch_latency_metric.observe_time(
                labels={"instance_type": InstanceType.DECODE.value}
            ):
                if self.enable_adaptive_pd and prefill_instance_id in decode_instance_infos:
                    decode_instance_id = prefill_instance_id
                    decode_instance_num_requests[decode_instance_id] += 1
                else:
                    decode_instance_id = self._dispatch(
                        InstanceType.DECODE,
                        decode_instance_infos,
                        decode_instance_num_requests,
                        prefill_instance_infos,
                        prefill_instance_num_requests,
                        dispatch_context
                    )
            instance_num_requests[decode_instance_id] += 1
            self._log_instance_dispatch_info(InstanceType.DECODE, decode_instance_num_requests)

        assert self.enable_pd_disagg or decode_instance_id, "Decode instance must be selected in the dispatch scheduler, " \
            "except for llumnix-based pd."

        return prefill_instance_id, decode_instance_id
