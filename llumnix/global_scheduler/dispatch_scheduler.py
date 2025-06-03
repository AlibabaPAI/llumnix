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

from dataclasses import dataclass
from typing import Dict, List, Tuple

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceType, InstanceInfo
from llumnix.constants import DISPATCH_LOG_FREQUENCY
from llumnix.global_scheduler.dispatch_policy import DispatchPolicyFactory, DispatchPolicy
from llumnix.global_scheduler.dispatch_filter import DispatchFilter, LoadBusyFilter
from llumnix.metrics.metrics_types import Summary

logger = init_logger(__name__)


@dataclass
class DispatchRule:
    instance_type: InstanceType = None # used to determine which load metric is used
    instance_filter: DispatchFilter = None

    def filter(self,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[List[InstanceInfo], Dict[str, int]]:
        available_instance_infos, available_instance_num_requests = instance_infos, instance_num_requests
        if self.instance_filter is not None:
            available_instance_infos, available_instance_num_requests = \
                self.instance_filter.filter(available_instance_infos, available_instance_num_requests)
        return available_instance_infos, available_instance_num_requests


class DispatchScheduler:
    def __init__(self,
                 dispatch_policy: str,
                 topk_random_dispatch: int,
                 enable_pd_disagg: bool,
                 enable_engine_pd_disagg: bool,
                 enable_adaptive_pd: bool,
                 dispatch_latency_metric: Summary = Summary("dummy")) -> None:
        self.enable_pd_disagg = enable_pd_disagg
        self.enable_engine_pd_disagg = enable_engine_pd_disagg
        self.enable_adaptive_pd = enable_adaptive_pd

        self.dispatch_policy: DispatchPolicy = DispatchPolicyFactory.get_policy(
            dispatch_policy,
            topk_random_dispatch=topk_random_dispatch)

        self.dispatch_latency_metric = dispatch_latency_metric

        # statistics
        self.instance_type_num_requests: Dict[InstanceType, int] = {
            InstanceType.NO_CONSTRAINTS: 0,
            InstanceType.DECODE: 0,
            InstanceType.PREFILL: 0
        }

        self._build_phase()

    def _build_phase(self):
        if not self.enable_engine_pd_disagg and not self.enable_pd_disagg:
            self.no_constrains_rules = []
            self.no_constrains_rules.append(DispatchRule(InstanceType.NO_CONSTRAINTS, None))
        else:
            self.busy_filter = LoadBusyFilter()
            self.prefill_dispatch_rules = []
            if self.enable_adaptive_pd:
                self.prefill_dispatch_rules.append(DispatchRule(InstanceType.PREFILL, self.busy_filter))
                self.prefill_dispatch_rules.append(DispatchRule(InstanceType.DECODE_AS_PREFILL, self.busy_filter))
            self.prefill_dispatch_rules.append(DispatchRule(InstanceType.PREFILL, None))

            self.decode_dispatch_rules = []
            if self.enable_engine_pd_disagg:
                if self.enable_adaptive_pd:
                    self.decode_dispatch_rules.append(DispatchRule(InstanceType.DECODE, self.busy_filter))
                    self.decode_dispatch_rules.append(DispatchRule(InstanceType.PREFILL_AS_DECODE, self.busy_filter))
                self.decode_dispatch_rules.append(DispatchRule(InstanceType.DECODE, None))

    def _log_instance_dispatch_info(self, instance_type: InstanceType, instance_num_requests: Dict[str, int]):
        self.instance_type_num_requests[instance_type] += 1
        num_total_requests = self.instance_type_num_requests[instance_type]
        if num_total_requests % DISPATCH_LOG_FREQUENCY == 0:
            logger.info("dispatch scheduler total_dispatched_requests: {}.".format(num_total_requests))
            for instance_id, num_requests in instance_num_requests.items():
                logger.info("{} instance {} num_dispatched_requests: {}.".format(
                    instance_type, instance_id, num_requests))

    def _dispatch(self,
                  phase_rules: List[DispatchRule],
                  phase_elements: List[Tuple[Dict[str, InstanceInfo], Dict[str, int]]]) -> str:
        target_instance_id = None
        for rule, element in zip(phase_rules, phase_elements):
            instance_infos, instance_num_requests = element
            available_instance_infos, available_instance_num_requests = rule.filter(instance_infos, instance_num_requests)
            if len(available_instance_infos) > 0:
                target_instance_id = self.dispatch_policy.dispatch(
                    instance_type=rule.instance_type,
                    instance_num_requests=available_instance_num_requests,
                    available_instance_infos=available_instance_infos,
                )
                instance_num_requests[target_instance_id] += 1
                break
        assert target_instance_id is not None, "No available instance for dispatch."
        return target_instance_id

    def dispatch_no_constrains(self,
                 instance_infos: Dict[str, InstanceInfo],
                 instance_num_requests: Dict[str, int]):
        instance_type = InstanceType.NO_CONSTRAINTS
        phase_elements = [(instance_infos, instance_num_requests)]

        with self.dispatch_latency_metric.observe_time(
            labels={"instance_type": InstanceType.NO_CONSTRAINTS.value}
        ):
            targer_instance_id = self._dispatch(self.no_constrains_rules, phase_elements)

        self._log_instance_dispatch_info(instance_type, instance_num_requests)
        return targer_instance_id

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
        prefill_dispatch_elements, decode_dispatch_elements = [], []
        if self.enable_adaptive_pd:
            prefill_dispatch_elements.extend([
                (prefill_instance_infos, prefill_instance_num_requests),
                (decode_instance_infos, decode_instance_num_requests)
            ])
            decode_dispatch_elements.extend([
                (decode_instance_infos, decode_instance_num_requests),
                (prefill_instance_infos, prefill_instance_num_requests)
            ])
        prefill_dispatch_elements.append((prefill_instance_infos, prefill_instance_num_requests))
        decode_dispatch_elements.append((decode_instance_infos, decode_instance_num_requests))

        with self.dispatch_latency_metric.observe_time(
            labels={"instance_type": InstanceType.PREFILL.value}
        ):
            prefill_instance_id = self._dispatch(self.prefill_dispatch_rules, prefill_dispatch_elements)
        instance_num_requests[prefill_instance_id] += 1

        self._log_instance_dispatch_info(InstanceType.PREFILL, prefill_instance_num_requests)

        decode_instance_id = None
        if self.enable_engine_pd_disagg:
            with self.dispatch_latency_metric.observe_time(
                labels={"instance_type": InstanceType.DECODE.value}
            ):
                if self.enable_adaptive_pd and prefill_instance_id in decode_instance_infos:
                    decode_instance_id = prefill_instance_id
                    decode_instance_num_requests[decode_instance_id] += 1
                else:
                    decode_instance_id = self._dispatch(self.decode_dispatch_rules, decode_dispatch_elements)
            instance_num_requests[decode_instance_id] += 1
            self._log_instance_dispatch_info(InstanceType.DECODE, decode_instance_num_requests)

        assert self.enable_pd_disagg or decode_instance_id, "Decode instance must be selected in the dispatch scheduler, " \
            "except for llumnix-based pd."

        return prefill_instance_id, decode_instance_id
