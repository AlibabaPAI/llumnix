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

from typing import Dict, List, Tuple, Union, Optional
from abc import ABC, abstractmethod
import random
import hashlib
import array
import json
import torch

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, INSTANCE_TYPE_TO_METRIC_FIELD, sort_instance_infos
from llumnix.global_scheduler.query_client import build_meta_client_from_config, CacheMetaClient
from llumnix.global_scheduler.dispatch_filter import MetricBasedFilter
from llumnix.internal_config import DispatchLoadMetricConfig
import llumnix.envs as llumnix_envs
from llumnix.utils import InstanceType

logger = init_logger(__name__)


class DispatchPolicy(ABC):
    def __init__(self, topk_random_dispatch: int = 1,
                 dispatch_load_metric_config: Optional[DispatchLoadMetricConfig] = None):
        self.topk_random_dispatch: int = topk_random_dispatch
        self.dispatch_load_metric_config: Optional[DispatchLoadMetricConfig] = dispatch_load_metric_config

    def filter(self,
               instance_type: InstanceType,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        raise NotImplementedError


    @abstractmethod
    def select(self,
               instance_type: InstanceType,
               instance_num_requests: Dict[str, int],
               available_instance_infos: Dict[str, InstanceInfo],
               dispatch_context: Dict,
               ) -> str:
        raise NotImplementedError

    def random_choice_from_top_k(self, sorted_instance_infos: List[InstanceInfo]):
        k = min(self.topk_random_dispatch, len(sorted_instance_infos))
        top_k_instance_infos = sorted_instance_infos[:k]
        return random.choice(top_k_instance_infos)


# Dispatch all requests to a single instance, used only for testing
class Flood(DispatchPolicy):
    def __init__(self, topk_random_dispatch: int, dispatch_load_metric_config: DispatchLoadMetricConfig):
        super().__init__(topk_random_dispatch, dispatch_load_metric_config)

    def filter(self,
               instance_type: InstanceType,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        return instance_infos, instance_num_requests

    def select(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo],
                 dispatch_context: Dict,
                 ) -> str:
        instance_id = max(instance_num_requests, key=instance_num_requests.get)
        return instance_id


class Balanced(DispatchPolicy):
    def __init__(self, topk_random_dispatch: int, dispatch_load_metric_config: DispatchLoadMetricConfig):
        super().__init__(topk_random_dispatch, dispatch_load_metric_config)

    def filter(self,
               instance_type: InstanceType,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        return instance_infos, instance_num_requests

    def select(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo],
                 dispatch_context: Dict,
                 ) -> str:
        instance_id = min(instance_num_requests, key=instance_num_requests.get)
        return instance_id


class Load(DispatchPolicy):

    def __init__(self, topk_random_dispatch: int, dispatch_load_metric_config: DispatchLoadMetricConfig):
        super().__init__(topk_random_dispatch, dispatch_load_metric_config)
        self.filters = {
              InstanceType.PREFILL: MetricBasedFilter(
                  metric=getattr(dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[InstanceType.PREFILL])
              ),
              InstanceType.DECODE: MetricBasedFilter(
                  metric=getattr(dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[InstanceType.DECODE])
              ),
              InstanceType.NEUTRAL: MetricBasedFilter(
                  metric=getattr(dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[InstanceType.NEUTRAL])
              ),
        }
        print(f"[zzy][debug] dispatch_load_metric_config: {dispatch_load_metric_config}")
        print(f"[zzy][debug] prefill filter metric: {self.filters[InstanceType.PREFILL].metric}")
        print(f"[zzy][debug] decode filter metric: {self.filters[InstanceType.DECODE].metric}")
        print(f"[zzy][debug] neutral filter metric: {self.filters[InstanceType.NEUTRAL].metric}")

    def filter(self,
               instance_type: InstanceType,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        instance_infos, instance_num_requests = self.filters[instance_type].filter(instance_infos, instance_num_requests)
        return instance_infos, instance_num_requests

    def select(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo],
                 dispatch_context: Dict,
                 ) -> str:
        dispatch_load_metric=getattr(self.dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[instance_type])
        sorted_instance_infos = sort_instance_infos(available_instance_infos.values(), dispatch_load_metric)
        print(f"[zzy][debug] dispatch_load_metric, : {dispatch_load_metric}, sorted_instance_infos: {sort_instance_infos}")
        instance_info_chosen = self.random_choice_from_top_k(sorted_instance_infos)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch request to {}, load: {}".format(instance_id, getattr(instance_info_chosen, dispatch_load_metric)))
        return instance_id


class Queue(DispatchPolicy):
    def __init__(self, topk_random_dispatch: int, dispatch_load_metric_config: DispatchLoadMetricConfig):
        super().__init__(topk_random_dispatch, dispatch_load_metric_config)

    def filter(self,
               instance_type: InstanceType,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        return instance_infos, instance_num_requests

    def select(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo],
                 dispatch_context: Dict,
                 ) -> str:
        sorted_instance_infos = sort_instance_infos(available_instance_infos.values(), 'num_waiting_requests')
        instance_info_chosen = self.random_choice_from_top_k(sorted_instance_infos)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch request to {}, queue size: {}".format(instance_id, instance_info_chosen.num_waiting_requests))
        return instance_id


class RoundRobin(DispatchPolicy):
    def __init__(self, topk_random_dispatch: int, dispatch_load_metric_config: DispatchLoadMetricConfig):
        self.prev_instance_type_idx: Dict[str, int] = {}
        super().__init__(topk_random_dispatch, dispatch_load_metric_config)

    def filter(self,
               instance_type: InstanceType,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        return instance_infos, instance_num_requests

    def select(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo],
                 dispatch_context: Dict,
                 ) -> str:
        prev_idx = self.prev_instance_type_idx.get(instance_type, -1)
        all_instance_ids = sorted(instance_num_requests.keys())
        cur_idx = (prev_idx + 1) % len(all_instance_ids)
        target_instance_id = all_instance_ids[cur_idx]
        self.prev_instance_type_idx[instance_type] = cur_idx
        return target_instance_id


class CacheAware(DispatchPolicy):

    def __init__(self,
                 meta_client: CacheMetaClient,
                 chunk_size: int,
                 save_unfull_chunk: bool,
                 topk_random_dispatch: int,
                 dispatch_load_metric_config: DispatchLoadMetricConfig):
        super().__init__(topk_random_dispatch, dispatch_load_metric_config)
        self.meta_client = meta_client
        self.chunk_size = chunk_size
        self.save_unfull_chunk = save_unfull_chunk
        self.transfer_threshold = llumnix_envs.CACHE_TRANSFER_THRESHOLD
        self.transfer_penalty_factor = llumnix_envs.CACHE_TRANSFER_PENALTY_FACTOR
        self.filters = {
              InstanceType.PREFILL: MetricBasedFilter(
                  metric=getattr(dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[InstanceType.PREFILL])
              ),
              InstanceType.DECODE: MetricBasedFilter(
                  metric=getattr(dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[InstanceType.DECODE])
              ),
              InstanceType.NEUTRAL: MetricBasedFilter(
                  metric=getattr(dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[InstanceType.NEUTRAL])
              ),
        }

    def filter(self,
               instance_type: InstanceType,
               instance_infos: Dict[str, InstanceInfo],
               instance_num_requests: Dict[str, int],
               ) -> Tuple[Dict[str, InstanceInfo], Dict[str, int]]:
        instance_infos, instance_num_requests = self.filters[instance_type].filter(instance_infos, instance_num_requests)
        return instance_infos, instance_num_requests

    def _get_init_hash(self) -> str:
        return ""

    def _hash(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: str,
    ) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens_bytes = tokens.cpu().to(torch.uint32).numpy().tobytes()
        elif isinstance(tokens, list):
            tokens_bytes = array.array("I", tokens).tobytes()
        return hashlib.sha256(prefix_hash.encode("ascii") + tokens_bytes).hexdigest()

    def _chunk_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> List[Union[torch.Tensor, List[int]]]:
        """
        Split the tokens into chunks according to self.chunk_size.

        :param tokens: Input tokens with shape [seq_len]
        :return: A list of token chunks, each with a size of chunk_size
        """
        end = (
            len(tokens)
            if self.save_unfull_chunk
            else (len(tokens) - len(tokens) % self.chunk_size)
        )
        chunks = []
        for i in range(0, end, self.chunk_size):
            chunks.append(tokens[i : i + self.chunk_size])
        return chunks

    def _prefix_hash(
        self,
        token_chunks: List[Union[torch.Tensor, List[int]]],
    ) -> List[str]:
        prefix_hash = self._get_init_hash()
        prefix_hashes = []
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            prefix_hashes.append(prefix_hash)
        return prefix_hashes

    def _query_prefix_hash_hit_info(self, prefix_hashes):
        """
        Query the hit status of each prefix_hash.

        :param prefix_hashes: List of prefix hashes
        :return: {prefix_hash: [instance_id_0, instance_id_1, ...], ...}
        """
        prefix_hash_hit_info = {}
        top_n = 0  # 0 means return all
        for prefix_hash in prefix_hashes:
            hit_list = self.meta_client.query_cache_locality(prefix_hash, top_n)
            prefix_hash_hit_info[prefix_hash] = hit_list
        return prefix_hash_hit_info

    def _calc_instance_prefix_hit_count(self, prefix_hashes, prefix_hash_hit_info):
        """
        Count the number of prefix chunks hit by each instance.

        Only consider prefix hits: if a chunk is not hit, subsequent chunks will not be counted even if they are hit.

        :param prefix_hashes: List of prefix hashes
        :param prefix_hash_hit_info: {prefix_hash: [instance_id_0, instance_id_1, ...], ...}
        :return: {instance_id: number of hit chunks}
        """
        instance_prefix_hit_count = {}
        instance_broken = {}

        for _, prefix_hash in enumerate(prefix_hashes):
            hit_instance_ids = prefix_hash_hit_info.get(prefix_hash, [])
            for instance_id in hit_instance_ids:
                if not instance_broken.get(instance_id, False):
                    instance_prefix_hit_count[instance_id] = instance_prefix_hit_count.get(instance_id, 0) + 1
            for instance_id in instance_prefix_hit_count:
                if instance_id not in hit_instance_ids:
                    instance_broken[instance_id] = True
        return instance_prefix_hit_count

    def select(self,
                 instance_type: InstanceType,
                 instance_num_requests: Dict[str, int],
                 available_instance_infos: Dict[str, InstanceInfo],
                 dispatch_context: Dict,
                 ) -> str:
        request = dispatch_context['engine_core_request']
        # First, split into chunks
        # The hash chunking and calculation method need to be consistent with the kv-store component
        prompt_token_ids = request.prompt_token_ids
        token_chunks = self._chunk_tokens(prompt_token_ids)
        prefix_hashes = self._prefix_hash(token_chunks)

        # Query the hit status of each chunk
        prefix_hash_hit_info = self._query_prefix_hash_hit_info(prefix_hashes)

        if instance_type in (InstanceType.PREFILL, InstanceType.NEUTRAL, InstanceType.DECODE_AS_PREFILL):
            # Calculate the number of prefix hit chunks for each instance
            instance_prefix_hit_count = self._calc_instance_prefix_hit_count(prefix_hashes, prefix_hash_hit_info)
            # Sort in descending order based on the number of hits
            sorted_instance_prefix_hit_count = sorted(instance_prefix_hit_count.items(), key=lambda item: item[1], reverse=True)
            if len(sorted_instance_prefix_hit_count) > 0:
                max_hit_count = sorted_instance_prefix_hit_count[0][1]
            else:
                max_hit_count = 0

            # Iterate through the sorted instances to find an instance in available_instance_infos
            has_hit = False
            local_hit_count = 0
            for instance_id, _ in sorted_instance_prefix_hit_count:
                if instance_id in available_instance_infos:
                    target_instance_id = instance_id
                    has_hit = True
                    local_hit_count = instance_prefix_hit_count[instance_id]
                    break

            if not has_hit:
                dispatch_load_metric=getattr(self.dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[instance_type])
                sorted_instance_infos = sort_instance_infos(available_instance_infos.values(),dispatch_load_metric)
                instance_info_chosen = self.random_choice_from_top_k(sorted_instance_infos)
                target_instance_id = instance_info_chosen.instance_id
                logger.info("dispatch request to {}, load: {}".format(target_instance_id, getattr(instance_info_chosen, dispatch_load_metric)))

            if (max_hit_count - local_hit_count) * self.chunk_size > self.transfer_threshold:
                hit_length = max_hit_count * self.chunk_size
                transfer_penalty = self.transfer_penalty_factor
            else:
                hit_length = local_hit_count * self.chunk_size
                transfer_penalty = 1

            dispatch_context['hit_length'] = hit_length
            dispatch_context['transfer_penalty'] = transfer_penalty

            return target_instance_id

        dispatch_load_metric=getattr(self.dispatch_load_metric_config, INSTANCE_TYPE_TO_METRIC_FIELD[instance_type])
        sorted_instance_infos = sort_instance_infos(available_instance_infos.values(), dispatch_load_metric)
        instance_info_chosen = self.random_choice_from_top_k(sorted_instance_infos)
        instance_id = instance_info_chosen.instance_id
        logger.info("dispatch request to {}, load: {}".format(instance_id, getattr(instance_info_chosen, dispatch_load_metric)))
        return instance_id


class DispatchPolicyFactory:
    _POLICY_REGISTRY = {
        'flood': Flood,
        'balanced': Balanced,
        'load': Load,
        'queue': Queue,
        'rr': RoundRobin,
        'cacheaware': CacheAware
    }

    @classmethod
    def get_policy(cls, policy_name: str, cache_meta_client_config_path: str = None, **kwargs) -> DispatchPolicy:
        policy_class = cls._POLICY_REGISTRY[policy_name.lower()]

        if policy_name.lower() == "cacheaware":
            if cache_meta_client_config_path is None:
                raise ValueError("cache_meta_client_config_path is required for cacheaware policy")

            meta_client = build_meta_client_from_config(cache_meta_client_config_path)
            # Get chunk_size and save_unfull_chunk from config
            with open(cache_meta_client_config_path, 'r', encoding='utf-8') as f:
                cache_aware_query_client_config = json.load(f)
            chunk_size = cache_aware_query_client_config.get("chunk_size", 256)
            save_unfull_chunk = cache_aware_query_client_config.get("save_unfull_chunk", False)
            return policy_class(meta_client=meta_client, chunk_size=chunk_size, save_unfull_chunk=save_unfull_chunk, **kwargs)

        return policy_class(**kwargs)
