from typing import Dict, List, Tuple, Optional
import functools
import numpy as np
import queue
import copy
import time

from vllm.config import RequestSchedulerConfig
from vllm.instance_info import InstanceLoadControlStrategy, InstanceInfo, RequestInfo
from vllm.instance_info import get_instance_load
from vllm.logger import init_logger

logger = init_logger(__name__)


class RequestScheduler:
    def __init__(self, 
                 request_scheduler_config: RequestSchedulerConfig, 
                 enable_global_dispatch: bool,
                 enable_migrate: bool) -> None:
        self.request_scheduler_config = request_scheduler_config
        self.enable_global_dispatch = enable_global_dispatch
        self.enable_migrate = enable_migrate
        self.num_instance = self.request_scheduler_config.instance_parallel_size
        self.load_metric = self.request_scheduler_config.load_metric
        self.enable_load_control_prefill = self.request_scheduler_config.enable_load_control_prefill
        self.prefill_SLO = self.request_scheduler_config.prefill_SLO
        self.load_control_strategy = InstanceLoadControlStrategy(load_metric=self.load_metric, 
                                                                 enable_load_control_prefill=self.enable_load_control_prefill,
                                                                 prefill_SLO=self.prefill_SLO)
        self.dispatch_strategy = self.request_scheduler_config.dispatch_strategy
        self.instance_ptr = 0
        self.session_instance : Dict[str, int] = {}
        self.instance_info: Dict[int, InstanceInfo] = {}

        self.sorted_instance_infos: List[InstanceInfo] = []
        self.migrate_out_load_threshold = request_scheduler_config.migrate_out_load_threshold
        self.migrate_in_load_threshold = request_scheduler_config.migrate_in_load_threshold

        self.instance_num_request: Dict[int, int] = {}
        for instance_id in range(self.num_instance):
            self.instance_num_request[instance_id] = 0
        self.num_request = 0
        self.num_first_turn_request = 0
        self.num_nonfirst_turn_request = 0

        if self.dispatch_strategy == 'unbalanced':
            self.unbalanced_ratio = 0.999
            print(f"self.unbalanced_ratio: {self.unbalanced_ratio}")
        
        self.scale_up_threshold = request_scheduler_config.scale_up_threshold
        self.scale_down_threshold = request_scheduler_config.scale_down_threshold
        self.scale_strategy = self.request_scheduler_config.scale_strategy

        self.global_dispatch_strategy = request_scheduler_config.global_dispatch_strategy
        self.request_info: Dict[str, RequestInfo] = {}
        self.request_queue = queue.Queue()
        self.instance_request_pre_dispatch: Dict[int, str] = {}
        self.dispatch_watermark_ratio = 0.0
    
    def add_request(
            self,
            request_id: str,
            session_id: str,
            num_block: int) -> None:
        # logger.info(f"Add request {request_id} to Request Scheduler")
        arrival_time = time.time()
        self.request_info[request_id] = RequestInfo(
            request_id, session_id, num_block, arrival_time=arrival_time)
        self.request_queue.put(request_id)

    def dispatch(self, session_id: str) -> int:
        self.num_request += 1
        if self.dispatch_strategy == 'naive':
            instance_id = self.dispatch_naive(session_id)
        if self.dispatch_strategy == 'unbalanced':
            instance_id = self.dispatch_unbalanced(session_id)
        elif self.dispatch_strategy == 'balanced':
            instance_id = self.dispatch_balanced(session_id)
        elif self.dispatch_strategy == 'load':
            instance_id = self.dispatch_load(session_id)
        elif self.dispatch_strategy == 'block':
            instance_id = self.dispatch_block(session_id)
        if self.num_request % 100 == 0:
            print(f"self.num_request: {self.num_request}")
            print(f"self.num_first_turn_request: {self.num_first_turn_request}")
            print(f"self.num_nonfirst_turn_request: {self.num_nonfirst_turn_request}")
            for id, num_request in self.instance_num_request.items():
                print(f"Instance {id} num_dispatched_request: {num_request}")

        return instance_id

    def dispatch_naive(self, session_id: str) -> int:
        if session_id not in self.session_instance:
            # poll dispatch
            instance_id = self.instance_ptr
            self.instance_ptr = (self.instance_ptr + 1) % self.num_instance
            self.session_instance[session_id] = instance_id
            self.num_first_turn_request += 1
        else:
            instance_id = self.session_instance[session_id]
            self.num_nonfirst_turn_request += 1
        self.instance_num_request[instance_id] += 1

        return instance_id

    def dispatch_unbalanced(self, session_id: str) -> int:
        # random unbalanced dispatch
        import random
        random_num = random.random()
        if random_num <= self.unbalanced_ratio:            
            instance_id = 0
        else:
            instance_id = 1
        self.instance_num_request[instance_id] += 1

        return instance_id

    def dispatch_balanced(self, session_id: str) -> int:
        if session_id not in self.session_instance:
            # dispatch request according to the number of request dispatched to instance by manager
            instance_id = min(self.instance_num_request, key=self.instance_num_request.get)
            self.session_instance[session_id] = instance_id
            self.num_first_turn_request += 1
        else:
            instance_id = self.session_instance[session_id]
            self.num_nonfirst_turn_request += 1
        self.instance_num_request[instance_id] += 1

        return instance_id
    
    def dispatch_load(self, session_id: str) -> int:
        if session_id not in self.session_instance:
            # dispatch request according to the instance load metric
            if len(self.instance_info) < self.num_instance:
                instance_id = self.instance_ptr
                self.instance_ptr = (self.instance_ptr + 1) % self.num_instance
            else:
                self._sort_instance_info_by_load(descending=False, key='dispatch')
                instance_id = self.sorted_instance_infos[0].instance_id
                logger.info(f"dispatch to {instance_id},load:{self.sorted_instance_infos[0].instance_load_dispatch}")
            self.session_instance[session_id] = instance_id
            self.num_first_turn_request += 1
        else:
            instance_id = self.session_instance[session_id]
            self.num_nonfirst_turn_request += 1
        self.instance_num_request[instance_id] += 1

        return instance_id

    def dispatch_block(self, session_id: str) -> int:
        if session_id not in self.session_instance:
            # dispatch request according to the instance load metric
            if len(self.instance_info) < self.num_instance:
                instance_id = self.instance_ptr
                self.instance_ptr = (self.instance_ptr + 1) % self.num_instance
            else:
                self._sort_instance_info_by_load(descending=False, key='dispatch')
                instance_id = self.sorted_instance_infos[0].instance_id
            self.session_instance[session_id] = instance_id
            self.num_first_turn_request += 1
        else:
            instance_id = self.session_instance[session_id]
            self.num_nonfirst_turn_request += 1
        self.instance_num_request[instance_id] += 1

        return instance_id

    def need_dispatch(self) -> List[Tuple[int, int]]:
        if len(self.instance_info) < self.num_instance:
            request_instance_list = self.global_dispatch_start()
            return request_instance_list
        
        if self.global_dispatch_strategy == 'FFIT':
            request_instance_list = self.global_dispatch_FFIT(self.enable_migrate, self.enable_load_control_prefill)
        elif self.global_dispatch_strategy == 'FCFS':
            request_instance_list = self.global_dispatch_FCFS(self.enable_migrate, self.enable_load_control_prefill)
        elif self.global_dispatch_strategy == 'BE':
            request_instance_list = self.global_dispatch_BE()
        elif self.global_dispatch_strategy == 'SJF':
            request_instance_list = self.global_dispatch_SJF()
        elif self.global_dispatch_strategy == 'LJF':
            request_instance_list = self.global_dispatch_LJF()

        return request_instance_list
    
    def global_dispatch_start(self) -> List[Tuple[int, int]]:
        request_instance_list = []
        while not self.request_queue.empty() and len(self.instance_info) < self.num_instance:
            request_id = self.request_queue.get()
            instance_id = self.instance_ptr
            self.instance_ptr = (self.instance_ptr + 1) % self.num_instance
            request_instance_list.append((request_id, instance_id))
        
        return request_instance_list

    def global_dispatch_FFIT(self, enable_migrate, enable_load_control_prefill) -> List[Tuple[int, int]]:
        # migrate_in_instance_infos is arranged in descending order of num_available_gpu_block
        self._sort_instance_info_by_load(descending=True, key='global_dispatch')
        # only dispatch request to instance which is below migrate out threshold
        migrate_in_instance_infos = [i for i in reversed(self.sorted_instance_infos) 
                                     if i.num_killed_request == 0 and i.num_waiting_request == 0 
                                     and i.instance_load_original < self.migrate_out_load_threshold]
        num_migrate_out_instance = len(self.sorted_instance_infos) - len(migrate_in_instance_infos)
        num_migrate_in_instance = len(migrate_in_instance_infos) - num_migrate_out_instance
        request_instance_list = []
        if num_migrate_in_instance <= 0:
            return []
        request_queue = queue.Queue()
        instance_ptr = 0
        instance_request_pre_dispatch: Dict[int, str] = {}
        # all the instance before instance_ptr is pre-dispatch instance
        # num_migrate_in_instance / 2 is to make sure each pre-dispatch instance can find a migrate-in instance
        while not self.request_queue.empty() and instance_ptr < num_migrate_in_instance / 2:
            request_id = self.request_queue.get()
            curr_instance_info = migrate_in_instance_infos[instance_ptr]
            instance_id = curr_instance_info.instance_id
            num_dispatch_watermark_block = (curr_instance_info.num_running_request + 1) * self.dispatch_watermark_ratio
            # dispatch when the instance with the most available blocks has enough free blocks
            # instance might exceeds the migrate-out threshold after dispatch
            if curr_instance_info.num_available_gpu_block_waiting \
                >= self.request_info[request_id].num_block + num_dispatch_watermark_block:
                request_instance_list.append((request_id, instance_id))
                # dispatch request continuously until current instance doesn't have enough blocks
                curr_instance_info.num_running_request += 1
                curr_instance_info.num_available_gpu_block_waiting -= self.request_info[request_id].num_block
                # resort the list to make instance_ptr always point to the instance with the most available blocks
                tmp_instance_ptr = instance_ptr
                while tmp_instance_ptr + 1 < num_migrate_in_instance and \
                    migrate_in_instance_infos[tmp_instance_ptr].num_available_gpu_block_waiting \
                        < migrate_in_instance_infos[tmp_instance_ptr + 1].num_available_gpu_block_waiting:
                    migrate_in_instance_infos[tmp_instance_ptr] = migrate_in_instance_infos[tmp_instance_ptr + 1]
                    tmp_instance_ptr += 1
                migrate_in_instance_infos[tmp_instance_ptr] = curr_instance_info
            # if instance with the most available blocks doesn't have enough blocks to dispatch
            # pre-dispatch the request to the instance
            else:
                request_queue.put(request_id)
                if enable_migrate and enable_load_control_prefill:
                    right_instance_info = migrate_in_instance_infos[num_migrate_in_instance - 1 - instance_ptr]
                    right_instance_load = get_instance_load(right_instance_info, self.load_control_strategy)
                    # make sure each pre-dispatch instance can find a migrate-in instance
                    if right_instance_load > self.migrate_out_load_threshold:
                        break
                    # always pre-dispatch current request to the instance with the most available blocks
                    instance_request_pre_dispatch[instance_id] = request_id
                # each instance has only 1 pre-dispatch request at most
                instance_ptr += 1
        self.instance_request_pre_dispatch = instance_request_pre_dispatch
        while not self.request_queue.empty():
            request_id = self.request_queue.get()
            request_queue.put(request_id)
        self.request_queue = request_queue

        return request_instance_list
    
    def global_dispatch_FCFS(self, enable_migrate, enable_load_control_prefill) -> List[Tuple[int, int]]:
        # A better way here is to use a heap
        # migrate_in_instance_infos is arranged in descending order of num_available_gpu_block
        self._sort_instance_info_by_load(descending=True, key='global_dispatch')
        # only dispatch request to instance which is below migrate out threshold
        migrate_in_instance_infos = [i for i in reversed(self.sorted_instance_infos) 
                                     if i.num_killed_request == 0 and i.num_waiting_request == 0 
                                     and i.instance_load_original < self.migrate_out_load_threshold]
        num_migrate_out_instance = len(self.sorted_instance_infos) - len(migrate_in_instance_infos)
        num_migrate_in_instance = len(migrate_in_instance_infos) - num_migrate_out_instance
        if num_migrate_in_instance <= 0:
            return []
        request_instance_list = []
        request_queue = queue.Queue()
        instance_ptr = 0
        instance_request_pre_dispatch: Dict[int, str] = {}
        while not self.request_queue.empty():
            request_id = self.request_queue.get()
            curr_instance_info = migrate_in_instance_infos[instance_ptr]
            instance_id = curr_instance_info.instance_id
            num_dispatch_watermark_block = (curr_instance_info.num_running_request + 1) * self.dispatch_watermark_ratio
            if curr_instance_info.num_available_gpu_block_waiting \
                >= self.request_info[request_id].num_block + num_dispatch_watermark_block:
                request_instance_list.append((request_id, instance_id))
                # dispatch request continuously until current instance doesn't have enough blocks
                curr_instance_info.num_running_request += 1
                curr_instance_info.num_available_gpu_block_waiting -= self.request_info[request_id].num_block
                # resort the migrate_in_instance_infos_list to make instance_ptr always point to the instance with the most available blocks
                tmp_instance_ptr = instance_ptr
                while tmp_instance_ptr + 1 < num_migrate_in_instance and \
                    migrate_in_instance_infos[tmp_instance_ptr].num_available_gpu_block_waiting \
                        < migrate_in_instance_infos[tmp_instance_ptr + 1].num_available_gpu_block_waiting:
                    migrate_in_instance_infos[tmp_instance_ptr] = migrate_in_instance_infos[tmp_instance_ptr + 1]
                    tmp_instance_ptr += 1
                migrate_in_instance_infos[tmp_instance_ptr] = curr_instance_info
            # if instance with the most available blocks can't serve current request
            else:
                request_queue.put(request_id)
                # pre-dispatch the request
                if enable_migrate and enable_load_control_prefill:
                    instance_request_pre_dispatch[instance_id] = request_id
                # FCFS, break when no instance can serve current request
                break
        self.instance_request_pre_dispatch = instance_request_pre_dispatch
        while not self.request_queue.empty():
            request_id = self.request_queue.get()
            request_queue.put(request_id)
        self.request_queue = request_queue

        return request_instance_list

    def global_dispatch_BE(self) -> List[Tuple[int, int]]:
        # A better way here is to use a heap
        self._sort_instance_info_by_load(descending=False, key='global_dispatch')
        request_instance_list = []
        request_queue = queue.Queue()
        instance_ptr = 0
        while not self.request_queue.empty():
            request_id = self.request_queue.get()
            curr_instance_info = self.sorted_instance_infos[instance_ptr]
            instance_id = curr_instance_info.instance_id
            num_dispatch_watermark_block = (curr_instance_info.num_running_request + 1) * self.dispatch_watermark_ratio
            if curr_instance_info.num_available_gpu_block_waiting >= self.request_info[request_id].num_block + num_dispatch_watermark_block:
                request_instance_list.append((request_id, instance_id))
                # dispatch request continuously until current instance doesn't have enough blocks
                curr_instance_info.num_running_request += 1
                curr_instance_info.num_available_gpu_block_waiting -= self.request_info[request_id].num_block
                # resort the sorted_instance_infos to make instance_ptr always point to the instance with the most available blocks
                tmp_instance_ptr = instance_ptr
                while tmp_instance_ptr + 1 < self.num_instance and \
                    self.sorted_instance_infos[tmp_instance_ptr].num_available_gpu_block_waiting \
                        < self.sorted_instance_infos[tmp_instance_ptr + 1].num_available_gpu_block_waiting:
                    self.sorted_instance_infos[tmp_instance_ptr] = self.sorted_instance_infos[tmp_instance_ptr + 1]
                    tmp_instance_ptr += 1
                self.sorted_instance_infos[tmp_instance_ptr] = curr_instance_info
            # if instance with the most available blocks can't serve current request
            else:
                request_queue.put(request_id)
        self.request_queue = request_queue

        return request_instance_list

    def global_dispatch_SJF(self) -> List[Tuple[int, int]]:
        # A better way here is to use a heap
        self._sort_instance_info_by_load(descending=False, key='global_dispatch')

        requests: List[Tuple[str, int]] = []
        sorted_request_queue = queue.Queue()
        request_queue = queue.Queue()
        while not self.request_queue.empty():
            request_id = self.request_queue.get()
            num_block = self.request_info[request_id].num_block
            requests.append((request_id, num_block))
            request_queue.put(request_id)
        self.request_queue = request_queue
        def cmp_request_len(a: Tuple[str, int], b: Tuple[str, int]):
                return a[1] - b[1]
        cmp_key_func = functools.cmp_to_key(cmp_request_len)
        sorted_requests = sorted(requests, key=cmp_key_func)
        for (request_id, num_block) in sorted_requests:
            sorted_request_queue.put(request_id)

        request_queue = queue.Queue()
        request_instance_list = []
        instance_ptr = 0
        while not sorted_request_queue.empty():
            request_id = sorted_request_queue.get()
            curr_instance_info = self.sorted_instance_infos[instance_ptr]
            instance_id = curr_instance_info.instance_id
            if curr_instance_info.num_available_gpu_block_waiting >= self.request_info[request_id].num_block:
                request_instance_list.append((request_id, instance_id))
                # dispatch request continuously until current instance doesn't have enough blocks
                curr_instance_info.num_running_request += 1
                curr_instance_info.num_available_gpu_block_waiting -= self.request_info[request_id].num_block
                # resort the sorted_instance_infos to make instance_ptr always point to the instance with the most available blocks
                tmp_instance_ptr = instance_ptr
                while tmp_instance_ptr + 1 < self.num_instance and \
                    self.sorted_instance_infos[tmp_instance_ptr].num_available_gpu_block_waiting \
                        < self.sorted_instance_infos[tmp_instance_ptr + 1].num_available_gpu_block_waiting:
                    self.sorted_instance_infos[tmp_instance_ptr] = self.sorted_instance_infos[tmp_instance_ptr + 1]
                    tmp_instance_ptr += 1
                self.sorted_instance_infos[tmp_instance_ptr] = curr_instance_info
            # if instance with the most available blocks can't serve current request
            else:
                request_queue.put(request_id)
        tmp_request_queue = queue.Queue()
        while not self.request_queue.empty():
            request_id = self.request_queue.get()
            if request_id in request_queue.queue:
                tmp_request_queue.put(request_id)
        self.request_queue = tmp_request_queue

        return request_instance_list

    def global_dispatch_LJF(self) -> List[Tuple[int, int]]:
        # A better way here is to use a heap
        self._sort_instance_info_by_load(descending=False, key='global_dispatch')

        requests: List[Tuple[str, int]] = []
        sorted_request_queue = queue.Queue()
        request_queue = queue.Queue()
        while not self.request_queue.empty():
            request_id = self.request_queue.get()
            num_block = self.request_info[request_id].num_block
            requests.append((request_id, num_block))
            request_queue.put(request_id)
        self.request_queue = request_queue
        def cmp_request_len(a: Tuple[str, int], b: Tuple[str, int]):
                return b[1] - a[1]
        cmp_key_func = functools.cmp_to_key(cmp_request_len)
        sorted_requests = sorted(requests, key=cmp_key_func)
        for (request_id, num_block) in sorted_requests:
            sorted_request_queue.put(request_id)

        request_queue = queue.Queue()
        request_instance_list = []
        instance_ptr = 0
        while not sorted_request_queue.empty():
            request_id = sorted_request_queue.get()
            curr_instance_info = self.sorted_instance_infos[instance_ptr]
            instance_id = curr_instance_info.instance_id
            if curr_instance_info.num_available_gpu_block_waiting >= self.request_info[request_id].num_block:
                request_instance_list.append((request_id, instance_id))
                # dispatch request continuously until current instance doesn't have enough blocks
                curr_instance_info.num_running_request += 1
                curr_instance_info.num_available_gpu_block_waiting -= self.request_info[request_id].num_block
                # resort the sorted_instance_infos to make instance_ptr always point to the instance with the most available blocks
                tmp_instance_ptr = instance_ptr
                while tmp_instance_ptr + 1 < self.num_instance and \
                    self.sorted_instance_infos[tmp_instance_ptr].num_available_gpu_block_waiting \
                        < self.sorted_instance_infos[tmp_instance_ptr + 1].num_available_gpu_block_waiting:
                    self.sorted_instance_infos[tmp_instance_ptr] = self.sorted_instance_infos[tmp_instance_ptr + 1]
                    tmp_instance_ptr += 1
                self.sorted_instance_infos[tmp_instance_ptr] = curr_instance_info
            # if instance with the most available blocks can't serve current request
            else:
                request_queue.put(request_id)
        tmp_request_queue = queue.Queue()
        while not self.request_queue.empty():
            request_id = self.request_queue.get()
            if request_id in request_queue.queue:
                tmp_request_queue.put(request_id)
        self.request_queue = tmp_request_queue

        return request_instance_list

    def update_instance_info(self, instance_info: InstanceInfo) -> None:
        # scheduler will dispatch according to the original instance load
        instance_info.instance_load_original = get_instance_load(instance_info, self.load_control_strategy)
        if not self.enable_global_dispatch:
            instance_info.instance_load_dispatch = get_instance_load(instance_info, self.load_control_strategy, 
                                                                     key='dispatch', dispatch_strategy=self.dispatch_strategy)
        else:
            instance_info.instance_load_dispatch = get_instance_load(instance_info, self.load_control_strategy, key='global_dispatch')
        # adjust instance info according to pre-dispatch dict
        instance_info = self._adjust_instance_info(instance_info)
        # compute instance load in consideration of pre-dispatch dict
        # scheduler will migrate and auto-scaling according to the instance load
        instance_info.instance_load = get_instance_load(instance_info, self.load_control_strategy)
        instance_id = instance_info.instance_id
        self.instance_info[instance_id] = instance_info

        return instance_info
    
    def _gather_instance_info(self) -> None:
        self.max_num_available_block_instance= max([i.num_available_gpu_block for i in self.instance_info.values()])
        self.tot_num_available_block = sum([i.num_available_gpu_block for i in self.instance_info.values()])

    def _adjust_instance_info(self, instance_info: InstanceInfo) -> None:
        instance_id = instance_info.instance_id
        if instance_info.num_waiting_request == 0 and instance_id in self.instance_request_pre_dispatch:
            request_id = self.instance_request_pre_dispatch[instance_id]
            request_info = self.request_info[request_id]
            instance_info.num_waiting_request = 1
            num_dispatch_watermark_block = (instance_info.num_running_request + 1) * self.dispatch_watermark_ratio
            # make sure instance keep migrating out until dispatch successfully
            instance_info.num_block_first_waiting_request = (request_info.num_block + num_dispatch_watermark_block)
        
        return instance_info

    def need_migrate(self) -> List[Tuple[int, int]]:
        if not self.enable_load_control_prefill:
            return self.need_migrate_balanced()
        else:
            # need_migrate_prefill_v2
            return self.need_migrate_prefill_v2()

    def need_migrate_balanced(self) -> List[Tuple[int, int]]:
        self._sort_instance_info_by_load(descending=True, key='migrate')
        # migrate out instance
        left_instance_infos = [i for i in self.sorted_instance_infos 
                              if i.num_killed_request > 0 or i.instance_load > self.migrate_out_load_threshold]
        # migrate in instance
        right_instance_infos = [i for i in reversed(self.sorted_instance_infos) 
                               if i.num_killed_request == 0 and i.instance_load < self.migrate_out_load_threshold]
        # Return the list of (migrate_out_instance_id, migrate_in_instance_id).
        migrate_instance_pairs = []
        for i in range(min(len(left_instance_infos), len(right_instance_infos))):
            # print(f"left id {left_instance_infos[i].instance_id}, right id {right_instance_infos[i].instance_id}")
            # print(f"left load {left_instance_infos[i].instance_load}, right load {right_instance_infos[i].instance_load}")
            load_diff_before_mig = left_instance_infos[i].instance_load - right_instance_infos[i].instance_load
            left_load_after_mig = self._get_instance_load_after_migrate(left_instance_infos[i], is_migrate_in=False)
            right_load_after_mig = self._get_instance_load_after_migrate(right_instance_infos[i], is_migrate_in=True)
            if right_load_after_mig > self.migrate_out_load_threshold:
                continue
            load_diff_after_mig = left_load_after_mig - right_load_after_mig
            if (load_diff_after_mig > 0 and load_diff_before_mig > load_diff_after_mig) or right_instance_infos[i].instance_load == -np.inf:
                migrate_instance_pairs.append((left_instance_infos[i].instance_id, right_instance_infos[i].instance_id))

        return migrate_instance_pairs
    
    def need_migrate_prefill_v1(self) -> List[Tuple[int, int]]:
        self._sort_instance_info_by_load(descending=True, key='migrate')
        # Return the list of (migrate_out_instance_id, migrate_in_instance_id).
        if len(self.sorted_instance_infos) < self.num_instance:
            return []
        return [(self.sorted_instance_infos[i].instance_id, 
                 self.sorted_instance_infos[self.num_instance - i - 1].instance_id) 
                 for i in range(int(self.num_instance / 2))]

    def need_migrate_prefill_v2(self) -> List[Tuple[int, int]]:
        self._sort_instance_info_by_load(descending=True, key='migrate')
        # migrate out instance
        left_instance_infos = [i for i in self.sorted_instance_infos 
                              if i.num_killed_request > 0 or i.instance_load > self.migrate_out_load_threshold]
        # migrate in instance
        right_instance_infos = [i for i in reversed(self.sorted_instance_infos) 
                               if i.num_killed_request == 0 and i.instance_load < self.migrate_out_load_threshold]
        # Return the list of (migrate_out_instance_id, migrate_in_instance_id).
        migrate_instance_pairs = []
        for i in range(min(len(left_instance_infos), len(right_instance_infos))):
            # print(f"left id {left_instance_infos[i].instance_id}, right id {right_instance_infos[i].instance_id}")
            # print(f"left load {left_instance_infos[i].instance_load}, right load {right_instance_infos[i].instance_load}")
            migrate_instance_pairs.append((left_instance_infos[i].instance_id, right_instance_infos[i].instance_id))

        return migrate_instance_pairs

    def need_scale(self) -> Tuple[int, int]:
        scale_up_num = 0
        scale_down_num = 0
        if len(self.instance_info.keys()) < self.num_instance:
            return scale_up_num, scale_down_num 
        load_metric_up = self._get_load_metric_up()
        load_metric_down = self._get_load_metric_down()
        if load_metric_up > self.scale_up_threshold:
            now_instances = [self.instance_info[i] for i in range(self.num_instance)]
            while self._get_avg_load(now_instances) > self.scale_up_threshold:
                scale_up_num += 1
                now_instances.append(self._get_dummy_instance_info())
        elif load_metric_down < self.scale_down_threshold:
            scale_down_num = 1
        
        return scale_up_num, scale_down_num

    def scale_up(self, scale_up_num) -> None:
        for _ in range(scale_up_num):
            self.instance_num_request[self.num_instance] = 0
            new_intance_info = self._get_dummy_instance_info()
            new_intance_info.instance_id = self.num_instance
            self.instance_info[new_intance_info.instance_id] = new_intance_info
            self.num_instance += 1

    def scale_down(self):
        scale_down_instance_id = self.num_instance - 1
        self.num_instance -= 1
        self.instance_info.pop(scale_down_instance_id, None)


    def _sort_instance_info_by_load(self, 
                                    descending: bool = True, 
                                    key: str = 'migrate') -> None:
        assert key in ['migrate', 'dispatch', 'global_dispatch']

        instance_infos: List[InstanceInfo] = []
        for instance_id in range(self.num_instance):
            if instance_id in self.instance_info:
                instance_infos.append(self.instance_info[instance_id])

        def cmp_instance_load(a: InstanceInfo, b: InstanceInfo):
            if key == 'migrate':
                a_load = a.instance_load
                b_load = b.instance_load
            elif key in ['dispatch', 'global_dispatch']:
                a_load = a.instance_load_dispatch
                b_load = b.instance_load_dispatch
            # if a.num_killed_request != b.num_killed_request:
            #     return b.num_killed_request - a.num_killed_request
            return b_load - a_load

        cmp_key_func = functools.cmp_to_key(cmp_instance_load)
        self.sorted_instance_infos = sorted(instance_infos, key=cmp_key_func)
        if not descending:
            self.sorted_instance_infos.reverse()

    def _get_instance_load_after_migrate(self, instance_info: InstanceInfo, is_migrate_in: bool) -> float:
        instance_info_after_migrate = copy.deepcopy(instance_info)
        num_block_last_running_request = instance_info_after_migrate.num_block_last_running_request
        if is_migrate_in:
            instance_info_after_migrate.num_running_request += 1
            instance_info_after_migrate.num_free_gpu_block -= num_block_last_running_request
        else:
            instance_info_after_migrate.num_running_request -= 1
            instance_info_after_migrate.num_free_gpu_block += num_block_last_running_request

        return get_instance_load(instance_info_after_migrate, self.load_control_strategy)

    def _get_dummy_instance_info(self):
        dummy_intance_info = InstanceInfo()
        dummy_intance_info.instance_id = -1
        dummy_intance_info.step_id = -1
        dummy_intance_info.num_total_gpu_block = self.instance_info[0].num_total_gpu_block
        dummy_intance_info.num_available_gpu_block = dummy_intance_info.num_total_gpu_block
        dummy_intance_info.num_free_gpu_block = dummy_intance_info.num_total_gpu_block
        dummy_intance_info.num_available_gpu_block_waiting = dummy_intance_info.num_total_gpu_block
        return dummy_intance_info

    def _get_avg_load(self, instance_info_list:List[InstanceInfo]) -> float:
        tot_instance_info = InstanceInfo()
        tot_instance_info.instance_id = -1
        tot_instance_info.step_id = -1
        tot_instance_info.num_running_request = sum([i.num_running_request for i in instance_info_list])
        tot_instance_info.num_waiting_request = sum([i.num_waiting_request for i in instance_info_list])
        tot_instance_info.num_free_gpu_block = sum([i.num_free_gpu_block for i in instance_info_list])
        tot_instance_info.num_total_gpu_block = sum([i.num_total_gpu_block for i in instance_info_list])
        tot_instance_info.num_watermark_block = sum([i.num_watermark_block for i in instance_info_list])
        tot_instance_info.num_block_all_waiting_request = sum([i.num_block_all_waiting_request for i in instance_info_list])
        tot_instance_info.num_available_gpu_block = tot_instance_info.num_free_gpu_block - tot_instance_info.num_watermark_block
        return get_instance_load(tot_instance_info, self.load_control_strategy, key="dispatch")

    def _get_load_metric_up(self):
        now_instances = [self.instance_info[i] for i in range(self.num_instance)]
        if self.scale_strategy == "max_load":
            load_metric_up = max([i.instance_load for i in now_instances])
        elif self.scale_strategy == "min_load":
            load_metric_up = min([i.instance_load for i in now_instances])
        elif self.scale_strategy == "avg_load":
            load_metric_up = self._get_avg_load(now_instances)
        
        return load_metric_up

    def _get_load_metric_down(self):
        now_instances = [self.instance_info[i] for i in range(self.num_instance)]
        if self.scale_strategy == "max_load":
            load_metric_down = max([i.instance_load for i in now_instances])
        elif self.scale_strategy == "min_load":
            load_metric_down = min([i.instance_load for i in now_instances])
        elif self.scale_strategy == "avg_load":
            tot_instance_info = InstanceInfo()
            tot_instance_info.instance_id = -1
            tot_instance_info.step_id = -1
            tot_instance_info.num_running_request = sum([i.num_running_request for i in now_instances])
            tot_instance_info.num_waiting_request = sum([i.num_waiting_request for i in now_instances])
            tot_instance_info.num_free_gpu_block = sum([i.num_free_gpu_block-i.num_total_gpu_block 
                                                        if i.instance_id+1==self.num_instance else i.num_free_gpu_block
                                                        for i in now_instances])
            tot_instance_info.num_free_gpu_block = max(0, tot_instance_info.num_free_gpu_block)
            tot_instance_info.num_total_gpu_block = sum([0 if i.instance_id+1==self.num_instance else i.num_total_gpu_block
                                                        for i in now_instances])
            tot_instance_info.num_watermark_block = sum([0 if i.instance_id+1==self.num_instance else i.num_watermark_block 
                                                        for i in now_instances])
            tot_instance_info.num_block_all_waiting_request = sum([i.num_block_all_waiting_request for i in now_instances])
            tot_instance_info.num_available_gpu_block = tot_instance_info.num_free_gpu_block - tot_instance_info.num_watermark_block
            load_metric_down = get_instance_load(tot_instance_info, self.load_control_strategy)
        
        return load_metric_down
