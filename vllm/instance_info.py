import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


class RequestInfo:
    def __init__(self,
                 request_id: str,
                 session_id: str,
                 num_block: int,
                 arrival_time: float):
        self.request_id = request_id
        self.session_id = session_id
        self.num_block = num_block
        self.arrival_time = arrival_time
        self.waiting_time = 0.0

class InstanceLoadControlStrategy:
    def __init__(self,
                 load_metric: str,
                 enable_load_control_prefill: bool,
                 prefill_SLO: float):
        self.load_metric = load_metric
        self.enable_load_control_prefill = enable_load_control_prefill
        self.prefill_SLO = prefill_SLO

class InstanceInfo:
    def __init__(self,
                 num_total_gpu_block: int = 0,
                 num_watermark_block: int= 0,
                 num_used_gpu_block: int = 0,
                 num_free_gpu_block: int = 0,
                 gpu_cache_usage: float = 0.0,
                 num_running_request: int = 0,
                 num_waiting_request: int = 0,
                 num_killed_request: int = 0,
                 num_block_first_waiting_request: int = 0,
                 waiting_time_first_waiting_request: int = 0,
                 num_block_all_waiting_request: int = 0,
                 inference_type: str = "",
                 num_batched_tokens: int = 0,
                 num_priority_request: int = 0,
                 ):
        self.num_total_gpu_block = num_total_gpu_block
        self.num_watermark_block = num_watermark_block
        self.num_used_gpu_block = num_used_gpu_block
        self.num_free_gpu_block = num_free_gpu_block
        self.num_available_gpu_block = self.num_free_gpu_block - self.num_watermark_block
        self.gpu_cache_usage = gpu_cache_usage
        self.num_running_request = num_running_request
        self.num_waiting_request = num_waiting_request
        self.num_killed_request = num_killed_request

        self.num_block_first_waiting_request = num_block_first_waiting_request
        self.waiting_time_first_waiting_request = waiting_time_first_waiting_request
        self.num_block_all_waiting_request = num_block_all_waiting_request
        self.num_available_gpu_block_waiting = self.num_available_gpu_block - self.num_block_all_waiting_request
        self.num_priority_request = num_priority_request

        self.instance_id: None
        self.step_id: None
        self.timestamp: None
        self.max_tot_tokens = 0

        self.inference_type = inference_type
        self.latency = 0.0
        self.num_batched_tokens = num_batched_tokens
        self.running_seq_lens = []
        self.num_seq = 0

        self.num_block_last_running_request = 0

        self.instance_load = -np.inf
        self.instance_load_original = -np.inf
        self.instance_load_dispatch = -np.inf

class InstanceLoadInfo:
    def __init__(self, instance_info: InstanceInfo = None):
        self.num_total_gpu_block = instance_info.num_total_gpu_block
        self.num_watermark_block = instance_info.num_watermark_block
        self.num_used_gpu_block = instance_info.num_used_gpu_block
        self.num_free_gpu_block = instance_info.num_free_gpu_block
        self.num_available_gpu_block = instance_info.num_available_gpu_block

        self.num_waiting_request = instance_info.num_waiting_request
        self.num_running_request = instance_info.num_running_request
        self.num_killed_request = instance_info.num_killed_request


        self.num_block_first_waiting_request = instance_info.num_block_first_waiting_request
        self.waiting_time_first_waiting_request = instance_info.waiting_time_first_waiting_request
        self.num_block_all_waiting_request = instance_info.num_block_all_waiting_request
        self.prefill_alpha = 0.0
        # reserving blocks for high priority_blocks, set target 2k tokens load, block_size==16
        self.priority_reserved_blocks = self.num_total_gpu_block - 1600//16
        # logger.info(f"high priority reserved blocks:{self.priority_reserved_blocks}")
        # self.priority_alpha = 1.0 if instance_info.num_priority_request else 0.0
        self.num_priority_request = instance_info.num_priority_request

        self.instance_id = instance_info.instance_id
        self.step_id = instance_info.step_id

    def compute_instance_load(self, load_control_strategy: InstanceLoadControlStrategy) -> float:
        if load_control_strategy.load_metric == 'consumed_speed':
            if not load_control_strategy.enable_load_control_prefill:
                # import pdb; pdb.set_trace()
                # num_request = self.num_running_request + self.num_waiting_request
                # num_available_gpu_block = self.num_available_gpu_block - self.num_block_all_waiting_request
                num_request = self.num_running_request
                num_available_gpu_block = self.num_available_gpu_block
                if num_request == 0:
                    return -np.inf
                else:
                    return (num_available_gpu_block / num_request)*(-1)
            else:
                # import pdb; pdb.set_trace()
                num_request = self.num_running_request
                if self.num_waiting_request != 0:
                    num_request += 1
                # num_request = self.num_running_request + self.num_waiting_request
                self.prefill_beta = 0.0
                if self.waiting_time_first_waiting_request >= load_control_strategy.prefill_SLO * self.prefill_beta:
                    self.prefill_alpha = 1.0
                num_available_gpu_block = self.num_available_gpu_block - self.num_block_all_waiting_request
                # num_available_gpu_block = self.num_available_gpu_block - self.prefill_alpha * self.num_block_all_waiting_request
                # num_request += 7 * self.num_priority_request
                if self.num_priority_request:
                    # num_available_gpu_block -= 10 * num_request
                    num_available_gpu_block -= self.priority_reserved_blocks
                if num_request == 0:
                    return -np.inf
                else:
                    return (num_available_gpu_block / num_request)*(-1)
        elif load_control_strategy.load_metric == 'used_ratio':
            return self.num_used_gpu_block / self.num_total_gpu_block

    def compute_instance_load_dispatch(self, load_control_strategy: InstanceLoadControlStrategy, dispatch_strategy: str) -> float:
        if dispatch_strategy == 'block':
            return (self.num_used_gpu_block + self.num_block_all_waiting_request) / self.num_total_gpu_block
            # return (self.num_used_gpu_block) / self.num_total_gpu_block
        if load_control_strategy.load_metric == 'consumed_speed':
            num_request = self.num_running_request + self.num_waiting_request
            # num_request = self.num_running_request
            num_available_gpu_block = self.num_available_gpu_block - self.num_block_all_waiting_request
            # num_available_gpu_block = self.num_available_gpu_block
            if (self.num_priority_request == 0 and dispatch_strategy=='priority') \
                or (self.num_priority_request > 0 and dispatch_strategy!='priority'):
                num_available_gpu_block -= self.priority_reserved_blocks

            if num_request == 0:
                return -np.inf
            else:
                return (num_available_gpu_block / num_request)*(-1)
        elif load_control_strategy.load_metric == 'used_ratio':
            return (self.num_used_gpu_block + self.num_block_all_waiting_request) / self.num_total_gpu_block
            # return (self.num_used_gpu_block) / self.num_total_gpu_block

    def compute_instance_load_global_dispatch(self) -> float:
        return (self.num_used_gpu_block + self.num_block_all_waiting_request) / self.num_total_gpu_block


def get_instance_load(instance_info: InstanceInfo,
                      load_control_strategy: InstanceLoadControlStrategy,
                      key: str = 'migrate',
                      dispatch_strategy: str = None) -> float:
    instance_load_info = InstanceLoadInfo(instance_info)
    assert key in ['migrate', 'dispatch', 'global_dispatch']
    if key == 'migrate':
        instance_load = instance_load_info.compute_instance_load(load_control_strategy)
    elif key == 'dispatch':
        instance_load = instance_load_info.compute_instance_load_dispatch(load_control_strategy, dispatch_strategy)
        # print("compute_instance_load_dispatch")
    elif key == 'global_dispatch':
        instance_load = instance_load_info.compute_instance_load_global_dispatch()

    return instance_load
