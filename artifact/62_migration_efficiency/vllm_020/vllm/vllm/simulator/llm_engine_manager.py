import asyncio
import time
import csv
from functools import partial
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict

from vllm.config import ParallelConfig, RequestSchedulerConfig
from vllm.engine.arg_utils import EngineManagerArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, EngineRequestInput
# from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray, RayWorker
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.core.request_scheduler import RequestScheduler
from vllm.instance_info import InstanceInfo
from vllm.logger import init_logger
from vllm.sequence import SequenceEvent
from vllm.simulator.profiling import *
from vllm.simulator.llm_engine import LLMEngine

if ray:
    from ray.air.util.torch_dist import init_torch_dist_process_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds


class LLMEngineManager:
    def __init__(self,
                 engine_args: EngineManagerArgs,
                 parallel_config: ParallelConfig,
                 request_scheduler_config: RequestSchedulerConfig,
                 log_requests: bool = True) -> None:
        self.engine_use_ray = engine_args.engine_use_ray
        self.worker_use_ray = engine_args.worker_use_ray
        self.enable_migrate = engine_args.enable_migrate
        self.async_migrate = engine_args.async_migrate
        self.profiling_result_file_path = engine_args.profiling_result_file_path
        logger.info(f"self.enable_migrate: {self.enable_migrate}")
        logger.info(f"self.async_migrate: {self.async_migrate}")
        self.parallel_config = parallel_config
        self.log_requests = log_requests
        self.record_instance_info = True
        self.record_req_info = True
        # # Create the parallel GPU workers.
        # if self.parallel_config.worker_use_ray:
        #     self._init_workers_ray(placement_group)
        # else:
        #     self._init_workers(distributed_init_method)
        self.scale_up_time = -1
        self.scale_down_time = -1
        self.scaling_up = False
        self.scaling_down = False
        self.max_replicas = engine_args.max_replicas
        self.min_replicas = engine_args.min_replicas
        self.enable_scaling = engine_args.enable_scaling
        self.engine_args = engine_args
        self.scaling_interval = 10
        self.last_check_scale_time = 100
        self.to_scale_up_num = 0
        self.to_scale_down_num = 0
        if not self.enable_scaling:
            self.max_replicas = self.min_replicas = self.parallel_config.instance_parallel_size
        self._init_instances(engine_args)
        self.dispatch_mode = engine_args.dispatch_mode
        # request_scheduler_class = ray.remote(num_cpus=0)(RequestScheduler).remote
        enable_global_dispatch = (self.dispatch_mode == 'global')
        self.generate_mode = 'original' if not enable_global_dispatch else 'global'
        self.request_scheduler = RequestScheduler(request_scheduler_config, enable_global_dispatch, self.enable_migrate)
        self.request_instance: Dict[str, int] = {}

        self.requests: Dict[str, Tuple] = {}
        self.request_queue = asyncio.Queue()
        self.request_events: Dict[str, asyncio.Event] = {}
        self.request_outputs: Dict[str, RequestOutput] = {}

        self.num_finished_request = 0
        self.migrated_requests: List[str] = []
        self.num_migrated_request = 0

        self.num_instance = self.parallel_config.instance_parallel_size
        self.instance_step: Dict[int, int] = {}
        for instance_id in range(self.num_instance):
            self.instance_step[instance_id] = -1
        self.num_instance_info_update = 0
        self.need_migrate_frequency = engine_args.need_migrate_frequency

        self.tokenize_instance_ptr = 0
        self.async_lock = asyncio.Lock()
        self.need_dispatch_frequency = engine_args.need_dispatch_frequency

        if self.record_instance_info:
            self._init_instance_info_csv(engine_args)
        if self.record_req_info:
            self._init_req_info_csv(engine_args)

    def generate(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None):
        if self.log_requests:
            logger.info(f"Received request {request_id}.")
            # logger.info(f"Received request {request_id}: "
            #             f"prompt: {prompt!r}, "
            #             f"sampling params: {sampling_params}, "
            #             f"prompt token ids: {prompt_token_ids}.")
        if self.generate_mode == 'original':
            self.generate_original(session_id, prompt, sampling_params, request_id, prompt_token_ids, arrival_time)
        elif self.generate_mode == 'global':
            self.generate_global(session_id, prompt, sampling_params, request_id, prompt_token_ids, arrival_time)

    def generate_original(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None):
        instance_id = self.request_scheduler.dispatch(session_id)
        self.request_instance[request_id] = instance_id
        # if self.engine_use_ray:
        #     outputs_generator = self.instances[instance_id].generate.remote(prompt, sampling_params, request_id)
        # else:
        #     outputs_generator = self.instances[instance_id].generate(prompt, sampling_params, request_id)
        self.instances[instance_id].add_request(request_id, prompt, sampling_params, prompt_token_ids, arrival_time=arrival_time)
    
    def generate_global(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None):
        self.add_request_dispatch(session_id, prompt, sampling_params, request_id, prompt_token_ids)
    
    def add_request_dispatch(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None) -> None:
        arrival_time = time.time()
        # dispatch request to engine to tokenize prompt
        instance_id = self.tokenize_instance_ptr
        self.tokenize_instance_ptr = (self.tokenize_instance_ptr + 1) % self.num_instance
        prompt_token_ids, num_block = self.instances[instance_id].tokenize(prompt)
        self.requests[request_id] = (session_id, prompt, sampling_params, arrival_time, prompt_token_ids)
        self.request_scheduler.add_request(request_id, session_id, num_block)
        # push request_scheduler to dispatch request once new request arrives and has been added to request_scheduler
        # push by manager rather than request_scheduler in order to control push dispatch frequency
        self._push_dispatch()
    
    def _push_dispatch(self) -> None:
        request_instance_list = self.request_scheduler.need_dispatch()
        for (request_id, instance_id) in request_instance_list:
            self.request_instance[request_id] = instance_id
            # logger.info(f"push request {request_id} to instance {instance_id}")
            session_id, prompt, sampling_params, arrival_time, prompt_token_ids = self.requests[request_id]
            self.instances[instance_id].add_request(request_id, prompt, sampling_params, prompt_token_ids, arrival_time=arrival_time)

    def step(self, timestamp, request_id):
        num_instance_now = len(self.instances)
        if self.scaling_up and self.scale_up_time != -1:# scaling up
            num_instance_now += 1
        instance_id = self.request_instance[request_id]
        post_process_args = self.instances[instance_id].step(timestamp)
        return post_process_args

    def _update_instance_info(self, timestamp, instance_info: InstanceInfo):
        instance_id = instance_info.instance_id
        step_id = instance_info.step_id
         # finish scale down
        if self.scaling_down and instance_id == self.num_instance - 1 and self.instances[instance_id].is_empty():
            self.num_instance -= 1
            self.scaling_down = False
        instance_info = self.request_scheduler.update_instance_info(instance_info)
        if self.instance_step[instance_id] < step_id:
            self.num_instance_info_update += 1
        # after update instance info, push request_scheduler to progress dispatch
        if self.dispatch_mode == 'global' and self.num_instance_info_update != 0 \
            and self.num_instance_info_update % self.need_dispatch_frequency == 0:
            self._push_dispatch()
        # Call migrate when the instance_info updates reaches a certain number of times.
        if self.enable_migrate and self.num_instance_info_update != 0 and self.num_instance_info_update % (self.num_instance * self.need_migrate_frequency) == 0:
            migrate_instance_pairs = self.request_scheduler.need_migrate()
            self._migrate(migrate_instance_pairs)
        if self.enable_scaling and timestamp - self.last_check_scale_time > self.scaling_interval:
            self.last_check_scale_time = timestamp
            scale_up_num, scale_down_num = self.request_scheduler.need_scale()
            scale_up_num = min(scale_up_num, self.max_replicas - self.num_instance)
            if scale_up_num and not self.scaling_up and self.num_instance + scale_up_num <= self.max_replicas:
                if self.scaling_down:
                    self._terminate_scaling_down()
                else:
                    self.to_scale_up_num = scale_up_num
                    self.scaling_up = True
                    self.scale_up_time = timestamp + 5
            if scale_down_num and not self.scaling_down and self.num_instance - scale_down_num >= self.min_replicas:
                self.scaling_down = True
                self.scale_down_time = timestamp

        if self.record_instance_info:
            self._record_instance_info_to_csv(instance_info, self.num_instance)
        if self.scale_up_time != -1 and timestamp > self.scale_up_time:
            logger.info(f"time:{timestamp},scale up")
            self.last_check_scale_time = timestamp
            self._scale_up()
        if self.scale_down_time != -1 and timestamp > self.scale_down_time:
            self.last_check_scale_time = timestamp
            logger.info(f"time:{timestamp},scale down")
            self._scale_down()
    def _terminate_scaling_down(self):
        self.scaling_down = False
        self.request_scheduler.scale_up(1)

    def _scale_up(self):
        for _ in range(self.to_scale_up_num):
            instance_id = self.num_instance
            self.num_instance += 1
            self.instance_step[instance_id] = -1
        self.request_scheduler.scale_up(self.to_scale_up_num)
        self.scale_up_time = -1
        self.scaling_up = False

    def _scale_down(self):
        scale_down_instance_id = self.num_instance - 1
        if self.enable_migrate:
            migrate_list = self.instances[scale_down_instance_id].scale_down_migrate(self.instances)
            for request_id, migarte_in_instance_id in migrate_list:
                self.request_instance[request_id] = migarte_in_instance_id
        self.request_scheduler.scale_down()
        self.scale_down_time = -1
        if self.instances[scale_down_instance_id].is_empty():
            self.num_instance -= 1
            self.scaling_down = False

    def _migrate(self, migrate_instance_pairs: List[Tuple[int, int]]):
        for i in range(len(migrate_instance_pairs)):
            migrate_out_instance_id, migrate_in_instance_id = migrate_instance_pairs[i]
            # migrate_in_instance_name = self.instances[migrate_in_instance_id].instance_name
            # migrate_rank_offset = self.instance_ranks[migrate_in_instance_id][0] - self.instance_ranks[migrate_out_instance_id][0]
            migrating_requests = self.instances[migrate_out_instance_id].migrate_out(self.instances[migrate_in_instance_id])
            for request_id in migrating_requests:
                self.request_instance[request_id] = migrate_in_instance_id

    def _init_instances(self, engine_args: EngineManagerArgs):
        instance_parallel_size = self.parallel_config.instance_parallel_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        self.instance_workers: Dict[int, List[int]] = defaultdict(list)
        self.instance_ranks: Dict[int, List[int]] = defaultdict(list)
        # for worker_id in range(len((self.workers))):
        #     rank = self.workers_rank[worker_id]
        #     instance_id = int(rank % (instance_parallel_size * tensor_parallel_size) / tensor_parallel_size)
        #     self.instance_workers[instance_id].append(worker_id)
        #     self.instance_ranks[instance_id].append(rank)
        
        # # Sort instance workers and ranks for consistency.
        # for instance_id in range(instance_parallel_size):
        #     self.instance_workers[instance_id].sort(key=lambda worker_id:self.workers_rank[worker_id])
        # for instance_id in range(instance_parallel_size):
        #     self.instance_workers[instance_id] = [self.workers[worker_id] for worker_id in self.instance_workers[instance_id]]
        # for instance_id in range(instance_parallel_size):
        #     self.instance_ranks[instance_id].sort()

        engine_configs = engine_args.create_engine_configs()
        self.instances: List[LLMEngine] = []
        # if not self.engine_use_ray:
        #     engine_class = AsyncLLMEngine
        # elif self.worker_use_ray:
        #     engine_class = ray.remote(num_cpus=1)(AsyncLLMEngine).remote
        # else:
        #     engine_class = ray.remote(num_gpus=1)(AsyncLLMEngine).remote
        database = ProfilingDatabase(self.profiling_result_file_path, False)
        self.profiling_result = database.get("llama-7b")
        engine_class = LLMEngine
        for instance_id in range(self.max_replicas):
            self.instances.append(engine_class(
                instance_id,
                engine_configs[0],
                engine_configs[1],
                engine_configs[2],
                engine_configs[3],
                self.profiling_result,
                "a10",
                False,
                ))

    @classmethod
    def from_engine_args(cls,
                         engine_args: EngineManagerArgs) -> "LLMEngineManager":
        engine_configs = engine_args.create_engine_configs()
        request_scheduler_config = engine_args.create_engine_manager_configs()
        parallel_config = engine_configs[2]
        # distributed_init_method, placement_group = initialize_cluster(
        #     parallel_config, engine_args.engine_use_ray)
        engine_manager = cls(engine_args,
                             parallel_config,
                             request_scheduler_config,
                             log_requests=not engine_args.disable_log_requests_manager)
        return engine_manager
    
    def is_ready(self):
        return True

    def _init_instance_info_csv(self, engine_args):
        self.instance_info_file = open(engine_args.results_filename+f'_instance.csv','w')
        self.instance_info_csv = csv.writer(self.instance_info_file)
        self.instance_info_csv.writerow([
            'timestamp',
            'instance_id',
            'step_id',
            'gpu_cache_usage',
            'instance_load',
            'max_tot_tokens',
            'num_running_request',
            'num_waiting_request',
            'num_killed_request',
            'inference_type',
            'bs',
            'latency',
            'seq_lens',
            'num_instance',
            'num_seq',
            'num_block_first_waiting_request',
            'num_block_all_waiting_request',
            'waiting_time_first_waiting_request'])

    def _init_req_info_csv(self, engine_args):
        self.req_info_file = open(engine_args.results_filename+f'_req.csv','w')
        self.req_info_csv = csv.writer(self.req_info_file)
        self.req_info_csv.writerow([
            'timestamp',
            'instance_id',
            'req_id',
            'event',
        ])

    def _record_instance_info_to_csv(self, instance_info: InstanceInfo, num_instance_now: int):
        self.instance_info_csv.writerow([
            instance_info.timestamp,
            instance_info.instance_id,
            instance_info.step_id,
            instance_info.gpu_cache_usage,
            instance_info.instance_load,
            instance_info.max_tot_tokens,
            instance_info.num_running_request,
            instance_info.num_waiting_request,
            instance_info.num_killed_request,
            instance_info.inference_type,
            instance_info.num_batched_tokens,
            instance_info.latency,
            instance_info.running_seq_lens,
            num_instance_now,
            instance_info.num_seq,
            instance_info.num_block_first_waiting_request,
            instance_info.num_block_all_waiting_request,
            instance_info.waiting_time_first_waiting_request])
        self.instance_info_file.flush()