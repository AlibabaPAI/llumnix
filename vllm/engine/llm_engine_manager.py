import asyncio
import time
import csv
from functools import partial
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict

from vllm.config import ParallelConfig, RequestSchedulerConfig
from vllm.engine.arg_utils import EngineManagerArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, EngineRequestInput
from vllm.engine.ray_utils import initialize_cluster, ray, RayWorker
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.core.request_scheduler import RequestScheduler
from vllm.instance_info import InstanceInfo
from vllm.logger import init_logger
from vllm.sequence import SequenceEvent

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
                 distributed_init_method: str,
                 placement_group: Optional["PlacementGroup"],
                 log_requests: bool = True) -> None:
        self.engine_use_ray = engine_args.engine_use_ray
        self.worker_use_ray = engine_args.worker_use_ray
        self.enable_migrate = engine_args.enable_migrate
        self.async_migrate = engine_args.async_migrate
        logger.info(f"self.enable_migrate: {self.enable_migrate}")
        logger.info(f"self.async_migrate: {self.async_migrate}")
        self.parallel_config = parallel_config
        self.log_requests = log_requests
        self.record_instance_info = True
        self.record_req_info = True
        self.scale_up_time = -1
        self.scale_down_time = -1
        self.scaling_up = False
        self.scaling_down = False
        self.max_replicas = engine_args.max_replicas
        self.min_replicas = engine_args.min_replicas
        self.enable_scaling = engine_args.enable_scaling
        self.engine_args = engine_args
        self.scaling_interval = engine_args.scaling_interval
        self.last_check_scale_time = time.time() + 100
        self.last_update_instance_info_time: Dict[int, int] = {}
        if not self.enable_scaling:
            self.max_replicas = self.min_replicas = self.parallel_config.instance_parallel_size

        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)
        self._init_instances(engine_args)
        self.dispatch_mode = engine_args.dispatch_mode
        request_scheduler_class = ray.remote(num_cpus=0)(RequestScheduler).remote
        # request_scheduler_class = RequestScheduler
        enable_global_dispatch = (self.dispatch_mode == 'global')
        self.generate_mode = 'callback' if not enable_global_dispatch else 'global'
        self.request_scheduler = request_scheduler_class(request_scheduler_config, enable_global_dispatch, self.enable_migrate)
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
        for instance_id in range(self.max_replicas):
            self.instance_step[instance_id] = -1
            self.last_update_instance_info_time[instance_id] = time.time()
        self.num_instance_info_update = 0
        self.need_migrate_frequency = engine_args.need_migrate_frequency

        self.tokenize_instance_ptr = 0
        self.async_lock = asyncio.Lock()
        self.need_dispatch_frequency = engine_args.need_dispatch_frequency

        if self.record_instance_info:
            self._init_instance_info_csv(engine_args)
        if self.record_req_info:
            self._init_req_info_csv(engine_args)
        # shutdown supertype instances
        for instance_id in range(self.num_instance, self.max_replicas):
            ray.get(self.instances[instance_id].engine.shutdown_workers.remote())
        self.migrating = False

    def start_backgroup_loop(self, loop) -> None:
        loop.create_task(self.background_loop())

    async def generate(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            priority_type: int = 0) -> RequestOutput:
        if self.log_requests:
            logger.info(f"Received request {request_id}.")
            # logger.info(f"Received request {request_id}: "
            #             f"prompt: {prompt!r}, "
            #             f"sampling params: {sampling_params}, "
            #             f"prompt token ids: {prompt_token_ids}.")
        arrival_time = time.time()
        if self.generate_mode == 'original':
            outputs_generator = self.generate_original(session_id, prompt, sampling_params, request_id, prompt_token_ids, arrival_time)
        elif self.generate_mode == 'loop':
            outputs_generator = self.generate_loop(session_id, prompt, sampling_params, request_id, prompt_token_ids, arrival_time)
        elif self.generate_mode == 'callback':
            outputs_generator = self.generate_callback(session_id, prompt, sampling_params, request_id, prompt_token_ids, arrival_time, priority_type=priority_type)
        elif self.generate_mode == 'global':
            outputs_generator = self.generate_global(session_id, prompt, sampling_params, request_id, prompt_token_ids, arrival_time)
        async for request_output in outputs_generator:
            yield request_output

    async def generate_original(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None) -> RequestOutput:
        instance_id = await self.request_scheduler.dispatch.remote(session_id)
        # instance_id = self.request_scheduler.dispatch(session_id)
        self.request_instance[request_id] = instance_id
        # if self.engine_use_ray:
        #     outputs_generator = self.instances[instance_id].generate.remote(prompt, sampling_params, request_id)
        # else:
        #     outputs_generator = self.instances[instance_id].generate(prompt, sampling_params, request_id)
        outputs_generator = self.instances[instance_id].generate(prompt, sampling_params, request_id, prompt_token_ids)
        async for request_output, instance_info in outputs_generator:
            yield request_output
            asyncio.create_task(self._update_instance_info(instance_info))

    async def generate_loop(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None) -> RequestOutput:
        self.request_events[request_id] = asyncio.Event()
        self.request_outputs[request_id] = None
        self.add_request(session_id, prompt, sampling_params, request_id, prompt_token_ids, arrival_time)
        asyncio.create_task(self.progress(is_engine_put=True))
        outputs_generator = self.get_request_output_loop(request_id)
        async for request_output in outputs_generator:
            yield request_output

    async def generate_callback(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None,
            priority_type: int = 0) -> RequestOutput:
        self.request_events[request_id] = asyncio.Event()
        self.request_outputs[request_id] = None
        instance_id = await self.request_scheduler.dispatch.remote(session_id, priority_type)
        # instance_id = self.request_scheduler.dispatch(session_id)
        self.request_instance[request_id] = instance_id
        asyncio.create_task(self.instances[instance_id].generate_callback(prompt, sampling_params, request_id, prompt_token_ids, priority_type))
        outputs_generator = self.get_request_output_callback(request_id)
        async for request_output in outputs_generator:
            yield request_output
    
    async def generate_global(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None) -> RequestOutput:
        self.request_events[request_id] = asyncio.Event()
        self.request_outputs[request_id] = None
        # add request to request_scheduler and push request_scheduler to dispatch
        asyncio.create_task(self.add_request_global(session_id, prompt, sampling_params, request_id, prompt_token_ids, arrival_time))
        outputs_generator = self.get_request_output_callback(request_id)
        async for request_output in outputs_generator:
            yield request_output

    async def _push_dispatch(self) -> None:
        request_instance_list = await self.request_scheduler.need_dispatch.remote()
        for (request_id, instance_id) in request_instance_list:
            # logger.info(f"Dispatch request {request_id} to instance {instance_id}")
            self.request_instance[request_id] = instance_id
            session_id, prompt, sampling_params, arrival_time, prompt_token_ids = self.requests[request_id]
            asyncio.create_task(self.instances[instance_id].generate_callback(prompt, sampling_params, request_id, prompt_token_ids))

    def add_request(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None) -> None:
        self.requests[request_id] = (session_id, prompt, sampling_params, arrival_time, prompt_token_ids)
        self.request_queue.put_nowait(request_id)
    
    async def add_request_global(
            self,
            session_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: float = None) -> None:
        # dispatch request to engine to tokenize prompt
        async with self.async_lock:
            instance_id = self.tokenize_instance_ptr
            self.tokenize_instance_ptr = (self.tokenize_instance_ptr + 1) % self.num_instance
        prompt_token_ids, num_block = await self.instances[instance_id].tokenize(prompt)
        # logger.info(f"Tokenize request {request_id}, num_block: {num_block}.")
        self.requests[request_id] = (session_id, prompt, sampling_params, arrival_time, prompt_token_ids)
        await self.request_scheduler.add_request.remote(request_id, session_id, num_block)
        # push request_scheduler to dispatch request once new request arrives and has been added to request_scheduler
        # push by manager rather than request_scheduler in order to control push dispatch frequency
        asyncio.create_task(self._push_dispatch())

    async def progress(self, is_engine_put=False, get_instance_id=None) -> None:
        if is_engine_put:
            while not self.request_queue.empty():
                request_id = self.request_queue.get_nowait()
                session_id, prompt, sampling_params, arrival_time, prompt_token_ids = self.requests[request_id]
                request_input = EngineRequestInput(request_id, prompt, sampling_params, arrival_time, prompt_token_ids)
                instance_id = await self.request_scheduler.dispatch.remote(session_id)
                # instance_id = self.request_scheduler.dispatch(session_id)
                self.request_instance[request_id] = instance_id
                self.instances[instance_id].input_queue_put_nowait.remote(request_input)
        else:
            while not await self.instances[get_instance_id].output_queue_empty.remote():
                request_output, instance_info = await self.instances[get_instance_id].output_queue_get_nowait.remote()
                request_id = request_output.request_id
                self.request_outputs[request_id] = request_output
                asyncio.create_task(self._update_instance_info(instance_info))
                self.request_events[request_id].set()
    
    async def progress_output(self, get_instance_id=None) -> None:
        while not await self.instances[get_instance_id].output_queue_empty():
            request_output, instance_info = await self.instances[get_instance_id].output_queue_get_nowait()
            request_id = request_output.request_id
            if get_instance_id != self.request_instance[request_id]:
                if request_id not in self.migrated_requests:
                    self.migrated_requests.append(request_id)
                    self.num_migrated_request += 1
                    logger.info(f"self.num_migrated_request {self.num_migrated_request}.")
            self.request_outputs[request_id] = request_output
            if instance_info.timestamp != self.last_update_instance_info_time[get_instance_id]:
                self.last_update_instance_info_time[get_instance_id] = instance_info.timestamp
                asyncio.create_task(self._update_instance_info(instance_info))
            self.request_events[request_id].set()

    async def progress_callback(self, instance_id) -> None:
        await self.progress_output(get_instance_id=instance_id)
    
    async def background_loop(self) -> None:
        while True:
            await asyncio.sleep(0.01)
            for instance_id in range(self.num_instance):
                asyncio.create_task(self.progress(is_engine_put=False, get_instance_id=instance_id))

    async def get_request_output_loop(self, request_id) -> RequestOutput:
        while True:
            await self.request_events[request_id].wait()
            self.request_events[request_id].clear()
            request_output = self.request_outputs[request_id]
            yield request_output
            self.request_events[request_id] = asyncio.Event()
            if request_output.finished:
                del self.requests[request_id]
                del self.request_events[request_id]
                del self.request_outputs[request_id]
                del self.request_instance[request_id]
                break

    async def get_request_output_callback(self, request_id) -> RequestOutput:
        while True:
            await self.request_events[request_id].wait()
            self.request_events[request_id].clear()
            request_output = self.request_outputs[request_id]
            yield request_output
            self.request_events[request_id] = asyncio.Event()
            if request_output.finished:
                if self.record_req_info:
                    self._record_req_info_to_csv(request_output)
                if self.log_requests:
                    logger.info(f"Finished request {request_id}.")
                    self.num_finished_request += 1
                    logger.info(f"self.num_finished_request {self.num_finished_request}.")
                del self.request_events[request_id]
                del self.request_outputs[request_id]
                del self.request_instance[request_id]
                break

    async def abort(self, request_id: str) -> None:
        if self.log_requests:
            logger.info(f"Aborted request {request_id}.")
        
        if self.generate_mode == 'original':
            self.abort_original(request_id)
        elif self.generate_mode == 'loop':
            self.abort_loop(request_id)
        elif self.generate_mode == 'callback':
            self.abort_callback(request_id)
        elif self.generate_mode == 'global':
            self.abort_global(request_id)

    async def abort_original(self, request_id: str) -> None:
        instance_id = self.request_instance[request_id]
        # if self.engine_use_ray:
        #     self.instances[instance_id].abort.remote(request_id)
        # else:
        #     self.instances[instance_id].abort(request_id)
        self.instances[instance_id].abort(request_id)

    async def abort_loop(self, request_id: str) -> None:
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return
        
        instance_id = self.request_instance[request_id]
        if request_id in self.requests:
            del self.requests[request_id]
        if request_id in self.request_instance:
            del self.request_instance[request_id]
        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.request_outputs:
            del self.request_outputs[request_id]

        await self.instances[instance_id].abort_loop.remote(request_id)

    async def abort_callback(self, request_id: str) -> None:
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return
        
        instance_id = self.request_instance[request_id]
        if request_id in self.request_instance:
            del self.request_instance[request_id]
        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.request_outputs:
            del self.request_outputs[request_id]

        await self.instances[instance_id].abort(request_id)

    async def abort_global(self, request_id: str) -> None:
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return
        
        instance_id = self.request_instance[request_id]
        if request_id in self.requests:
            del self.requests[request_id]
        if request_id in self.request_instance:
            del self.request_instance[request_id]
        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.request_outputs:
            del self.request_outputs[request_id]

        await self.instances[instance_id].abort(request_id)
    
    async def _update_instance_info(self, instance_info: InstanceInfo) -> None:
        instance_id = instance_info.instance_id
        step_id = instance_info.step_id
        instance_info = await self.request_scheduler.update_instance_info.remote(instance_info)
        if self.record_instance_info:
                self._record_instance_info_to_csv(instance_info)
        # self.request_scheduler.update_instance_info(instance_info)
        if self.instance_step[instance_id] < step_id:
            self.num_instance_info_update += 1
        # after update instance info, push request_scheduler to progress dispatch
        if self.dispatch_mode == 'global' and self.num_instance_info_update != 0 \
            and self.num_instance_info_update % self.need_dispatch_frequency == 0:
            asyncio.create_task(self._push_dispatch())
        # Call migrate when the instance_info updates reaches a certain number of times.
        if self.enable_migrate and not self.migrating and self.num_instance_info_update != 0 and self.num_instance_info_update % (self.num_instance * self.need_migrate_frequency) == 0:
            self.migrating = True
            migrate_instance_pairs = await self.request_scheduler.need_migrate.remote()
            # migrate_instance_pairs = self.request_migrate_scheduler.need_migrate()
            asyncio.create_task(self._migrate(migrate_instance_pairs))
            self.migrating = False
        # auto-scaling
        timestamp = time.time()
        if self.enable_scaling and timestamp - self.last_check_scale_time > self.scaling_interval:
            self.last_check_scale_time = timestamp
            scale_up_num, scale_down_num = await self.request_scheduler.need_scale.remote()
            scale_up_num = min(scale_up_num, self.max_replicas - self.num_instance)
            scale_down_num = min(scale_down_num, self.num_instance - self.min_replicas)
            if scale_up_num and not self.scaling_up and self.num_instance + scale_up_num <= self.max_replicas:
                if self.scaling_down:
                    asyncio.create_task(self._terminate_scaling_down())
                else:
                    asyncio.create_task(self._scale_up(scale_up_num))
                # task.add_done_callback(scaling_done_callback)
            if scale_down_num and not self.scaling_down and self.num_instance - scale_down_num >= self.min_replicas:
                asyncio.create_task(self._scale_down())
                # task.add_done_callback(scaling_done_callback)

    async def _terminate_scaling_down(self):
        logger.info(f"terminate scale down")
        scale_down_instance_id = self.num_instance - 1
        await self.instances[scale_down_instance_id].stop_shutdown()
        self.scaling_down = False

    async def _scale_up(self, scale_up_num):
        self.scaling_up = True
        logger.info(f"begin scale up instance {range(self.num_instance, self.num_instance+scale_up_num)}")
        scale_up_events = [self.instances[scale_up_instance_id].restart_engine() \
            for scale_up_instance_id in range(self.num_instance, self.num_instance+scale_up_num)]
        await asyncio.gather(*scale_up_events)
        await self.request_scheduler.scale_up.remote(scale_up_num)
        self.num_instance += scale_up_num
        logger.info(f"scale up done")
        self.scaling_up = False
    
    async def _scale_down(self):
        self.scaling_down = True
        scale_down_instance_id = self.num_instance - 1
        logger.info(f"begin scale down instance {scale_down_instance_id}")
        # wait until migrating done
        # while self.instances[scale_down_instance_id].is_engine_migrating:
        #     await asyncio.sleep(0.1)
        migrating_requests_list = await self.instances[scale_down_instance_id].shutdown_engine(do_migrate=self.enable_migrate)
        if self.scaling_down:
            self.scaling_down = False
            await self.request_scheduler.scale_down.remote()
            # for request_id, migrate_in_instance_id in migrating_requests_list:
            #     asyncio.create_task(self.instances[migrate_in_instance_id].generate_migrate(request_id))
            self.num_instance -= 1
            logger.info(f"scale down instance {scale_down_instance_id}")
    
    async def _migrate(self, migrate_instance_pairs: List[Tuple[int, int]]) -> None:
        for i in range(len(migrate_instance_pairs)):
            migrate_out_instance_id, migrate_in_instance_id = migrate_instance_pairs[i]
            if self.instances[migrate_in_instance_id].is_engine_migrating or self.instances[migrate_out_instance_id].is_engine_migrating:
                continue
            logger.info(f"{migrate_out_instance_id}->{migrate_in_instance_id} begin migrate out")
            self.instances[migrate_in_instance_id].is_engine_migrating = True
            self.instances[migrate_out_instance_id].is_engine_migrating = True
            migrate_in_instance_name = self.instances[migrate_in_instance_id].instance_name
            migrate_rank_offset = self.instance_ranks[migrate_in_instance_id][0] - self.instance_ranks[migrate_out_instance_id][0]
            task = asyncio.create_task(self.instances[migrate_out_instance_id].migrate_out(migrate_in_instance_name, migrate_rank_offset))
            def migrate_done_callback(instance_pair, fut):
                migrate_out_request_ids = fut.result()
                logger.info(f"{instance_pair[1]}->{instance_pair[0]} migrate done, migrate request{migrate_out_request_ids}")
                self.instances[instance_pair[0]].is_engine_migrating = False
                self.instances[instance_pair[1]].is_engine_migrating = False
                for request_id in migrate_out_request_ids:
                    asyncio.create_task(self.instances[instance_pair[0]].generate_migrate(request_id))
            callback = partial(migrate_done_callback, (migrate_in_instance_id, migrate_out_instance_id))
            task.add_done_callback(callback)

    def _init_workers(self, distributed_init_method: str) -> None:
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")

        self.workers: List[Worker] = []
        worker = Worker(
            # self.model_config,
            self.parallel_config,
            # self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        self.workers_rank = self._run_workers(
            # "init_model",
            "init_distributed_environment",
            get_all_outputs=True,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup") -> None:
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel

        self.workers: List[RayWorker] = []
        max_concurrency = 2 if self.enable_migrate and self.async_migrate else 1
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                max_concurrency=max_concurrency,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
            )(RayWorker).remote()
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend= None if self.parallel_config.migrate_backend=="gloo" else "nccl")
        self._run_workers("init_worker",
                          get_all_outputs=True,
                          worker_init_fn=lambda: Worker(
                            #   self.model_config,
                              self.parallel_config,
                            #   self.scheduler_config,
                              None,
                              None,
                          ))
        self.workers_rank = self._run_workers(
            # "init_model",
            "init_distributed_environment",
            get_all_outputs=True,
        )

    def _init_instances(self, engine_args: EngineManagerArgs):
        max_replicas = self.max_replicas
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        self.instance_workers: Dict[int, List[int]] = defaultdict(list)
        self.instance_ranks: Dict[int, List[int]] = defaultdict(list)
        for worker_id in range(len((self.workers))):
            rank = self.workers_rank[worker_id]
            instance_id = int(rank % (max_replicas * tensor_parallel_size) / tensor_parallel_size)
            self.instance_workers[instance_id].append(worker_id)
            self.instance_ranks[instance_id].append(rank)
        
        # Sort instance workers and ranks for consistency.
        for instance_id in range(max_replicas):
            self.instance_workers[instance_id].sort(key=lambda worker_id:self.workers_rank[worker_id])
        for instance_id in range(max_replicas):
            self.instance_workers[instance_id] = [self.workers[worker_id] for worker_id in self.instance_workers[instance_id]]
        for instance_id in range(max_replicas):
            self.instance_ranks[instance_id].sort()

        engine_configs = engine_args.create_engine_configs()
        self.instances: List[AsyncLLMEngine] = []
        # if not self.engine_use_ray:
        #     engine_class = AsyncLLMEngine
        # elif self.worker_use_ray:
        #     engine_class = ray.remote(num_cpus=1)(AsyncLLMEngine).remote
        # else:
        #     engine_class = ray.remote(num_gpus=1)(AsyncLLMEngine).remote
        engine_class = AsyncLLMEngine
        for instance_id in range(max_replicas):
            self.instances.append(engine_class(
                instance_id,
                engine_args.worker_use_ray,
                engine_args.engine_use_ray,
                engine_configs[0],
                engine_configs[1],
                engine_configs[2],
                engine_configs[3],
                workers=self.instance_workers[instance_id],
                log_requests=not engine_args.disable_log_requests_engine,
                log_stats=not engine_args.disable_log_stats,
                async_engine_actor = (self.enable_migrate and self.async_migrate) or self.enable_scaling))

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output

        return output

    @classmethod
    def from_engine_args(cls,
                         engine_args: EngineManagerArgs) -> "LLMEngineManager":
        engine_configs = engine_args.create_engine_configs()
        request_scheduler_config = engine_args.create_engine_manager_configs()
        parallel_config = engine_configs[2]
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        engine_manager = cls(engine_args,
                             parallel_config,
                             request_scheduler_config,
                             distributed_init_method,
                             placement_group,
                             log_requests=not engine_args.disable_log_requests_manager)

        return engine_manager

    def register_callback_to_instances(self) -> None:
        for instance_id in range(self.max_replicas):
            self.instances[instance_id].register_callback(self, self.progress_callback)

    async def is_ready(self) -> bool:
        tasks: List[asyncio.Task] = []
        for instance_id in range(self.max_replicas):
            task = asyncio.create_task(self.instances[instance_id].is_ready())
            tasks.append(task)
        is_ready_list = await asyncio.gather(*tasks)

        return all(is_ready_list)

    def _init_instance_info_csv(self, engine_args) -> None:
        self.instance_info_file = open(engine_args.results_filename+f'_instance.csv','w')
        self.instance_info_csv = csv.writer(self.instance_info_file)
        self.instance_info_csv.writerow([
            'timestamp',
            'instance_id',
            'step_id',
            'gpu_cache_usage',
            'num_available_gpu_block',
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
            'waiting_time_first_waiting_request',
            'num_priority_request',])
    
    def _init_req_info_csv(self, engine_args) -> None:
        self.req_info_file = open(engine_args.results_filename+f'_req.csv','w')
        self.req_info_csv = csv.writer(self.req_info_file)
        self.req_info_csv.writerow([
            'timestamp',
            'instance_id',
            'req_id',
            'event',
            'output_len'
        ])
            
    def _record_instance_info_to_csv(self, instance_info: InstanceInfo) -> None:
        self.instance_info_csv.writerow([
            instance_info.timestamp,
            instance_info.instance_id,
            instance_info.step_id,
            instance_info.gpu_cache_usage,
            instance_info.num_available_gpu_block,
            instance_info.instance_load,
            instance_info.max_tot_tokens,
            instance_info.num_running_request,
            instance_info.num_waiting_request,
            instance_info.num_killed_request,
            instance_info.inference_type,
            instance_info.num_batched_tokens,
            instance_info.latency,
            instance_info.running_seq_lens,
            self.num_instance,
            instance_info.num_seq,
            instance_info.num_block_first_waiting_request,
            instance_info.num_block_all_waiting_request,
            instance_info.waiting_time_first_waiting_request,
            instance_info.num_priority_request,])
        self.instance_info_file.flush()

    def _record_req_info_to_csv(self, request_output: RequestOutput) -> None:
        for (timestamp, event) in request_output.event_timeline:
            self.req_info_csv.writerow([
                timestamp,
                request_output.instance_id,
                request_output.request_id,
                SequenceEvent.get_event(event),
                len(request_output.outputs[-1].token_ids),
            ])
        self.req_info_file.flush()
