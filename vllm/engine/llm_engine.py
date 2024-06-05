import asyncio
import time
from functools import partial
from typing import Any, List, Tuple, Optional, TYPE_CHECKING

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.ray_utils import initialize_cluster, ray, RayWorker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceEvent, SequenceGroup, SequenceStatus, SequenceGroupMetadata
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.utils import Counter
from vllm.instance_info import InstanceInfo

if ray:
    from ray.air.util.torch_dist import init_torch_dist_process_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        stage_devices: The list of devices for each stage. Each stage is a list
            of (rank, node_resource, device) tuples.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        instance_id: int,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        workers,
        # distributed_init_method: str,
        # placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"use_dummy_weights={model_config.use_dummy_weights}, "
            f"download_dir={model_config.download_dir!r}, "
            f"use_np_weights={model_config.use_np_weights}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.instance_id = instance_id
        self.workers = workers
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code)
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        # if self.parallel_config.worker_use_ray:
        #     self._init_workers_ray(placement_group)
        # else:
        #     self._init_workers(distributed_init_method)

        self._init_model()
        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)
        self.recv_rayobj = None

        self.step_counter = Counter()

        self.num_migrated_out_request_killed = 0
        self.num_migrated_out_request_running = 0

    # def _init_workers(self, distributed_init_method: str):
    #     # Lazy import the Worker to avoid importing torch.cuda/xformers
    #     # before CUDA_VISIBLE_DEVICES is set in the Worker
    #     from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel

    #     assert self.parallel_config.world_size == 1, (
    #         "Ray is required if parallel_config.world_size > 1.")

    #     self.workers: List[Worker] = []
    #     worker = Worker(
    #         self.model_config,
    #         self.parallel_config,
    #         self.scheduler_config,
    #         0,
    #         distributed_init_method,
    #     )
    #     self.workers.append(worker)
    #     self._run_workers(
    #         "init_model",
    #         get_all_outputs=True,
    #     )

    # def _init_workers_ray(self, placement_group: "PlacementGroup"):
    #     # Lazy import the Worker to avoid importing torch.cuda/xformers
    #     # before CUDA_VISIBLE_DEVICES is set in the Worker
    #     from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel

    #     self.workers: List[Worker] = []
    #     for bundle in placement_group.bundle_specs:
    #         if not bundle.get("GPU", 0):
    #             continue
    #         worker = ray.remote(
    #             num_cpus=0,
    #             num_gpus=1,
    #             scheduling_strategy=PlacementGroupSchedulingStrategy(
    #                 placement_group=placement_group,
    #                 placement_group_capture_child_tasks=True),
    #         )(RayWorker).remote()
    #         self.workers.append(worker)

    #     # Initialize torch distributed process group for the workers.
    #     init_torch_dist_process_group(self.workers, backend="nccl")
    #     self._run_workers("init_worker",
    #                       get_all_outputs=True,
    #                       worker_init_fn=lambda: Worker(
    #                           self.model_config,
    #                           self.parallel_config,
    #                           self.scheduler_config,
    #                           None,
    #                           None,
    #                       ))
    #     self._run_workers(
    #         "init_model",
    #         get_all_outputs=True,
    #     )

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_model(self) -> None:
        self._run_workers(
            "init_model",
            get_all_outputs=True,
            model_config=self.model_config,
            scheduler_config=self.scheduler_config
        )
    
    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    def _schedule(
            self
        ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs,
                Optional[List[RequestOutput]], InstanceInfo]:
            seq_group_metadata_list, scheduler_outputs, instance_info = self.scheduler.schedule()
            if scheduler_outputs.is_empty():
                return seq_group_metadata_list, scheduler_outputs, [
                    RequestOutput.from_seq_group(seq_group)
                    for seq_group in scheduler_outputs.ignored_seq_groups
                ], instance_info
            return seq_group_metadata_list, scheduler_outputs, None, instance_info

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config)
        # Create the LLM engine.
        engine = cls(engine_configs[0],
                     engine_configs[1],
                     engine_configs[2],
                     engine_configs[3],
                     distributed_init_method,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        priority_type: int = 0,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            seqs.append(seq)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, seqs, sampling_params,
                                  arrival_time, priority_type)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def tokenize(self, prompt: str) -> Tuple[List[int], int]:
        prompt_token_ids = self.tokenizer.encode(prompt)
        num_block = len(prompt_token_ids) / self.cache_config.block_size
        return prompt_token_ids, num_block

    def abort_request(self, request_id: str) -> None:
        """Aborts a request with the given ID.

        Args:
            request_id: The ID of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def migrate_out(self, dst_instance_name: str, rank_offset: int):
        migrating_seq_groups = self.scheduler.get_migrating_seq_groups()
        # if migrating_seq_groups:
        #     logger.info(f"migrate out seq_groups: {migrating_seq_groups[0].get_seqs()[0].prompt}")
        # else:
        #     logger.info(f"migrate out seq_groups: []")
        if not migrating_seq_groups:
            return []
        dst_actor_handle = ray.get_actor(dst_instance_name)
        migrated_mask = ray.get(dst_actor_handle.migrate_in.remote(
            migrating_seq_groups, -rank_offset))
        assert (len(migrated_mask) == len(migrating_seq_groups))
        migrating_requests = []
        for i in range(len(migrating_seq_groups)):
            if migrated_mask[i]:
                request_id = migrating_seq_groups[i].request_id
                # prompt = migrating_seq_groups[i].get_seqs()[0].prompt
                # sampling_params = migrating_seq_groups[i].sampling_params
                # prompt_token_ids = migrating_seq_groups[i].get_seqs[0].data.prompt_token_ids
                migrating_requests.append(request_id)
        # logger.info(f"migrate out len:{len(migrating_requests)}")
        send_blocks = []
        for i, is_migrated in enumerate(migrated_mask):
            seq = migrating_seq_groups[i].get_seqs()[0]
            if not is_migrated:
                if seq.status == SequenceStatus.RUNNING:
                    self.scheduler.running.append(migrating_seq_groups[i])
                else:
                    self.scheduler.waiting.append(migrating_seq_groups[i])
                    self.scheduler.killed.append(migrating_seq_groups[i])
                continue
            if seq.status == SequenceStatus.RUNNING:
                self.num_migrated_out_request_running += 1
                logger.info(f"self.num_migrated_out_request_running: {self.num_migrated_out_request_running}")
            else:
                self.num_migrated_out_request_killed += 1
                logger.info(f"self.num_migrated_out_request_killed: {self.num_migrated_out_request_killed}")
            seq = migrating_seq_groups[i].get_seqs()[0]
            # logger.info(f"seqlen: {seq.get_len()}")
            if seq.status == SequenceStatus.RUNNING:
                # logger.info(f"send seqid:{seq.seq_id}")
                block_table = self.scheduler.block_manager.get_block_table(seq)
                send_blocks.extend(block_table)
            self.scheduler.free_seq(seq, SequenceStatus.FINISHED_MIGRATED)
        # logger.info(f"send_blocks:{send_blocks},len:{len(send_blocks)}")
        if send_blocks:
            # t0 = time.time()
            self._run_workers("send_gpu_cache", rank_offset, send_blocks)
            # pass
            # t1 = time.time()
            # logger.info(f"send_cost:{(t1-t0)*1000}ms")
        return migrating_requests

    def migrate_in(self, migrating_seq_groups: List[SequenceGroup], rank_offset: int) -> List[bool]:
        # logger.info(f"migrate in seq_groups:{migrating_seq_groups[0].get_seqs()[0].prompt},rank offset:{rank_offset}")
        for seq_group in migrating_seq_groups:
            for seq in seq_group.seqs:
                seq.seq_id = next(self.seq_counter)
        migrating_seq_groups = migrating_seq_groups
        migrated_mask = self.scheduler.allocate_migrate_seq_groups(
            migrating_seq_groups)
        recv_blocks = []
        for i, seq_group in enumerate(migrating_seq_groups):
            seq_group.add_event(SequenceEvent.MIGRATE_IN)
            seq = seq_group.get_seqs()[0]
            if migrated_mask[i] and seq.status == SequenceStatus.RUNNING:
                block_table = self.scheduler.block_manager.get_block_table(seq)
                if seq.get_len()%seq.block_size == 1:# last block is empty
                    recv_blocks.extend(block_table[:-1])
                else:
                    recv_blocks.extend(block_table)
                self.scheduler.running.append(seq_group)
        # logger.info(f"recv_blocks: {recv_blocks}, len: {len(recv_blocks)}")
        if recv_blocks:
            self.recv_rayobj = self._run_workers(
                "recv_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=recv_blocks, rank_offset=rank_offset)
            # pass
        return migrated_mask
    
    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        t00 = time.time()
        # if dst_instance_name:
        #     t0 = time.time()
        #     self.migrate_out(dst_instance_name, rank_offset)
        #     t1 = time.time()
        #     logger.info(f"total_migrate cost:{(t1-t0)*1000}ms")
        (seq_group_metadata_list, scheduler_outputs,
         early_return, instance_info) = self._schedule()
        if early_return is not None:
            return early_return, instance_info

        if self.recv_rayobj:
            ray.get(self.recv_rayobj)
            self.recv_rayobj = None
        # Execute the model.
        t0 = time.time()
        t0_inference_begin = time.time()
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # logger.info(f"inference cost:{(t1-t0)*1000}ms")
        # Update the scheduler with the model outputs.
        seq_groups = self.scheduler.update(output)

        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # logger.info(f"decode time{(t1-t0)*1000}ms")
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        t1_inference_end = time.time()
        self._update_sequences_time(seq_groups, t0_inference_begin, t1_inference_end)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups + scheduler_outputs.ignored_seq_groups:
            if seq_group.is_finished():
                seq_group.add_event(SequenceEvent.FINISHED)
            else:
                instance_info.max_tot_tokens += seq_group.sampling_params.max_tokens
            request_output = RequestOutput.from_seq_group(seq_group)
            request_output.instance_id = self.instance_id
            # logger.info(request_output.outputs)
            request_outputs.append(request_output)
        # update instance info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.latency = (t1_inference_end - t0_inference_begin) * 1000
        
        if len(seq_groups):
            tot_blocks = []
            for seq in seq_groups[-1].get_seqs(SequenceStatus.RUNNING):
                blocks = self.scheduler.block_manager.get_block_table(seq)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_block_last_running_request = len(tot_blocks)

        return request_outputs, instance_info

    def _decode_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Decodes the sequence outputs."""
        for seq_group in seq_groups:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                new_token, new_output_text = detokenize_incrementally(
                    self.tokenizer,
                    seq.output_tokens,
                    seq.get_last_token_id(),
                    skip_special_tokens=True,
                )
                if new_token is not None:
                    seq.output_tokens.append(new_token)
                    seq.output_text = new_output_text

    def _stop_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Stop the finished sequences."""
        for seq_group in seq_groups:
            sampling_params = seq_group.sampling_params
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Check if the sequence has generated a stop string.
                stopped = False
                for stop_str in sampling_params.stop:
                    if seq.output_text.endswith(stop_str):
                        # Truncate the output text so that the stop string is
                        # not included in the output.
                        seq.output_text = seq.output_text[:-len(stop_str)]
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED)
                        stopped = True
                        break
                if stopped:
                    continue

                # Check if the sequence has reached max_seq_len.
                if seq.get_len() > self.scheduler_config.max_model_len:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                # Check if the sequence has reached max_tokens.
                if seq.get_output_len() == sampling_params.max_tokens:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                # Check if the sequence has generated the EOS token.
                if not sampling_params.ignore_eos:
                    if seq.get_last_token_id() == self.tokenizer.eos_token_id:
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED)
                        continue

    def _update_sequences_time(self, seq_groups: List[SequenceGroup], t0_inference_begin, t1_inference_end) -> None:
        for seq_group in seq_groups:
            seq_group.total_inference_time += t1_inference_end - t0_inference_begin

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        get_async_outputs: bool = False,
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
            if not get_async_outputs:
                all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        # for other_output in all_outputs[1:]:
        #     assert output == other_output
        return output

    def is_ready(self):
        return True

class AsyncActorLLMEngine(LLMEngine):
    def __init__(self, instance_id ,*args,**kwargs):
        super(AsyncActorLLMEngine, self).__init__(instance_id, *args, **kwargs)
        self.recv_fut = None
        self.scaling_down = False
    
    async def stop_shutdown(self):
        self.scaling_down = False

    async def shutdown_workers(self, do_migrate:bool = False):
        migrated_requests = []
        # if do_migrate:
        #     migrating_seq_group_list=[]
        #     while self.scheduler.running or self.scheduler.waiting:
        #         seq_group = self.scheduler.running.pop() if self.scheduler.running else self.scheduler.waiting.pop()
        #         migrating_seq_group_list.append(seq_group)
        #         seq = seq_group.get_seqs()[0]
        #         if seq.status == SequenceStatus.RUNNING:
        #             self.scheduler.free_seq(seq, SequenceStatus.FINISHED_MIGRATED)
        #     for instance_idx in range(self.instance_id):
        #         dst_instance_name = f"instance_{instance_idx}"
        #         migrating_seq_groups = []
        #         for idx in range(instance_idx,len(migrating_seq_group_list), self.instance_id):
        #             migrating_seq_groups.append(migrating_seq_group_list[idx])
        #             migrated_requests.append((migrating_seq_group_list[idx].request_id, instance_idx))
        #         dst_actor_hadel = ray.get_actor(dst_instance_name)
        #         await dst_actor_hadel.migrate_in_waiting.remote(migrating_seq_groups)

        self.scaling_down = True
        while (self.has_unfinished_requests() or self.recv_fut) and self.scaling_down:
            await asyncio.sleep(1)
        await asyncio.sleep(0.1)
        if self.scaling_down:
            await self._run_workers(
                "shutdown",
                get_async_outputs=True)
            # self.scaling_down = False
        return migrated_requests

    async def restart_workers(self):
        await self._run_workers(
            "restart",
            get_async_outputs=True)
        self.scaling_down = False

    async def migrate_in_waiting(self, migrating_seq_groups: List[SequenceGroup]):
        for seq_group in migrating_seq_groups:
            for seq in seq_group.seqs:
                seq.seq_id = next(self.seq_counter)
            self.scheduler.waiting.append(seq_group)

    async def migrate_out_async(self, dst_instance_name: str, rank_offset: int) -> List[str]:
        # t00 = time.time()
        migrating_seq_groups = self.scheduler.get_migrating_seq_groups()
        if not migrating_seq_groups:
            return []
        dst_actor_hadel = ray.get_actor(dst_instance_name)
        migrated_mask = await dst_actor_hadel.migrate_in_async.remote(
            migrating_seq_groups, -rank_offset, self.is_scheduled)
        assert (len(migrated_mask) == len(migrating_seq_groups))
        migrating_requests = []
        for i in range(len(migrating_seq_groups)):
            if migrated_mask[i]:
                request_id = migrating_seq_groups[i].request_id
                # prompt = migrating_seq_groups[i].get_seqs()[0].prompt
                # sampling_params = migrating_seq_groups[i].sampling_params
                # prompt_token_ids = migrating_seq_groups[i].get_seqs[0].data.prompt_token_ids
                migrating_requests.append(request_id)
        send_blocks = []
        free_block_seqs = []
        for i, is_migrated in enumerate(migrated_mask):
            seq = migrating_seq_groups[i].get_seqs()[0]
            if not is_migrated:
                if seq.status == SequenceStatus.RUNNING:
                    self.scheduler.running.append(migrating_seq_groups[i])
                else:
                    self.scheduler.waiting.append(migrating_seq_groups[i])
                    self.scheduler.killed.append(migrating_seq_groups[i])
                continue
            if seq.status == SequenceStatus.RUNNING:
                self.num_migrated_out_request_running += 1
                logger.info(f"self.num_migrated_out_request_running: {self.num_migrated_out_request_running}")
            else:
                self.num_migrated_out_request_killed += 1
                logger.info(f"self.num_migrated_out_request_killed: {self.num_migrated_out_request_killed}")
            # logger.info(f"seqlen:{seq.get_len()}")
            if seq.status == SequenceStatus.RUNNING:
                # logger.info(f"send seqid:{seq.seq_id}")
                block_table = self.scheduler.block_manager.get_block_table(seq)
                send_blocks.extend(block_table)
                free_block_seqs.append(seq)
        if send_blocks:
            # logger.info(f"send_blocks:{send_blocks},len:{len(send_blocks)}")
            # t0 = time.time()
            await asyncio.gather(*self._run_workers("send_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=send_blocks, rank_offset=rank_offset))
            for seq in free_block_seqs:
                self.scheduler.free_seq(seq, SequenceStatus.FINISHED_MIGRATED)
            # t1 = time.time()
            # logger.info(f"send_cost:{(t1-t0)*1000}ms")
        # t11 = time.time()
        # logger.info(f"tot_migrate_cost:{(t11-t00)*1000}ms")
        return migrating_requests

    async def migrate_in_async(self, migrating_seq_groups: List[SequenceGroup], rank_offset: int, is_scheduled: bool) -> List[bool]:
        # logger.info(f"migrate in seq_groups:{migrating_seq_groups[0].get_seqs()[0].prompt},rank offset:{rank_offset}")
        for seq_group in migrating_seq_groups:
            for seq in seq_group.seqs:
                seq.seq_id = next(self.seq_counter)
        migrated_mask = self.scheduler.allocate_migrate_seq_groups(
            migrating_seq_groups)
        recv_blocks = []
        migrate_in_running_seqs = []
        for i, seq_group in enumerate(migrating_seq_groups):
            seq_group.add_event(SequenceEvent.MIGRATE_IN)
            seq = seq_group.get_seqs()[0]
            if migrated_mask[i] and seq.status == SequenceStatus.RUNNING:
                block_table = self.scheduler.block_manager.get_block_table(seq)
                migrate_in_running_seqs.append(migrating_seq_groups[i])
                # FIXME maybe this seq is newly migrate in
                if not is_scheduled and seq.get_len() % seq.block_size == 1:# last block is empty
                    recv_blocks.extend(block_table[:-1])
                else:
                    recv_blocks.extend(block_table)
        # logger.info(f"recv_blocks:{recv_blocks},len:{len(recv_blocks)}")
        if recv_blocks:
            fut = asyncio.gather(*self._run_workers(
                "recv_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=recv_blocks, rank_offset=rank_offset))
            def add_to_running(future):
                for seq_group in migrate_in_running_seqs:
                    self.scheduler.running.append(seq_group)
            fut.add_done_callback(add_to_running)
            self.recv_fut = fut
        return migrated_mask

    async def migrate_out_multistage(self, dst_instance_name: str, rank_offset: int) -> List[str]:
        # t0 = time.time()
        migrating_seq_groups = self.scheduler.get_migrating_seq_groups(remove_seq=False)
        if not migrating_seq_groups:
            return []
        # get tot blocks(exp last block)
        seq_group = migrating_seq_groups[0]
        #TODO handle the multi seqs(beam search etc.)
        seq = seq_group.get_seqs()[0]
        blocks = self.scheduler.block_manager.get_block_table(seq)
        block_table = self.scheduler.block_manager.block_tables[seq.seq_id]
        send_blocks = blocks[:-1]
        # logger.info(f"{t0}: {self.instance_id} migrate_out:{len(send_blocks)}, seq_id:{seq.seq_id}")
        # check if migrate in instace have enough blocks
        dst_actor_hadel = ray.get_actor(dst_instance_name)
        recv_block_table = await dst_actor_hadel.migrate_in_stage_0.remote(
            len(send_blocks), -rank_offset)
        # logger.info(f"recv_block_table:{len(recv_block_table)}")
        if len(recv_block_table) != len(send_blocks):
            return []
        # t00 = time.time()
        # print(f"meta cost{t00-t0}")
        if send_blocks :
            await asyncio.gather(*self._run_workers("send_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=send_blocks, rank_offset=rank_offset))
        # t1 = time.time()
        # logger.info(f"{t1}: stage 0 cost:{t1-t0}")
        new_block_num = 0
        migrating_seq_group = None
        new_append_blocks = []
        now_block_table = self.scheduler.block_manager.block_tables.get(seq.seq_id)
        # check is seq_group be killed or finished during send process
        if seq_group in self.scheduler.running and id(block_table) == id(now_block_table):
            self.scheduler.running.remove(seq_group)
            migrating_seq_group = seq_group
            blocks = self.scheduler.block_manager.get_block_table(seq)
            new_block_num = len(blocks) - len(send_blocks)
            new_append_blocks = blocks[-new_block_num:]
        # logger.info(f"new_append_blocks:{new_append_blocks}")
        is_success = await dst_actor_hadel.migrate_in_stage_1.remote(len(new_append_blocks), recv_block_table, migrating_seq_group, -rank_offset)
        # logger.info(f"is_success:{is_success}, {seq.seq_id}")
        migrating_requests = []
        if not is_success:
            if migrating_seq_group:
                self.scheduler.running.append(seq_group)
        else:
            self.scheduler.free_seq(seq, SequenceStatus.FINISHED_MIGRATED)
            if new_append_blocks:
                await self._run_workers("send_gpu_cache", get_async_outputs=True, block_tensor=new_append_blocks, rank_offset=rank_offset)
            migrating_requests.append(seq_group.request_id)
        # t2 = time.time()
        # logger.info(f"stage 1 cost:{t2-t1}")
        return migrating_requests

    async def migrate_in_stage_0(self, block_num:int, rank_offset:int):
        if block_num == 0:
            return []
        block_table = self.scheduler.block_manager.get_free_blocks(block_num)
        if block_table:
            recv_blocks = [block.block_number for block in block_table]
            asyncio.gather(*self._run_workers(
                "recv_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=recv_blocks, rank_offset=rank_offset))
            return block_table
        else:
            return []

    async def migrate_in_stage_1(self, new_block_num:int, recv_block_table:List[int], migrating_seq_group:SequenceGroup, rank_offset:int) -> bool:
        block_table = self.scheduler.block_manager.get_free_blocks(new_block_num)
        # seq is finished or fail to allocate block, release block_table
        if migrating_seq_group == None or len(block_table)==0:
            self.scheduler.block_manager._free_block_table(recv_block_table)
            return False
        recv_block_table.extend(block_table)
        seq = migrating_seq_group.get_seqs()[0]
        seq.seq_id = next(self.seq_counter)
        self.scheduler.block_manager.add_block_table(recv_block_table, seq.seq_id)
        if block_table:
            recv_blocks = [block.block_number for block in block_table]
            fut = asyncio.gather(*self._run_workers(
                "recv_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=recv_blocks, rank_offset=rank_offset))
            self.recv_fut = fut
            def recv_done_callback(future):
                migrating_seq_group.add_event(SequenceEvent.MIGRATE_IN)
                self.scheduler.running.append(migrating_seq_group)
                self.recv_fut = None
            fut.add_done_callback(recv_done_callback)
        return True
            
    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        t00 = time.time()
        (seq_group_metadata_list, scheduler_outputs,
         early_return, instance_info) = self._schedule()
        self.is_scheduled = True
        if self.scaling_down:
            instance_info.num_running_request = 1
            instance_info.num_available_gpu_block = -1000
            instance_info.num_available_gpu_block_waiting = -1000
        if early_return is not None:
            self.is_scheduled = False
            return early_return, instance_info
        
        # Execute the model.
        t0_inference_begin = time.time()
        output = await self._run_workers(
            "execute_model",
            get_async_outputs=True,
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # Update the scheduler with the model outputs.
        seq_groups = self.scheduler.update(output)
        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        t1_inference_end = time.time()
        self._update_sequences_time(seq_groups, t0_inference_begin, t1_inference_end)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()
        self.is_scheduled = False
        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups + scheduler_outputs.ignored_seq_groups:
            if seq_group.is_finished():
                seq_group.add_event(SequenceEvent.FINISHED)
            else:
                instance_info.max_tot_tokens += seq_group.sampling_params.max_tokens
            request_output = RequestOutput.from_seq_group(seq_group)
            request_output.instance_id = self.instance_id
            # logger.info(request_output.outputs)
            request_outputs.append(request_output)
        t11 = time.time()
        # logger.info(f"step time:{(t11-t00)*1000}ms")
        # update instance info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.latency = (t1_inference_end - t0_inference_begin)*1000

        if len(seq_groups):
            tot_blocks = []
            for seq in seq_groups[-1].get_seqs(SequenceStatus.RUNNING):
                blocks = self.scheduler.block_manager.get_block_table(seq)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_block_last_running_request = len(tot_blocks)

        return request_outputs, instance_info