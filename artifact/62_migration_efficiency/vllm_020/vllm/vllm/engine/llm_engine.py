import asyncio
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.ray_utils import initialize_cluster, ray, RayWorker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceStatus, SequenceEvent)
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

_LOGGING_INTERVAL_SEC = 5


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
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
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
            f"revision={model_config.revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.
        self.instance_id = instance_id
        self.workers = workers
        self.model_config = model_config
        self.cache_config = cache_config
        assert self.cache_config.sliding_window == getattr(
            self.model_config.hf_config, "sliding_window", None)
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision)
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
        self.scheduler = Scheduler(scheduler_config, cache_config)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

    def _init_workers(self, distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")

        self.workers: List[Worker] = []
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel

        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorker).remote(self.model_config.trust_remote_code)
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend=None)
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        self._run_workers("init_worker",
                          get_all_outputs=True,
                          worker_init_fn=lambda: Worker(
                              model_config,
                              parallel_config,
                              scheduler_config,
                              None,
                              None,
                          ))
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

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
        engine = cls(*engine_configs,
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
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
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
            seq.status = SequenceStatus.FINISHED_MIGRATED
            self.scheduler.free_seq(seq)
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
    def _schedule(
        self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs,
               List[RequestOutput]]:
        seq_group_metadata_list, scheduler_outputs, instance_info = self.scheduler.schedule()
        return seq_group_metadata_list, scheduler_outputs, [
            RequestOutput.from_seq_group(seq_group)
            for seq_group in scheduler_outputs.ignored_seq_groups
        ], instance_info

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = (current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id))
        if early_stopping is False:
            highest_attainable_score = (best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=self.tokenizer.eos_token_id))
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.tokenizer.eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.tokenizer.eos_token_id))
        return current_worst_score >= highest_attainable_score

    def _process_sequence_group_samples(
            self, seq_group: SequenceGroup,
            samples: List[SequenceOutputs]) -> None:
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutputs] = parent_child_dict[
                parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token,
                                      child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token,
                                   last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.best_of
        length_penalty = seq_group.sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False)
                                  for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                             if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id),
                               reverse=True)
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                              if not seq.is_finished()]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id),
                                reverse=True)

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params, best_running_seq, current_worst_seq)

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)

    def _process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group in scheduled_seq_groups:
            for samples in output:
                if seq_group.get_seqs()[0].seq_id==samples[0].parent_seq_id:
                    self._process_sequence_group_samples(seq_group, samples)
                    break
        # for seq_group, samples in zip(scheduled_seq_groups, output):
        #     self._process_sequence_group_samples(seq_group, samples)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups +
                          scheduler_outputs.ignored_seq_groups):
            if seq_group.is_finished():
                seq_group.add_event(SequenceEvent.FINISHED)
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(scheduler_outputs.prompt_run,
                                   scheduler_outputs.num_batched_tokens)
        return request_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored, instance_info = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored, instance_info

        # Execute the model.
        t0_inference_begin = time.time()
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        t1_inference_end = time.time()
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.latency = (t1_inference_end - t0_inference_begin) * 1000
        self._update_sequences_time(scheduler_outputs.scheduled_seq_groups, t0_inference_begin, t1_inference_end)
        seq_groups = self.scheduler.get_migrating_seq_groups(remove_seq=False)
        if len(seq_groups):
            tot_blocks = []
            for seq in seq_groups[-1].get_seqs(SequenceStatus.RUNNING):
                blocks = self.scheduler.block_manager.get_block_table(seq)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_block_last_running_request = len(tot_blocks)
        tot_used_blocks_num = 0
        for seq_group in self.running:
            seq = seq_group.get_seqs()[0]
            tot_used_blocks_num += self.scheduler.block_manager.get_block_table(seq)
        logger.info(f"tot block num:{tot_used_blocks_num+self.scheduler.block_manager.get_num_free_gpu_blocks()}")

        return self._process_model_outputs(output, scheduler_outputs) + ignored, instance_info

    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.time()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        elapsed_time = now - self.last_logging_time
        if elapsed_time < _LOGGING_INTERVAL_SEC:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens
                                  if now - t < _LOGGING_INTERVAL_SEC]
        self.num_generation_tokens = [(t, n)
                                      for t, n in self.num_generation_tokens
                                      if now - t < _LOGGING_INTERVAL_SEC]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n
                                   for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (
                self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now

    def _decode_sequence(self, seq: Sequence,
                         sampling_params: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             self.tokenizer,
             all_input_ids=seq.get_token_ids(),
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             skip_special_tokens=sampling_params.skip_special_tokens,
         )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence,
                    sampling_params: SamplingParams) -> None:
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                # Truncate the output text so that the stop string is
                # not included in the output.
                seq.output_text = seq.output_text[:-len(stop_str)]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if seq.get_last_token_id() in sampling_params.stop_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == self.tokenizer.eos_token_id):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return
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
            # if not get_async_outputs:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        # for other_output in all_outputs[1:]:
        #     assert output == other_output
        return output

class AsyncActorLLMEngine(LLMEngine):
    def __init__(self, instance_id ,*args,**kwargs):
        super(AsyncActorLLMEngine, self).__init__(instance_id, *args, **kwargs)
        self.step_counter = Counter()
        self.num_migrated_out_request_killed = 0
        self.num_migrated_out_request_running = 0
        self.recv_fut = None
        self.scaling_down = False

    def stop_shutdown(self):
        self.scaling_down = False

    def shutdown_workers(self, do_migrate:bool = False):
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
            asyncio.sleep(1)
        asyncio.sleep(0.1)
        if self.scaling_down:
            self._run_workers(
                "shutdown",
                get_async_outputs=True)
            # self.scaling_down = False
        return migrated_requests

    def restart_workers(self):
        self._run_workers(
            "restart",
            get_async_outputs=True)
        self.scaling_down = False

    def migrate_in_waiting(self, migrating_seq_groups: List[SequenceGroup]):
        for seq_group in migrating_seq_groups:
            for seq in seq_group.seqs:
                seq.seq_id = next(self.seq_counter)
            self.scheduler.waiting.append(seq_group)

    def migrate_out_async(self, dst_instance_name: str, rank_offset: int) -> List[str]:
        # t00 = time.time()
        migrating_seq_groups = self.scheduler.get_migrating_seq_groups()
        if not migrating_seq_groups:
            return []
        dst_actor_hadel = ray.get_actor(dst_instance_name)
        migrated_mask = ray.get(dst_actor_hadel.migrate_in_async.remote(
            migrating_seq_groups, -rank_offset, self.is_scheduled))
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
            self._run_workers("send_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=send_blocks, rank_offset=rank_offset)
            for seq in free_block_seqs:
                seq.status = SequenceStatus.FINISHED_MIGRATED
                self.scheduler.free_seq(seq)
            # t1 = time.time()
            # logger.info(f"send_cost:{(t1-t0)*1000}ms")
        # t11 = time.time()
        # logger.info(f"tot_migrate_cost:{(t11-t00)*1000}ms")
        return migrating_requests

    def migrate_in_async(self, migrating_seq_groups: List[SequenceGroup], rank_offset: int, is_scheduled: bool) -> List[bool]:
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

    def migrate_out_multistage(self, dst_instance_name: str, rank_offset: int) -> List[str]:
        t0 = time.time()
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
        # logger.info(f"{t0},{self.instance_id} migrate_out:{len(send_blocks)}, seq_id:{seq.seq_id}")
        # check if migrate in instace have enough blocks
        dst_actor_hadel = ray.get_actor(dst_instance_name)
        recv_block_table = ray.get(dst_actor_hadel.migrate_in_stage_0.remote(
            len(send_blocks), -rank_offset))
        recv_blocks = [block.block_number for block in recv_block_table]
        # logger.info(f"recv_blocks:{len(recv_blocks)}")
        if len(recv_blocks) != len(send_blocks):
            return []
        if send_blocks :
            seq_group.add_event(SequenceEvent.MIGRATE_OUT_STAGE_0)
            ref = dst_actor_hadel._run_workers.remote("recv_gpu_cache", get_all_outputs=True, block_tensor=recv_blocks, rank_offset=-rank_offset)
            self._run_workers("send_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=send_blocks, rank_offset=rank_offset)
            # ref = dst_actor_hadel._run_workers.remote("recv_gpu_cache_ray", get_all_outputs=True, block_list=recv_blocks, src_rank=0)
            # self._run_workers("send_gpu_cache_ray", get_async_outputs=True, get_all_outputs=True, block_list=send_blocks, dst_rank=1)
            # logger.info("stage 0 done!!")
        t1 = time.time()
        # logger.info(f"{t1},stage 0 cost:{t1-t0}")

        # stage 1
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
        new_block_table = ray.get(dst_actor_hadel.migrate_in_stage_1.remote(len(new_append_blocks), recv_block_table, migrating_seq_group, -rank_offset))
        recv_block_table.extend(new_block_table)
        recv_blocks = [block.block_number for block in new_block_table]
        migrating_requests = []
        if len(recv_blocks) == 0:
            if migrating_seq_group:
                self.scheduler.running.append(seq_group)
        else:
            self.scheduler.free_seq(seq)
            seq_group.add_event(SequenceEvent.MIGRATE_OUT_STAGE_1)
            ref = dst_actor_hadel._run_workers.remote("recv_gpu_cache", get_all_outputs=True, block_tensor=recv_blocks, rank_offset=-rank_offset)
            self._run_workers("send_gpu_cache", get_async_outputs=True, block_tensor=new_append_blocks, rank_offset=rank_offset)
            # ref = dst_actor_hadel._run_workers.remote("recv_gpu_cache_ray", get_all_outputs=True, block_list=recv_blocks, src_rank=0)
            # self._run_workers("send_gpu_cache_ray", get_async_outputs=True, block_list=new_append_blocks, dst_rank=1)
            ray.get(ref)
            ray.get(dst_actor_hadel.add_request_running.remote(migrating_seq_group=migrating_seq_group, block_table=recv_block_table))
            migrating_requests.append(seq_group.request_id)
        t2 = time.time()
        # logger.info(f"stage 1 cost:{t2-t1}")
        return migrating_requests

    def migrate_in_stage_0(self, block_num:int, rank_offset:int):
        if block_num == 0:
            return []
        block_table = self.scheduler.block_manager.get_free_blocks(block_num)
        return block_table

    def migrate_in_stage_1(self, new_block_num:int, recv_block_table:List[int], migrating_seq_group:SequenceGroup, rank_offset:int) -> bool:
        block_table = self.scheduler.block_manager.get_free_blocks(new_block_num)
        # seq is finished or fail to allocate block, release block_table
        if migrating_seq_group == None or len(block_table)==0:
            self.scheduler.block_manager._free_block_table(recv_block_table)
            return []
        return block_table

    def add_request_running(self, migrating_seq_group: SequenceGroup, block_table: List[int]):
        seq = migrating_seq_group.get_seqs()[0]
        seq.seq_id = next(self.seq_counter)
        self.scheduler.block_manager.add_block_table(block_table, seq.seq_id)
        migrating_seq_group.add_event(SequenceEvent.MIGRATE_IN)
        self.scheduler.running.append(migrating_seq_group)
        logger.info("add_running!!")

    def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        t00 = time.time()
        (seq_group_metadata_list, scheduler_outputs,
         ignored, instance_info) = self._schedule()
        self.is_scheduled = True
        if self.scaling_down:
            instance_info.num_running_request = 1
            instance_info.num_available_gpu_block = -1000
            instance_info.num_available_gpu_block_waiting = -1000
        if scheduler_outputs.is_empty():
            return ignored, instance_info

        # Execute the model.
        t0_inference_begin = time.time()
        output =  self._run_workers(
            "execute_model",
            get_async_outputs=True,
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # Update the scheduler with the model outputs.
        t1_inference_end = time.time()
        # logger.info(f"step time:{(t11-t00)*1000}ms")
        # update instance info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        instance_info.latency = (t1_inference_end - t0_inference_begin)*1000
        self._update_sequences_time(scheduler_outputs.scheduled_seq_groups, t0_inference_begin, t1_inference_end)
        if scheduler_outputs.prompt_run:
            for seq_group in scheduler_outputs.scheduled_seq_groups:
                seq_group.add_event(SequenceEvent.PREFILL)
        seq_groups = self.scheduler.get_migrating_seq_groups(remove_seq=False)
        if len(seq_groups):
            tot_blocks = []
            for seq in seq_groups[-1].get_seqs(SequenceStatus.RUNNING):
                blocks = self.scheduler.block_manager.get_block_table(seq)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_block_last_running_request = len(tot_blocks)
        # for seq_group, samples in zip(scheduler_outputs.scheduled_seq_groups, output):

        #     self._process_sequence_group_samples(seq_group, samples)
        # tot_used_blocks_num = 0
        # for seq_group in self.scheduler.running:
        #     seq = seq_group.get_seqs()[0]
        #     tot_used_blocks_num += len(self.scheduler.block_manager.get_block_table(seq))
        # logger.info(f"instance:{self.instance_id} tot block num:{tot_used_blocks_num+self.scheduler.block_manager.get_num_free_gpu_blocks()}")
        # logger.info(f"instance:{self.instance_id},{len(self.scheduler.waiting)}, {len(self.scheduler.running)}, {len(self.scheduler.swapped)}")
        return self._process_model_outputs(output, scheduler_outputs) + ignored, instance_info

    def is_ready(self):
        return True