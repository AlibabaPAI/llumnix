import copy
import time
from functools import partial
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

from vllm.config import (_GB, CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.worker.cache_engine import CacheEngine
from vllm.engine.ray_utils import initialize_cluster, ray, RayWorker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Sequence, SequenceEvent, SequenceGroup, SequenceStatus,
                             SequenceGroupMetadata, SequenceOutputs)
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.utils import Counter
from vllm.instance_info import InstanceInfo
from vllm.simulator.profiling import (ProfilingResult, SimParallelConfig, SimCacheConfig)

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
        profiling_result: ProfilingResult,
        gpu: str,
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
        # self.workers = workers
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

        # self._init_model()
        # Profile the memory usage and initialize the cache.
        self.migrate_cost = 0
        self.profile = profiling_result
        sim_parallel_config = SimParallelConfig(self.parallel_config.tensor_parallel_size,
                                                self.parallel_config.pipeline_parallel_size)
        self.latency_mem = profiling_result.para_dict[sim_parallel_config]
        sim_cache_config = SimCacheConfig(gpu, self.cache_config.gpu_memory_utilization, 
                                          self.cache_config.block_size, self.scheduler_config.max_num_batched_tokens)
        self.migrate_bw = self.latency_mem.migrate_bw[gpu]
        # self.swap_bw = self.latency_mem.swap_bw[gpu]
        self.num_gpu_blocks = self.latency_mem.cache_dict[sim_cache_config]
        self._init_cache()
        self.cache_block_size = CacheEngine.get_cache_block_size(
            self.cache_config.block_size, self.model_config, self.parallel_config)
        self.cache_block_size /= _GB
        logger.info(f"block cache size: {self.cache_block_size}")
        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)

        self.step_counter = Counter()

        self.num_migrated_out_request_killed = 0
        self.num_migrated_out_request_running = 0

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
        self.cache_config.num_gpu_blocks = self.num_gpu_blocks
        self.cache_config.num_cpu_blocks = 2048

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

    def migrate_out_waiting(self, dst_actor_handle):
        migrating_seq_groups = self.scheduler.get_migrating_seq_groups()
        # if migrating_seq_groups:
        #     logger.info(f"migrate out seq_groups: {migrating_seq_groups[0].get_seqs()[0].prompt}")
        # else:
        #     logger.info(f"migrate out seq_groups: []")
        if not migrating_seq_groups:
            return []
        migrating_requests = []
        migrated_mask = dst_actor_handle.migrate_in_waiting(
            copy.deepcopy(migrating_seq_groups))
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
            migrating_requests.append(migrating_seq_groups[i].request_id)
            seq = migrating_seq_groups[i].get_seqs()[0]
            # logger.info(f"seqlen: {seq.get_len()}")
            self.scheduler.free_seq(seq, SequenceStatus.FINISHED_MIGRATED)
        return migrating_requests 

    def migrate_in_waiting(self, migrating_seq_groups: List[SequenceGroup]):
        for seq_group in migrating_seq_groups:
            for seq in seq_group.seqs:
                seq.seq_id = next(self.seq_counter)
            self.scheduler.waiting.append(seq_group)
        return [True]*len(migrating_seq_groups)

    def migrate_out(self, dst_actor_handle):
        migrating_seq_groups = self.scheduler.get_migrating_seq_groups()
        # if migrating_seq_groups:
        #     logger.info(f"migrate out seq_groups: {migrating_seq_groups[0].get_seqs()[0].prompt}")
        # else:
        #     logger.info(f"migrate out seq_groups: []")
        if not migrating_seq_groups:
            return []
        migrated_mask = dst_actor_handle.migrate_in(
            copy.deepcopy(migrating_seq_groups))
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
            self.migrate_cost += (self.cache_block_size * len(send_blocks)) / self.migrate_bw
            # self._run_workers("send_gpu_cache", rank_offset, send_blocks)
            # pass
            # t1 = time.time()
            # logger.info(f"send_cost:{(t1-t0)*1000}ms")
        return migrating_requests

    def migrate_in(self, migrating_seq_groups: List[SequenceGroup]) -> List[bool]:
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
                logger.info(f"{self.instance_id} migrate in seq_groups:{migrating_seq_groups[0].request_id}")
        # logger.info(f"recv_blocks: {recv_blocks}, len: {len(recv_blocks)}")
        if recv_blocks:
            # self.recv_rayobj = self._run_workers(
            #     "recv_gpu_cache", get_async_outputs=True, get_all_outputs=True, block_tensor=recv_blocks, rank_offset=rank_offset)
            self.migrate_cost += (self.cache_block_size * len(recv_blocks)) / self.migrate_bw
        return migrated_mask

    def scale_down_migrate(self, all_instances) -> List[Tuple[int, int]]:
        migrating_list = []
        migrating_request_list = []
        while self.scheduler.running or self.scheduler.waiting:
            seq_group = self.scheduler.running.pop() if self.scheduler.running else self.scheduler.waiting.pop()
            migrating_list.append(seq_group)
            seq = seq_group.get_seqs()[0]
            if seq.status == SequenceStatus.RUNNING:
                self.scheduler.free_seq(seq, SequenceStatus.FINISHED_MIGRATED)
        for instance_idx in range(self.instance_id):
            migrating_seq_groups = []
            for idx in range(instance_idx,len(migrating_list), self.instance_id):
                migrating_seq_groups.append(migrating_list[idx])
            all_instances[instance_idx].migrate_in_waiting(copy.deepcopy(migrating_seq_groups))
            for seq_group in migrating_seq_groups:
                migrating_request_list.append((seq_group.request_id, instance_idx))

        return migrating_request_list


    def _get_dummy_output(self, scheduler_outputs: SchedulerOutputs) -> Dict[int, SequenceOutputs]:
        output_dict = {}
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                dummy_seq_output = SequenceOutputs(seq_id, seq_id, 20, {20:1.0})
                output_dict[seq_id] =  dummy_seq_output
        return output_dict

    def step(self, timestamp: float = 0) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        (seq_group_metadata_list, scheduler_outputs,
         early_return, instance_info) = self._schedule()
        # logger.info(early_return)
        if early_return is not None:
            instance_info.instance_id = self.instance_id
            instance_info.step_id = next(self.step_counter)
            instance_info.timestamp = timestamp

            return early_return, instance_info, scheduler_outputs, timestamp, timestamp

        # Execute the model.
        t0_inference_begin = timestamp
        output = self._get_dummy_output(scheduler_outputs)
        # output = self._run_workers(
        #     "execute_model",
        #     seq_group_metadata_list=seq_group_metadata_list,
        #     blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
        #     blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
        #     blocks_to_copy=scheduler_outputs.blocks_to_copy,
        # )
        t1_inference_end = timestamp + self.migrate_cost
        self.migrate_cost = 0
        if scheduler_outputs.prompt_run:
            bs = 1
            sum_seq_len = instance_info.num_batched_tokens
            sum_seq_len = _pad_to_alignment(sum_seq_len, 8)
            t1_inference_end += self.latency_mem.prefill_latency[(1,sum_seq_len)][0] / 1000
        else:
            sum_seq_len = sum(instance_info.running_seq_lens)
            bs = instance_info.num_seq
            bs = _pad_to_alignment(bs, 8)
            bs = min(bs, 56)
            t1_inference_end += self.latency_mem.decode_latency[(bs,sum_seq_len)][0] / 1000
            # print(f"instance_{self.instance_id}, timestamp:{timestamp}, sum_seq_len{sum_seq_len},{self.latency_mem.decode_latency[sum_seq_len][0]}")
        # # swapping cost
        # t1_inference_end += len(scheduler_outputs.blocks_to_swap_in) * self.cache_block_size / self.swap_bw
        # t1_inference_end += len(scheduler_outputs.blocks_to_swap_out) * self.cache_block_size / self.swap_bw
        return output, instance_info, scheduler_outputs, t0_inference_begin, t1_inference_end

    def post_process(self, output, instance_info, scheduler_outputs, t0_inference_begin, t1_inference_end):
        # Update the scheduler with the model outputs.
        seq_groups = self.scheduler.update(output)

        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
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
        instance_info.timestamp = t1_inference_end
        instance_info.latency = (t1_inference_end - t0_inference_begin) * 1000
        if self.scheduler.waiting:
            first_waiting_seq_group = self.scheduler.waiting[0]
            instance_info.waiting_time_first_waiting_request = instance_info.timestamp - first_waiting_seq_group.arrival_time
            # logger.info(f"instance_info.timestamp: {instance_info.timestamp}")
            # logger.info(f"first_waiting_seq_group.arrival_time: {first_waiting_seq_group.arrival_time}")
        else:
            instance_info.waiting_time_first_waiting_request = 0
        # logger.info(f"waiting_time_first_waiting_request: {instance_info.waiting_time_first_waiting_request}")
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
                new_token, new_output_text = '#' ,seq.output_text + '#'
                # new_token, new_output_text = detokenize_incrementally(
                #     self.tokenizer,
                #     seq.output_tokens,
                #     seq.get_last_token_id(),
                #     skip_special_tokens=True,
                # )
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
    
    def is_empty(self):
        return self.scheduler.get_num_unfinished_seq_groups() == 0


def _pad_to_alignment(x, multiple_of):
    return x + ((-1*x) % multiple_of)
