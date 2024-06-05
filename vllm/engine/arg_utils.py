import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, RequestSchedulerConfig)


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    use_np_weights: bool = False
    use_dummy_weights: bool = False
    dtype: str = 'auto'
    seed: int = 0
    worker_use_ray: bool = True
    instance_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    migrate_backend: str = "gloo"
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 2560
    max_num_seqs: int = 256
    disable_log_stats: bool = False
    enable_migrate: bool = False
    async_migrate: bool = False
    enable_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 1
    scaling_interval: int = 10
    migrate_strategy: str = 'LCFS'

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model
        self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            default='facebook/opt-125m',
            help='name or path of the huggingface model to use')
        parser.add_argument(
            '--tokenizer',
            type=str,
            default=EngineArgs.tokenizer,
            help='name or path of the huggingface tokenizer to use')
        parser.add_argument('--tokenizer-mode',
                            type=str,
                            default=EngineArgs.tokenizer_mode,
                            choices=['auto', 'slow'],
                            help='tokenizer mode. "auto" will use the fast '
                            'tokenizer if available, and "slow" will '
                            'always use the slow tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='trust remote code from huggingface')
        parser.add_argument('--download-dir',
                            type=str,
                            default=EngineArgs.download_dir,
                            help='directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface')
        parser.add_argument('--use-np-weights',
                            action='store_true',
                            help='save a numpy copy of model weights for '
                            'faster loading. This can increase the disk '
                            'usage by up to 2x.')
        parser.add_argument('--use-dummy-weights',
                            action='store_true',
                            help='use dummy values for model weights')
        # TODO(woosuk): Support FP32.
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=['auto', 'half', 'bfloat16', 'float'],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models.')
        # Parallel arguments
        parser.add_argument('--worker-use-ray',
                            action='store_true',
                            help='use Ray for distributed serving, will be '
                            'automatically set when using more than 1 GPU')
        parser.add_argument('--instance-parallel-size',
                            '-ip',
                            type=int,
                            default=EngineArgs.instance_parallel_size,
                            help='number of model instance')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        parser.add_argument('--migrate-backend',
                            type=str,
                            default=EngineArgs.migrate_backend,
                            help='torch distributed backend')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='token block size')
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='random seed')
        parser.add_argument('--swap-space',
                            type=int,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=EngineArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                            'the model executor')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                            'iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='disable logging statistics')
        parser.add_argument('--enable-migrate',
                            action='store_true',
                            help='enable request migrating between instance')
        parser.add_argument('--async-migrate',
                            action='store_true',
                            help='use async migrating api')
        parser.add_argument('--enable-scaling',
                            action='store_true',
                            help='enable auto scaline')
        parser.add_argument('--min-replicas',
                            type=int,
                            default=EngineArgs.min_replicas,
                            help='min replicas num')
        parser.add_argument('--max-replicas',
                            type=int,
                            default=EngineArgs.max_replicas,
                            help='max replicas num')
        parser.add_argument('--scaling-interval',
                            type=int,
                            default=EngineArgs.scaling_interval,
                            help='interval time to check scaling')
        parser.add_argument('--migrate-strategy',
                            type=str,
                            default=EngineArgs.migrate_strategy,
                            choices=['LCFS', 'SJF', 'LJF'],
                            help='migrate strategy')
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
        # Initialize the configs.
        model_config = ModelConfig(self.model, self.tokenizer,
                                   self.tokenizer_mode, self.trust_remote_code,
                                   self.download_dir, self.use_np_weights,
                                   self.use_dummy_weights, self.dtype,
                                   self.seed)
        cache_config = CacheConfig(self.block_size,
                                   self.gpu_memory_utilization,
                                   self.swap_space)
        parallel_config = ParallelConfig(self.pipeline_parallel_size,
                                         self.tensor_parallel_size,
                                         self.worker_use_ray,
                                         self.instance_parallel_size,
                                         self.max_replicas if self.enable_scaling else self.instance_parallel_size,
                                         self.migrate_backend)
        scheduler_config = SchedulerConfig(self.max_num_batched_tokens,
                                           self.max_num_seqs,
                                           model_config.get_max_model_len(),
                                           self.migrate_strategy)
        return model_config, cache_config, parallel_config, scheduler_config


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    engine_use_ray: bool = True
    disable_log_requests_engine: bool = False

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--engine-use-ray',
                            action='store_true',
                            help='use Ray to start the LLM engine in a '
                            'separate process as the server process.')
        parser.add_argument('--disable-log-requests-engine',
                            action='store_true',
                            help='disable logging requests')
        return parser

@dataclass
class EngineManagerArgs(AsyncEngineArgs):
    dispatch_strategy: str = 'naive'
    load_metric: str = 'consumed_speed'
    enable_load_control_prefill: bool = False,
    prefill_SLO: float = 5
    need_migrate_frequency: int = 4
    migrate_out_threshold: float = 1.5
    migrate_in_threshold: float = 3.0
    scale_strategy: str = 'max_load'
    scale_up_threshold: float = 4
    scale_down_threshold: float = 100
    disable_log_requests_manager: bool = False
    results_filename: str = "result"
    profiling_result_file_path: str = "profiling_result.pkl"
    dispatch_mode: str = 'local'
    need_dispatch_frequency: int = 4
    global_dispatch_strategy: str = 'FCFS'

    def create_engine_manager_configs(
        self,
    ) -> Tuple[RequestSchedulerConfig]:
        request_scheduler_config = RequestSchedulerConfig(self.instance_parallel_size,
                                                          self.load_metric,
                                                          bool(self.enable_load_control_prefill),
                                                          self.prefill_SLO,
                                                          self.dispatch_strategy,
                                                          self.migrate_out_threshold, 
                                                          self.migrate_in_threshold,
                                                          self.scale_strategy,
                                                          self.scale_up_threshold,
                                                          self.scale_down_threshold,
                                                          self.global_dispatch_strategy)
        return request_scheduler_config

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = AsyncEngineArgs.add_cli_args(parser)
        parser.add_argument('--load-metric',
                            type=str,
                            default=EngineManagerArgs.load_metric,
                            choices=['consumed_speed', 'used_ratio'],
                            help='load metric')
        parser.add_argument('--enable-load-control-prefill',
                            type=int,
                            default=EngineManagerArgs.enable_load_control_prefill,
                            help='enable prefill control')
        parser.add_argument('--prefill-SLO',
                            type=float,
                            default=EngineManagerArgs.prefill_SLO,
                            help='load metric')
        parser.add_argument('--dispatch-strategy',
                            type=str,
                            default=EngineManagerArgs.dispatch_strategy,
                            choices=['naive', 'balanced', 'load', 'unbalanced', 'block'],
                            help='dispatch strategy')
        parser.add_argument('--need-migrate-frequency',
                            type=int,
                            default=EngineManagerArgs.need_migrate_frequency,
                            help='migrate frequency')
        parser.add_argument('--migrate-out-threshold',
                            type=float,
                            default=EngineManagerArgs.migrate_out_threshold,
                            help='migrate out load threshold')
        parser.add_argument('--migrate-in-threshold',
                            type=float,
                            default=EngineManagerArgs.migrate_in_threshold,
                            help='migrate in load threshold')
        parser.add_argument('--disable-log-requests-manager',
                            action='store_true',
                            help='disable logging requests')
        parser.add_argument('--results-filename',
                            type=str,
                            help='record instance and request info')
        parser.add_argument('--scale-strategy',
                            type=str,
                            default=EngineManagerArgs.scale_strategy,
                            choices=['max_load', 'avg_load'],
                            help='scale strategy')
        parser.add_argument('--scale-up-threshold',
                            type=float,
                            default=EngineManagerArgs.scale_up_threshold,
                            help='scaling up threshold')
        parser.add_argument('--scale-down-threshold',
                            type=float,
                            default=EngineManagerArgs.scale_down_threshold,
                            help='scaling down threshold')
        parser.add_argument('--profiling-result-file-path',
                            type=str,
                            default=EngineManagerArgs.profiling_result_file_path,
                            help='profiling result file path')
        parser.add_argument("--dispatch-mode", 
                            type=str,
                            default=EngineManagerArgs.dispatch_mode,
                            choices=['local', 'global'],
                            help='generate method')
        parser.add_argument('--need-dispatch-frequency',
                            type=int,
                            default=EngineManagerArgs.need_dispatch_frequency,
                            help='dispatch frequency')
        parser.add_argument('--global-dispatch-strategy',
                            type=str,
                            default=EngineManagerArgs.global_dispatch_strategy,
                            choices=['FFIT', 'FCFS', 'BE', 'SJF', 'LJF'],
                            help='global dispatch strategy')
        return parser
