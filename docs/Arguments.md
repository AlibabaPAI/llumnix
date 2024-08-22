Below, you can find an explanation of each argument for Llumnix, and the explanation of arguments for vLLM is shown in the following link: [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.4.2/models/engine_args.html). Please note that Llumnix does not currently support all features of vLLM. The vLLM features that are not supported by Llumnix are listed at the end of this document.

# Llumnix arguments

Note: since Llumnix is still in alpha stage, the interface and arguments are *subject to change*.

```
usage: -m llumnix.entrypoints.vllm.api_server [-h]
            [--fixed-node-init-instance]
            [--init-instance-by-manager]
            [--initial-instances INITIAL_INSTANCES]
            [--load-metric {remaining_step,usage_ratio}]
            [--polling-interval POLLING_INTERVAL]
            [--dispatch-policy {balanced,load,queue}]
            [--enable-migration]
            [--pair-migration-frequency PAIR_MIGRATION_FREQUENCY]
            [--pair-migration-policy {balanced,prefill_constrained,prefill_relaxed}]
            [--migrate-out-threshold MIGRATE_OUT_THRESHOLD]
            [--request-migration-policy {LCFS,SJF,LJF}]
            [--enable-defrag ENABLE_DEFRAG]
            [--enable-scaling]
            [--min-instances MIN_INSTANCES]
            [--max-instances MAX_INSTANCES]
            [--scaling-interval SCALING_INTERVAL]
            [--scaling-policy {max_load,avg_load}]
            [--scale-up-threshold SCALE_UP_THRESHOLD]
            [--scale-down-threshold SCALE_DOWN_THRESHOLD]
            [--disable-log-requests-manager]
            [--log-instance-info]
            [--log-filename LOG_FILENAME]
            [--profiling-result-file-path PROFILING_RESULT_FILE_PATH]
            [--gpu-type GPU_TYPE]
            [--migration-backend {gloo,rpc}]
            [--migration-cache_blocks MIGRATION_CACHE_BLOCKS]
            [--last-stage-max-blocks LAST_STAGE_MAX_BLOCKS]
            [--max-stages MAX_STAGES]
```

`--fixed-node-init-instance`
- Fix the placement of instance to current node.

`--init-instance-by-manager`
- initialize instance by manager.

`--initial-instances`
- Number of model instances created at initialization.
- Default: 1

`--load-metric`
- Instance load metric.
- Possible choices: remaining_step, usage_ratio
- Default: "remaining_step"

`--polling-interval`
- Time interval(s) to update instance info and pair migration.
- Default: 0.1

`--dispatch-policy`
- Request dispatch policy.
- Possible choices: balanced, load, queue
- Default: "load"

`--enable-migration`
- Enable migrate requests between instances.

`--pair-migration-frequency`
- Pair migration frequency.
- Default: 1

`--pair-migration-policy`
- Pair migration policy.

`--migrate-out-threshold`
- Migrate out instance load threshold.
- Default: 3.0

`--request-migration-policy`
- Request migration policy.
- Possible choices: LCFS, SJF, LJF
- Default: "SJF"

`--enable-defrag`
- Enable defragmentation.
- Default: False

`--enable-scaling`
- Enable auto scaling.

`--min-instances`
- Minimum number of instances.
- Default: 1

`--max-instances`
- Maximum number of instances.
- Default: 1

`--scaling-interval`
- Interval time of check scaling.
- Default: 10

`--scaling-policy`
- Scaling policy.
- Possible choices: max_load, avg_load
- default: "max_load"

`--scale-up-threshold`
- Scale up threshold.
- Default: 4

`--scale-down-threshold`
- Scale down threshold.
- Default: 100

`--disable-log-requests-manager`
- Disable logging requests in manager.

`--log-instance-info`
- Enable logging instance info.

`--log-filename`
- Log filename.
- Default: "server.log"

`--profiling-result-file-path`
- Profiling result file path.
- Default: ""

`--gpu-type`
- GPU type specified when using simulator.
- Default: "a10"

`--migration-backend`
- Communication backend of migration.
- Possible choices: gloo, rpc
- Default: "rpc"

`--migration-cache-blocks`
- Number of cache blocks in migration.
- Default: 512

`--last-stage-max-blocks`
- If the remaining blocks num < last_stage_max_blocks, do last stage migration.
- Default: 4

`--max-stages`
- Drop migration if stage num > max_stages.
- Default: 3

# Unsupported vLLM feature options

`--device`
- Llumnix supports cuda device, other device type is unsupported currently.

`--enable-lora`
- Llumnix does not support multi-lara serving currently.

`--enable-prefix-caching`
- Llumnix does not support automatic prefix caching currently.

`--enable-chunked-prefill`
- Llumnix does not support chunked prefill currently.

`--use-v2-block-manager`
- Llumnix does not support speculative decoding currently.

`--speculative-model`
- Llumnix does not support speculative decoding currently.

Besides, Llumnix does not support sampling algorithms whose number of ouput sequences is greater than one (vllm.SamplingParams.n > 1), such as beam search.
