Below, you can find an explanation of each argument for Llumnix, and the explanation of arguments for vLLM is shown in the following link: [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.4.2/models/engine_args.html). Please note that Llumnix does not currently support all features of vLLM. The vLLM features that are not supported by Llumnix are listed at the end of this document.

# Llumnix arguments

Note: since Llumnix is still in alpha stage, the interface and arguments are *subject to change*.

```
usage: -m llumnix.entrypoints.vllm.api_server [-h]
            [--fixed-node-init]
            [--initial-instances INITIAL_INSTANCES]
            [--load-metric {consumed_speed,used_ratio}]
            [--dispatch-policy {balanced,load,queue}]
            [--enable-migrate]
            [--check-migrate-frequency CHECK_MIGRATE_FREQUENCY]
            [--check-migrate-policy {balanced,prefill_constrained,prefill_relaxed}]
            [--migrate-out-threshold MIGRATE_OUT_THRESHOLD]
            [--migrate-policy {LCFS,SJF,LJF}]
            [--enable-prefill-migrate ENABLE_PREFILL_MIGRATE]
            [--enable-scaling]
            [--min-instances MIN_INSTANCES]
            [--max-instances MAX_INSTANCES]
            [--scaling-interval SCALING_INTERVAL]
            [--scale-policy {max_load,avg_load}]
            [--scale-up-threshold SCALE_UP_THRESHOLD]
            [--scale-down-threshold SCALE_DOWN_THRESHOLD]
            [--disable-log-requests-manager]
            [--record-instance-info]
            [--results-filename RESULTS_FILENAME]
            [--gpu-type GPU_TYPE]
            [--profiling-result-file-path PROFILING_RESULT_FILE_PATH]
            [--polling-interval POLLING_INTERVAL]
            [--migration-backend {gloo,rpc}]
            [--migration-cache-blocks MIGRATION_CACHE_BLOCKS]
            [--last-stage-max-blocks LAST_STAGE_MAX_BLOCKS]
            [--max-stages MAX_STAGES]
```

`--fixed-node-init`
- Place llumlet and workers on the current node.

`--initial-instances`
- Number of model instances.
- Default: 1

`--load-metric`
- Load metric.
- Possible choices: consumed_speed, used_ratio
- Default: "consumed_speed"

`--dispatch-policy`
- Dispatch policy.
- Possible choices: balanced, load, queue
- Default: "load"

`--enable-migrate`
- Enable migrate request between instances.

`--check-migrate-frequency`
- Check migrate frequency.
- Default: 1

`--check-migrate-policy`
- Check migrate policy.

`--migrate-out-threshold`
- Migrate out load threshold.
- Default: 3.0

`--migrate-policy`
- Migrate policy.
- Possible choices: LCFS, SJF, LJF
- Default: "LCFS"

`--enable-prefill-migrate`
- Enable prefill migrate.
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

`--scale-policy`
- Scale policy.
- Possible choices: max_load, avg_load
- default: "max_load"

`--scale-up-threshold`
- Scaling up threshold.
- Default: 4

`--scale-down-threshold`
- Scaling down threshold.
- Default: 100

`--disable-log-requests-manager`
- Disable logging requests in manager.
- Default: False

`--record-instance-info`
- Enable recording instance-info data to a csv file.
- Default: False

`--results-filename`
- Results filename.

`--gpu-type`
- GPU type specified when using simulator.
- Default: "a10"

`--profiling-result-file-path`
- Profiling result file path.
- Default: ""

`--polling-interval`
- Time interval(s) to update instance info/migration.
- Default: 0.1

`--migration-backend`
- Communication backend during migration.
- Possible choices: gloo, rpc
- Default: "rpc"

`--migration-backend-init-timeout`
- Timeout(s) for initializing migration backend.
- Default: 5.0

`--migration-cache-blocks`
- Cache blocks num during migration.
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
