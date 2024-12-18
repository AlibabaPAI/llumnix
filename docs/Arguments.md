Below, you can find an explanation of each argument for Llumnix, and the explanation of arguments for vLLM is shown in the following link: [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.4.2/models/engine_args.html). Please note that Llumnix does not currently support all features of vLLM. The vLLM features that are not supported by Llumnix are listed at the end of this document.

# Llumnix arguments

Note: since Llumnix is still in alpha stage, the interface and arguments are *subject to change*.

```
usage: -m llumnix.entrypoints.vllm.api_server [-h]
            [--config-file CONFIG_FILE]
            [--disable-fixed-node-init-instance]
            [--disable-init-instance-by-manager]
            [--initial-instances INITIAL_INSTANCES]
            [--load-metric {remaining_steps,usage_ratio}]
            [--polling-interval POLLING_INTERVAL]
            [--dispatch-policy {balanced,load,queue,rr}]
            [--enable-migration]
            [--pair-migration-frequency PAIR_MIGRATION_FREQUENCY]
            [--pair-migration-policy {balanced,defrag_constrained,defrag_relaxed}]
            [--migrate-out-threshold MIGRATE_OUT_THRESHOLD]
            [--request-migration-policy {LCR,SR,LR,FCW,FCWSR}]
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
            [--polling-interval POLLING_INTERVAL]
            [--migration-backend {gloo,nccl,rayrpc,grpc,kvtransfer}]
            [--migration-buffer-blocks MIGRATION_BUFFER_BLOCKS]
            [--migration-backend-transfer-type {cuda_ipc,rdma,}]
            [--migration-backend-kvtransfer-naming-url MIGRATION_BACKEND_KVTRANSFER_NAMING_URL]
            [--migration-backend-server-address MIGRATION_BACKEND_SERVER_ADDRESS]
            [--migration-backend-init-timeout MIGRATION_BACKEND_INIT_TIMEOUT]
            [--migration-num-layers MIGRATION_NUM_LAYERS]
            [--last-stage-max-blocks LAST_STAGE_MAX_BLOCKS]
            [--max-stages MAX_STAGES]
            [--enable-pd-disagg]
            [--num-dispatch-instances NUM_DISPATCH_INSTANCES]
            [--log-request-timestamps]

```

`--config-file`
- Path to config file.

`--disable-fixed-node-init-instance`
- Disable fixing the placement of instance to current node.

`--disable-init-instance-by-manager`
- Disable initializing instance by manager.

`--initial-instances`
- Number of instances created at initialization.
- Default: 1

`--load-metric`
- Instance load metric.
- Possible choices: remaining_steps, usage_ratio
- Default: "remaining_steps"

`--polling-interval`
- Time interval(s) to update instance info and pair migration.
- Default: 0.05

`--dispatch-policy`
- Request dispatch policy.
- Possible choices: balanced, load, queue, rr
- Default: "load"

`--enable-migration`
- Enable migrate requests between instances.

`--pair-migration-frequency`
- Pair migration frequency.
- Default: 1

`--pair-migration-policy`
- Pair migration policy.
- Possible choices: balanced, defrag_constrained, defrag_relaxed
- Default: "defrag_constrained"

`--migrate-out-threshold`
- Migrate out instance load threshold.
- Default: 3.0

`--request-migration-policy`
- Request migration policy.
- Possible choices: LCR, SR, LR, FCW, FCWSR
- Default: "SR"

`--enable-defrag`
- Enable defragmentation through migration based on virtual usage.
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
- default: "avg_load"

`--scale-up-threshold`
- Scale up threshold.
- Default: 10

`--scale-down-threshold`
- Scale down threshold.
- Default: 60

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
- Possible choices: gloo, rayrpc, nccl, grpc, kvtransfer. [gloo, rayrpc, nccl] are available for vllm and [grpc, kvtransfer] are available for bladellm.
- Default: "gloo"

`--migration-backend-transfer-type`
- Transfer type for migration backend kvTransfer.
- Possible choices: cuda_ipc, rdma
- Default: "rdma"

`--migration-backend-server-address`
- Address of grpc server for migration backend
- Default: "127.0.0.1:50051"

`--migration-backend-kvtransfer-naming-url`
- URL of naming server for kvtransfer migration backend
- Default: "file:/tmp/llumnix/naming/"

`--migration-buffer-blocks`
- Number of buffer blocks in migration.
- Default: 512

`--migration-backend-init-timeout`
- Timeout(s) for initializing migration backend.
- Default: 10.0

`--migration-num-layers`
- number of kv-cache layers to transfer in each round during migration
- Default: 1

`--last-stage-max-blocks`
- If the number of remaining blocks < last_stage_max_blocks, do last stage migration.
- Default: 4

`--max-stages`
- Drop migration if the number of stages > max_stages.
- Default: 3

`--log-request-timestamps`
- Enable logging request timestamps.

`--enable-pd-disagg`
- Enable prefill decoding disaggregation.

`--num-dispatch-instances`
- Number of available instances for dispatch.

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
