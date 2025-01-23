Below, you can find an explanation of each argument for Llumnix, and the explanation of arguments for vLLM is shown in the following link: [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.6.3.post1/models/engine_args.html). Please note that Llumnix does not currently support all features of vLLM. The vLLM features that are not supported by Llumnix are listed at the end of this document.

# Llumnix arguments

Note: since Llumnix is still in alpha stage, the interface and arguments are *subject to change*.

```
usage: -m llumnix.entrypoints.vllm.api_server [-h]
            [--host HOST]
            [--port PORT]
            [--ssl-keyfile SSL_KEYFILE]
            [--ssl-certfile SSL_CERTFILE]
            [--log-level {debug,info,warning,error}]
            [--launch-ray-cluster]
            [--ray-cluster-port RAY_CLUSTER_PORT]
            [--disable-log-to-driver]
            [--request-output-queue-type {rayqueue,zmq}]
            [--request-output-queue-port REQUEST_OUTPUT_QUEUE_PORT]
            [--disable-log-requests-server]
            [--log-request-timestamps]
            [--config-file CONFIG_FILE]
            [--initial-instances INITIAL_INSTANCES]
            [--dispatch-load-metric {remaining_steps,usage_ratio}]
            [--migration-load-metric {remaining_steps,usage_ratio}]
            [--scaling-load-metric {remaining_steps,usage_ratio}]
            [--polling-interval POLLING_INTERVAL]
            [--dispatch-policy {balanced,load,queue,rr}]
            [--power-of-k-choice POWER_OF_K_CHOICE]
            [--enable-migration]
            [--enable-defrag]
            [--pair-migration-frequency PAIR_MIGRATION_FREQUENCY]
            [--pair-migration-policy {balanced,defrag_constrained,defrag_relaxed}]
            [--migrate-out-threshold MIGRATE_OUT_THRESHOLD]
            [--request-migration-policy {LCR,SR,LR,FCW,FCWSR}]
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
            [--simulator-mode]
            [--profiling-result-file-path PROFILING_RESULT_FILE_PATH]
            [--gpu-type GPU_TYPE]
            [--migration-backend {gloo,nccl,rayrpc,grpc,kvtransfer}]
            [--migration-buffer-blocks MIGRATION_BUFFER_BLOCKS]
            [--migration-num-layers MIGRATION_NUM_LAYERS]
            [--migration-backend-init-timeout MIGRATION_BACKEND_INIT_TIMEOUT]
            [--migration-backend-transfer-type {cuda_ipc,rdma,}]
            [--grpc-migration-backend-server-address GRPC_MIGRATION_BACKEND_SERVER_ADDRESS]
            [--kvtransfer-migration-backend-naming-url KVTRANSFER_MIGRATION_BACKEND_NAMING_URL]
            [--migration-max-stages MIGRATION_MAX_STAGES]
            [--migration-last-stage-max-blocks MIGRATION_LAST_STAGE_MAX_BLOCKS]
            [--enable-pd-disagg]
            [--pd-ratio PD_RATIO]
            [--enable-port-increment]
            [--enable-port-offset-store]
            [--instance-type INSTANCE_TYPE]

```

`--host`
- Hostname of the server.
- Default: "localhost"

`--port`
- Port number of the server.
- Default: 8000

`--ssl-keyfile`
- Path to SSL key file.
- Default: None

`--ssl-certfile`
- Path to SSL certificate file.
- Default: None

`--log-level`
- Log level for the server.
- Possible choices: debug, info, warning, error
- Default: "info"

`--launch-ray-cluster`
- If launch ray cluster.

`--ray-cluster-port`
- Ray cluster port.
- Default: 6379

`--disable-log-to-driver`
- Disable redirecting all worker logs to driver.

`--request-output-queue-type`
- Queue type for request output queue.
- Possible choices: rayqueue, zmq
- Default: "rayqueue"

`--request-output-queue-port`
- Port number for the zmq request output queue.
- Default: 1234

`--disable-log-requests-server`
- Disable logging requests in server.

`--log-request-timestamps`
- If log request timestamps.

`--config-file`
- Path to config file of arguments.
- Default: None

`--initial-instances`
- Number of instances created at initialization.
- Default: 1

`--dispatch-load-metric`
- Instance dispatch load metric.
- Possible choices: remaining_steps, usage_ratio
- Default: "remaining_steps"

`--migration-load-metric`
- Instance migration load metric.
- Possible choices: remaining_steps, usage_ratio
- Default: "remaining_steps"

`--scaling-load-metric`
- Instance scaling load metric.
- Possible choices: remaining_steps, usage_ratio
- Default: "remaining_steps"

`--polling-interval`
- Time interval(s) to update instance info and pair migration.
- Default: 0.05

`--dispatch-policy`
- Request dispatch policy.
- Possible choices: balanced, load, queue, rr
- Default: "load"

`--power-of-k-choice`
- Number of candidate instances for dispatch policy
- Default: 1

`--enable-migration`
- Enable migrate requests between instances.

`--enable-defrag`
- Enable defragmentation through migration based on virtual usage.

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

`--simulator-mode`
- Enable simulator mode.

`--profiling-result-file-path`
- Profiling result file path when using simulator.
- Default: None

`--migration-backend`
- Communication backend of migration.
- Possible choices: gloo, rayrpc, nccl, grpc, kvtransfer. [gloo, rayrpc, nccl] are available for vllm and [grpc, kvtransfer] are available for bladellm.
- Default: "gloo"

`--migration-buffer-blocks`
- Number of buffer blocks in migration.
- Default: 512

`--migration-num-layers`
- number of kv-cache layers to transfer in each round during migration
- Default: 1

`--migration-backend-init-timeout`
- Timeout(s) for initializing migration backend.
- Default: 10.0

`--migration-backend-transfer-type`
- Transfer type for migration backend grpc and kvTransfer.
- Possible choices: cuda_ipc, rdma
- Default: "rdma"

`--grpc-migration-backend-server-address`
- Address of grpc server for migration backend
- Default: "127.0.0.1:50051"

`--kvtransfer-migration-backend-naming-url`
- URL of naming server for kvtransfer migration backend
- Default: "file:/tmp/llumnix/naming/"

`--migration-max-stages`
- Drop migration if the number of stages > migration_max_stages.
- Default: 3

`--migration-last-stage-max-blocks`
- If the number of remaining blocks < migration_last_stage_max_blocks, do last stage migration.
- Default: 16

`--enable-pd-disagg`
- Enable prefill decoding disaggregation.

`--pd-ratio`
- The p:d ratio used in gloabl launch model.
- Default: "1:1"

`--enable-port-increment`
- Enable port increment when desploying multiple servers.

`--enable-port-offset-store`
- Enable store port offset when desploying multiple servers.

`--instance-type`
- Instance types for the engine.
- Possible choices: prefill, decode, no_constraints

# Unsupported vLLM feature options

`--device`
- Llumnix supports cuda device, other device type is unsupported currently.

`--enable-lora`
- Llumnix does not support multi-lara serving currently.

`--enable-prefix-caching`
- Llumnix does not support automatic prefix caching currently.

`--enable-chunked-prefill`
- Llumnix does not support chunked prefill currently.

`--speculative-model`
- Llumnix does not support speculative decoding currently.

`--pipeline-parallel-size`
- Llumnix does not support pipeline parallel currently.

`--num-schedule-steps`
- Llumnix does not support multi-step scheduling currently.

Besides, Llumnix does not support sampling algorithms whose number of ouput sequences is greater than one (vllm.SamplingParams.n > 1), such as beam search.
