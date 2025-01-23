# Getting Started
Llumnix can generate latency data from logs. After run a real benchmark with `--log-instance-info`, you can find a `$LOG_FILENAME.csv` file.

After running profiling with `python llumnix.backends.profiling.py`. You can get a `$PROFILING_RESULT_FILE_PATH.pkl`

Then, you can run simulator with `--simulator-mode` and `--profiling-result-file-path PROFILING_RESULT_FILE_PATH`.


```
usage: -m llumnix.backends.profiling [-h]
            [--database PROFILING_RESULT_FILE_PATH]
            [--log-csv-path CSV_FILE_PATH]
            [--model MODEL_NAME]
            [--tp TENSOR_PARALLEL_SIZE]
            [--pp PIPELINE_PARALLEL_SIZE]
            [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
            [--block-size BLOCK_SIZE]
            [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
            [--num-gpu-blocks NUM_GPU_BLOCKS]
            [--new-data]
```

`--database`
- Path to profiling result file.

`--log-csv-path`
- Path to real llumnix benchmark csv file.

`--model`
- Name of model (same as huggingface model name when use vllm).

`--tp`
- Number of tensor parallel replicas.
- Default: 1

`--pp`
- Number of pipeline parallel replicas.
- Default: 1

`--gpu-memory-utilization`
- The fraction of GPU memory to be used for the model executor.
- Default: 0.9

`--block-size`
- Token block size for contiguous chunks of tokens.
- Default: 16

`--max-num-batched-tokens`
- Maximum number of batched tokens per iteration.
- Default: 16

`--num-gpu-blocks`
- Number of gpu blocks profiled by inference engine.

`--new-data`
- create a new profiling result file, otherwise write to a exist file.

