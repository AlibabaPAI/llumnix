# 1: config, 2: imean, 3: omean, 4: enable_load_control_prefill, 5: dispatch_mode
# 6: global_dispatch_strategy, 7: qps, 8: query_distribution, 9: coefficient_variation, 10: enable_migrate, 11: migrate_strategy

# 1024 1024 0.9
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 1024 1024 0 local FCFS 0.9 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 1024 1024 1 local FCFS 0.9 poisson 1.0 1 LCFS


# 512 512 3.0
# 2.8 2.9 3.0
# 1h
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 2.6 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 1 local FCFS 2.6 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 2.7 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 1 local FCFS 2.7 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 2.8 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 1 local FCFS 2.8 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 2.9 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 1 local FCFS 2.9 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 3.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 3.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 1 local FCFS 3.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 3.1 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 1 local FCFS 3.1 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 3.2 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 1 local FCFS 3.2 poisson 1.0 1 LCFS


# 256 256 8.0
# 7.6 7.7 7.8 7.9 8.0
# 0.5h
# mo: 10>6>3
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 0 local FCFS 7.4 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 1 local FCFS 7.4 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 0 local FCFS 7.5 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 1 local FCFS 7.5 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 0 local FCFS 7.6 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 1 local FCFS 7.6 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 0 local FCFS 7.7 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 1 local FCFS 7.7 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 0 local FCFS 7.8 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 1 local FCFS 7.8 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 0 local FCFS 7.9 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 1 local FCFS 7.9 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 0 local FCFS 8.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 256 256 1 local FCFS 8.0 poisson 1.0 1 LCFS


# 128 128
# 31.0 33.0 35.0
# 32.0 33.0 34.0
# 31.0 32.0 33.0 34.0 35.0
# 10min
# mo: 3>6>10
bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 0 local FCFS 28.0 poisson 1.0 0 LCFS
bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 1 local FCFS 28.0 poisson 1.0 1 LCFS
bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 0 local FCFS 29.0 poisson 1.0 0 LCFS
bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 1 local FCFS 29.0 poisson 1.0 1 LCFS
bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 0 local FCFS 30.0 poisson 1.0 0 LCFS
bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 1 local FCFS 30.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 0 local FCFS 31.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 1 local FCFS 31.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 0 local FCFS 32.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 1 local FCFS 32.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 0 local FCFS 33.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 1 local FCFS 33.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 0 local FCFS 34.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 1 local FCFS 34.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 0 local FCFS 35.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 128 1 local FCFS 35.0 poisson 1.0 1 LCFS

# 512 128
# 10.0 12.0 14.0 16.0 18.0
# 15min
# mo: 3>10>6
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 0 local FCFS 10.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 1 local FCFS 10.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 0 local FCFS 12.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 1 local FCFS 12.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 0 local FCFS 14.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 1 local FCFS 14.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 0 local FCFS 16.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 1 local FCFS 16.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 0 local FCFS 18.0 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 128 1 local FCFS 18.0 poisson 1.0 1 LCFS

# 128 512
# 4.2 4.25 4.3 4.35 4.4
# 50min
# mo: 10>3>6
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 0 local FCFS 4.15 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 1 local FCFS 4.15 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 0 local FCFS 4.2 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 1 local FCFS 4.2 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 0 local FCFS 4.25 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 1 local FCFS 4.25 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 0 local FCFS 4.3 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 1 local FCFS 4.3 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 0 local FCFS 4.35 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 1 local FCFS 4.35 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 0 local FCFS 4.4 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 1 local FCFS 4.4 poisson 1.0 1 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 0 local FCFS 4.45 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 128 512 1 local FCFS 4.45 poisson 1.0 1 LCFS
