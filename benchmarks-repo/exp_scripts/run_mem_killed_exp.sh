# 1: config, 2: imean, 3: omean, 4: enable_load_control_prefill, 5: dispatch_mode
# 6: global_dispatch_strategy, 7: qps, 8: query_distribution, 9: coefficient_variation, 10: enable_migrate, 11: migrate_strategy

bash vllm_variable_size_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.43 poisson 1.0 0 LCFS
bash vllm_variable_size_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.44 poisson 1.0 0 LCFS
bash vllm_variable_size_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.45 poisson 1.0 0 LCFS
bash vllm_variable_size_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.46 poisson 1.0 0 LCFS
bash vllm_variable_size_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.47 poisson 1.0 0 LCFS