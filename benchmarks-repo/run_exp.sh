# 1: config, 2: imean, 3: omean, 4: enable_load_control_prefill, 5: dispatch_mode
# 6: global_dispatch_strategy, 7: qps, 8: query_distribution, 9: coefficient_variation, 10: enable_migrate, 11: migrate_strategy

# 512 512 3.0
# 2.8 2.9 3.0
# bash vllm_variable_size_exp ./config/prefill_load_control_exp 512 512 0 local FCFS 2.6 poisson 1.0 0 LCFS