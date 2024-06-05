# 1: config, 2: imean, 3: omean, 4: enable_load_control_prefill, 5: dispatch_mode
# 6: global_dispatch_strategy, 7: qps, 8: query_distribution, 9: coefficient_variation, 10: enable_migrate, 11: migrate_strategy

nohup bash vllm_variable_size_sim_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.4 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.45 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.5 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./config/mem_killed_exp 256 256 0 local FCFS 0.55 poisson 1.0 0 LCFS &

# nohup bash vllm_variable_size_sim_exp ./config/mem_killed_exp 128 128 0 local FCFS 1.8 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./config/mem_killed_exp 128 128 0 local FCFS 1.9 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./config/mem_killed_exp 128 128 0 local FCFS 2.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./config/mem_killed_exp 128 128 0 local FCFS 2.1 poisson 1.0 0 LCFS &
