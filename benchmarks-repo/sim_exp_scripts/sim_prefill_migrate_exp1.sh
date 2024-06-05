# 1: config, 2: imean, 3: omean, 4: enable_load_control_prefill, 5: dispatch_mode
# 6: global_dispatch_strategy, 7: qps, 8: query_distribution, 9: coefficient_variation, 10: enable_migrate, 11: migrate_strategy

# 1024 1024 0.9
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 0.9 poisson 1.0 0 LCFS

# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 0.9 poisson 1.0 1 LCFS
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 0.9 poisson 1.0 1 SJF
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 0.9 poisson 1.0 1 LJF

# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 1 local FCFS 0.9 poisson 1.0 1 LCFS
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 1 local FCFS 0.9 poisson 1.0 1 SJF
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 1 local FCFS 0.9 poisson 1.0 1 LJF


# 512 512 3.0
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 3.0 poisson 1.0 0 LCFS

bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 3.0 poisson 1.0 1 LCFS
bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 3.0 poisson 1.0 1 SJF
bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 3.0 poisson 1.0 1 LJF

bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.0 poisson 1.0 1 SJF
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.0 poisson 1.0 1 LJF


# 256 256 8.0
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 8.0 poisson 1.0 0 LCFS

# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 8.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 8.0 poisson 1.0 1 SJF
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 8.0 poisson 1.0 1 LJF

# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 8.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 8.0 poisson 1.0 1 SJF
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 8.0 poisson 1.0 1 LJF


# 128 128 35.0
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 35.0 poisson 1.0 0 LCFS

# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 35.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 35.0 poisson 1.0 1 SJF
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 35.0 poisson 1.0 1 LJF

# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 1 local FCFS 35.0 poisson 1.0 1 LCFS
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 1 local FCFS 35.0 poisson 1.0 1 SJF
# bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 1 local FCFS 35.0 poisson 1.0 1 LJF