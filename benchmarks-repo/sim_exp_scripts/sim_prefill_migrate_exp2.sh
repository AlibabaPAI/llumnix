# 1: config, 2: imean, 3: omean, 4: enable_load_control_prefill, 5: dispatch_mode
# 6: global_dispatch_strategy, 7: qps, 8: query_distribution, 9: coefficient_variation, 10: enable_migrate, 11: migrate_strategy


# 1024 1024
# 0.6 0.7 0.8 0.9 1.0
# 3.5h
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 0.6 poisson 1.0 0 LCFS & 
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 1 local FCFS 0.6 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 0.7 poisson 1.0 0 LCFS & 
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 1 local FCFS 0.7 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 0.8 poisson 1.0 0 LCFS & 
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 1 local FCFS 0.8 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 0.9 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 1 local FCFS 0.9 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 0 local FCFS 1.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 1024 1024 1 local FCFS 1.0 poisson 1.0 1 LCFS &


# 512 512
# 2.7 2.8 2.9 3.0 3.1
# 1h
# mo: 3>6>10
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 2.7 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 2.7 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 2.8 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 2.8 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 2.9 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 2.9 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 3.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 0 local FCFS 3.1 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.1 poisson 1.0 1 LCFS &


# 256 256
# 7.6 7.7 7.8 7.9 8.0
# 0.5h
# mo: 10>6>3
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 7.6 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 7.6 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 7.7 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 7.7 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 7.8 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 7.8 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 7.9 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 7.9 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 0 local FCFS 8.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 8.0 poisson 1.0 1 LCFS &


# 128 128
# 31.0 33.0 35.0
# 32.0 33.0 34.0
# 31.0 32.0 33.0 34.0 35.0
# 10min
# mo: 3>6>10
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 31.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 1 local FCFS 31.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 32.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 1 local FCFS 32.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 33.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 1 local FCFS 33.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 34.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 1 local FCFS 34.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 0 local FCFS 35.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 128 1 local FCFS 35.0 poisson 1.0 1 LCFS &


# 512 128
# 10.0 12.0 14.0 16.0 18.0
# 15min
# mo: 3>10>6
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 10.0 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 1 local FCFS 10.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 12.0 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 12.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 1 local FCFS 12.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 14.0 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 14.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 1 local FCFS 14.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 16.0 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 16.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 1 local FCFS 16.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 16.5 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 16.5 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 1 local FCFS 16.5 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 17.0 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 17.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 1 local FCFS 17.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 17.5 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 17.5 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 1 local FCFS 17.5 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 18.0 poisson 1.0 0 LCFS &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 0 local FCFS 18.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 128 1 local FCFS 18.0 poisson 1.0 1 LCFS &


# 128 512
# 4.2 4.25 4.3 4.35 4.4
# 50min
# mo: 10>3>6
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 0 local FCFS 4.2 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 1 local FCFS 4.2 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 0 local FCFS 4.25 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 1 local FCFS 4.25 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 0 local FCFS 4.3 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 1 local FCFS 4.3 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 0 local FCFS 4.35 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 1 local FCFS 4.35 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 0 local FCFS 4.4 poisson 1.0 0 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 128 512 1 local FCFS 4.4 poisson 1.0 1 LCFS &
