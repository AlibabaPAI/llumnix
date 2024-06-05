# 512 512
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.0 poisson 1.0 1 SJF &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.0 poisson 1.0 1 LJF &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.1 poisson 1.0 1 LCFS &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.1 poisson 1.0 1 SJF &
nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 512 512 1 local FCFS 3.1 poisson 1.0 1 LJF &

# 256 256
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 8.0 poisson 1.0 1 LCFS &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 8.0 poisson 1.0 1 SJF &
# nohup bash vllm_variable_size_sim_exp ./sim_config/prefill_load_control_exp 256 256 1 local FCFS 8.0 poisson 1.0 1 LJF &