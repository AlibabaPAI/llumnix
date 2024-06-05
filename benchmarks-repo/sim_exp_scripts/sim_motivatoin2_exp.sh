# 256 256
# 7.5 7.6 7.7
# 0.5h
# mo: 10>6>3
nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 0 local FCFS 1.7 poisson 1.0 0 LCFS load &
nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 0 local FCFS 1.7 poisson 1.0 1 LCFS load &
nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 1 local FCFS 1.7 poisson 1.0 1 LCFS load &

nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 0 local FCFS 1.8 poisson 1.0 0 LCFS load &
# nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 0 local FCFS 1.8 poisson 1.0 1 LCFS load &
# nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 1 local FCFS 1.8 poisson 1.0 1 LCFS load &

nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 0 local FCFS 1.9 poisson 1.0 0 LCFS load &
nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 0 local FCFS 1.9 poisson 1.0 1 LCFS load &
nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 1 local FCFS 1.9 poisson 1.0 1 LCFS load &

# nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 0 local FCFS 2.0 poisson 1.0 1 LCFS load &
# nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 1 local FCFS 2.0 poisson 1.0 1 LCFS load &
# nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 0 local FCFS 2.2 poisson 1.0 1 LCFS load &
# nohup bash vllm_variable_size_sim_exp_motivation2 ./sim_config/motivation2_exp 256 256 1 local FCFS 2.2 poisson 1.0 1 LCFS load &

