# 256 256
# 7.5 7.6 7.7
# 0.5h
# mo: 10>6>3
# bash vllm_variable_size_exp ./config/motivation2_exp 256 256 0 local FCFS 1.7 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/motivation2_exp 256 256 0 local FCFS 1.8 poisson 1.0 0 LCFS
bash vllm_variable_size_exp ./config/motivation2_exp 256 256 0 local FCFS 1.9 poisson 1.0 0 LCFS
# bash vllm_variable_size_exp ./config/motivation2_exp 256 256 0 local FCFS 2.0 poisson 1.0 0 LCFS
