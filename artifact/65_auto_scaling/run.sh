# script parameters:
# 1: qps, 2: query_distribution, 3: coefficient_variation
# 4: imean, 5: omean
# 6: dispatch_strategy, 7: enable_migrate, 8: enable_defrag, 9: scale_threshold
# 10: log_dirname

# poission trace
bash ./llumnix_exp ./config/serving_exp 2.0 poisson 1 512 512 load 0 1 10 'Poisson/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 poisson 1 512 512 load 1 1 10 'Poisson/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.2 poisson 1 512 512 load 0 1 10 'Poisson/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.2 poisson 1 512 512 load 1 1 10 'Poisson/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.4 poisson 1 512 512 load 0 1 10 'Poisson/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.4 poisson 1 512 512 load 1 1 10 'Poisson/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.6 poisson 1 512 512 load 0 1 10 'Poisson/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.6 poisson 1 512 512 load 1 1 10 'Poisson/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.8 poisson 1 512 512 load 0 1 10 'Poisson/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.8 poisson 1 512 512 load 1 1 10 'Poisson/Llumnix'

bash ./llumnix_exp ./config/serving_exp 3.0 poisson 1 512 512 load 0 1 10 'Poisson/INFaas++'
bash ./llumnix_exp ./config/serving_exp 3.0 poisson 1 512 512 load 1 1 10 'Poisson/Llumnix'

# gamma trace
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 0 1 10 'Gamma/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 1 1 10 'Gamma/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.0 gamma 3 512 512 load 0 1 10 'Gamma/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 3 512 512 load 1 1 10 'Gamma/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.0 gamma 4 512 512 load 0 1 10 'Gamma/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 4 512 512 load 1 1 10 'Gamma/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.0 gamma 5 512 512 load 0 1 10 'Gamma/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 5 512 512 load 1 1 10 'Gamma/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.0 gamma 6 512 512 load 0 1 10 'Gamma/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 6 512 512 load 1 1 10 'Gamma/Llumnix'

# run with different scaling thresholds
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 0 0 5 'Threshold/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 1 1 5 'Threshold/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 0 0 25 'Threshold/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 1 1 25 'Threshold/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 0 0 45 'Threshold/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 1 1 45 'Threshold/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 0 0 65 'Threshold/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 1 1 65 'Threshold/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 0 0 85 'Threshold/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.0 gamma 2 512 512 load 1 1 85 'Threshold/Llumnix'
