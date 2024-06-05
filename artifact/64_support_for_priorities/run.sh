# script parameters:
# 1: qps, 2: query_distribution, 3: coefficient_variation
# 4: imean, 5: omean
# 6: dispatch_strategy, 7: enable_migrate, 8: enable_defrag, 9: migrate_threshold
# 10: log_dirname

bash ./llumnix_exp ./config/serving_exp 26 gamma 2 128 128 load 1 1 3 '128-128/Llumnix'
bash ./llumnix_exp ./config/serving_exp 26 gamma 4 128 128 load 1 1 3 '128-128/Llumnix'
bash ./llumnix_exp ./config/serving_exp 26 gamma 6 128 128 load 1 1 3 '128-128/Llumnix'
bash ./llumnix_exp ./config/serving_exp 26 gamma 8 128 128 load 1 1 3 '128-128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 26 gamma 2 128 128 load 0 1 3 '128-128/Llumnix-base'
bash ./llumnix_exp ./config/serving_exp 26 gamma 4 128 128 load 0 1 3 '128-128/Llumnix-base'
bash ./llumnix_exp ./config/serving_exp 26 gamma 6 128 128 load 0 1 3 '128-128/Llumnix-base'
bash ./llumnix_exp ./config/serving_exp 26 gamma 8 128 128 load 0 1 3 '128-128/Llumnix-base'