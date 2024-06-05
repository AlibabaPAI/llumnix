# script parameters:
# 1: qps, 2: query_distribution, 3: coefficient_variation
# 4: imean, 5: omean
# 6: dispatch_strategy, 7: enable_migrate, 8: enable_defrag, 9: migrate_threshold
# 10: log_dirname


# We test 7 qps values rather than 5 in paper for each trace to show more stable results in once run.

# Estimated Time of this script: 47h
# sharegpt: 4h, burstgpt: 4h, 128-128: 2.5h, 256-256: 7h, 512-512: 14h, 128-512: 11.5h, 512-128: 3.5h


# sharegpt dataset
# 7.0 7.25 7.5 7.75 8.0
# 5x3=15 commands, 15min per command, 4h in total

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.0 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' balanced 0 0 3 'sharegpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.0 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 0 0 3 'sharegpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.0 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 1 1 3 'sharegpt/Llumnix'

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.25 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' balanced 0 0 3 'sharegpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.25 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 0 0 3 'sharegpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.25 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 1 1 3 'sharegpt/Llumnix'

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.50 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' balanced 0 0 3 'sharegpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.50 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 0 0 3 'sharegpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.50 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 1 1 3 'sharegpt/Llumnix'

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.75 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' balanced 0 0 3 'sharegpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.75 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 0 0 3 'sharegpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.75 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 1 1 3 'sharegpt/Llumnix'

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.0 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' balanced 0 0 3 'sharegpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.0 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 0 0 3 'sharegpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.0 poisson 1.0 sharegpt './sharegpt_gpt4.jsonl' load 1 1 3 'sharegpt/Llumnix'


# burstgpt dataset
# 7.5 7.75 8.0 8.25 8.5
# 5x3=15 commands, 15min per command, 4h in total

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.50 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' balanced 0 0 3 'burstgpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.50 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 0 0 3 'burstgpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.50 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 1 1 3 'burstgpt/Llumnix'

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.75 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' balanced 0 0 3 'burstgpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.75 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 0 0 3 'burstgpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 7.75 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 1 1 3 'burstgpt/Llumnix'

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.00 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' balanced 0 0 3 'burstgpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.00 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 0 0 3 'burstgpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.00 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 1 1 3 'burstgpt/Llumnix'

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.25 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' balanced 0 0 3 'burstgpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.25 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 0 0 3 'burstgpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.25 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 1 1 3 'burstgpt/Llumnix'

bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.50 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' balanced 0 0 3 'burstgpt/Round_Robin'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.50 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 0 0 3 'burstgpt/INFaas++'
bash ./llumnix_exp_dataset ./config/serving_exp_dataset 8.50 poisson 1.0 burstgpt './BurstGPT_GPT4-Conversation.csv' load 1 1 3 'burstgpt/Llumnix'


# 128-128
# 32.0 33.0 34.0 35.0 36.0 37.0 38.0
# 7x2=14 commands, 10min per command, 2.5h in total

bash ./llumnix_exp ./config/serving_exp 32.0 poisson 1.0 128 128 load 0 0 3 '128_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 32.0 poisson 1.0 128 128 load 1 1 3 '128_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 33.0 poisson 1.0 128 128 load 0 0 3 '128_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 33.0 poisson 1.0 128 128 load 1 1 3 '128_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 34.0 poisson 1.0 128 128 load 0 0 3 '128_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 34.0 poisson 1.0 128 128 load 1 1 3 '128_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 35.0 poisson 1.0 128 128 load 0 0 3 '128_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 35.0 poisson 1.0 128 128 load 1 1 3 '128_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 36.0 poisson 1.0 128 128 load 0 0 3 '128_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 36.0 poisson 1.0 128 128 load 1 1 3 '128_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 37.0 poisson 1.0 128 128 load 0 0 3 '128_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 37.0 poisson 1.0 128 128 load 1 1 3 '128_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 38.0 poisson 1.0 128 128 load 0 0 3 '128_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 38.0 poisson 1.0 128 128 load 1 1 3 '128_128/Llumnix'


# 256 256
# 7.4 7.5 7.6 7.7 7.8 7.9 8.0
# 7x2=14 commands, 0.5h per command, 7h in total

bash ./llumnix_exp ./config/serving_exp 7.4 poisson 1.0 256 256 load 0 0 10 '256_256/INFaas++'
bash ./llumnix_exp ./config/serving_exp 7.4 poisson 1.0 256 256 load 1 1 10 '256_256/Llumnix'

bash ./llumnix_exp ./config/serving_exp 7.5 poisson 1.0 256 256 load 0 0 10 '256_256/INFaas++'
bash ./llumnix_exp ./config/serving_exp 7.5 poisson 1.0 256 256 load 1 1 10 '256_256/Llumnix'

bash ./llumnix_exp ./config/serving_exp 7.6 poisson 1.0 256 256 load 0 0 10 '256_256/INFaas++'
bash ./llumnix_exp ./config/serving_exp 7.6 poisson 1.0 256 256 load 1 1 10 '256_256/Llumnix'

bash ./llumnix_exp ./config/serving_exp 7.7 poisson 1.0 256 256 load 0 0 10 '256_256/INFaas++'
bash ./llumnix_exp ./config/serving_exp 7.7 poisson 1.0 256 256 load 1 1 10 '256_256/Llumnix'

bash ./llumnix_exp ./config/serving_exp 7.8 poisson 1.0 256 256 load 0 0 10 '256_256/INFaas++'
bash ./llumnix_exp ./config/serving_exp 7.8 poisson 1.0 256 256 load 1 1 10 '256_256/Llumnix'

bash ./llumnix_exp ./config/serving_exp 7.9 poisson 1.0 256 256 load 0 0 10 '256_256/INFaas++'
bash ./llumnix_exp ./config/serving_exp 7.9 poisson 1.0 256 256 load 1 1 10 '256_256/Llumnix'

bash ./llumnix_exp ./config/serving_exp 8.0 poisson 1.0 256 256 load 0 0 10 '256_256/INFaas++'
bash ./llumnix_exp ./config/serving_exp 8.0 poisson 1.0 256 256 load 1 1 10 '256_256/Llumnix'


# 512 512
# 2.6 2.7 2.8 2.9 3.0 3.1 3.2
# 7x2=14 commands, 1h per command, 14h in total

bash ./llumnix_exp ./config/serving_exp 2.6 poisson 1.0 512 512 load 0 0 3 '512_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.6 poisson 1.0 512 512 load 1 1 3 '512_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.7 poisson 1.0 512 512 load 0 0 3 '512_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.7 poisson 1.0 512 512 load 1 1 3 '512_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.8 poisson 1.0 512 512 load 0 0 3 '512_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.8 poisson 1.0 512 512 load 1 1 3 '512_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 2.9 poisson 1.0 512 512 load 0 0 3 '512_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 2.9 poisson 1.0 512 512 load 1 1 3 '512_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 3.0 poisson 1.0 512 512 load 0 0 3 '512_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 3.0 poisson 1.0 512 512 load 1 1 3 '512_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 3.1 poisson 1.0 512 512 load 0 0 3 '512_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 3.1 poisson 1.0 512 512 load 1 1 3 '512_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 3.2 poisson 1.0 512 512 load 0 0 3 '512_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 3.2 poisson 1.0 512 512 load 1 1 3 '512_512/Llumnix'


# 128-512
# 4.10 4.15 4.2 4.25 4.30 4.35 4.40
# 14 commands, 50min per command, 11.5h in total

bash ./llumnix_exp ./config/serving_exp 4.10 poisson 1.0 128 512 load 0 0 10 '128_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 4.10 poisson 1.0 128 512 load 1 1 10 '128_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 4.15 poisson 1.0 128 512 load 0 0 10 '128_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 4.15 poisson 1.0 128 512 load 1 1 10 '128_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 4.20 poisson 1.0 128 512 load 0 0 10 '128_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 4.20 poisson 1.0 128 512 load 1 1 10 '128_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 4.25 poisson 1.0 128 512 load 0 0 10 '128_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 4.25 poisson 1.0 128 512 load 1 1 10 '128_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 4.30 poisson 1.0 128 512 load 0 0 10 '128_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 4.30 poisson 1.0 128 512 load 1 1 10 '128_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 4.35 poisson 1.0 128 512 load 0 0 10 '128_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 4.35 poisson 1.0 128 512 load 1 1 10 '128_512/Llumnix'

bash ./llumnix_exp ./config/serving_exp 4.40 poisson 1.0 128 512 load 0 0 10 '128_512/INFaas++'
bash ./llumnix_exp ./config/serving_exp 4.40 poisson 1.0 128 512 load 1 1 10 '128_512/Llumnix'


# 512-128
# 10.0 12.0 14.0 16.0 18.0 20.0 22.0
# 7x2=14 commands, 15min per command, 3.5h in total

bash ./llumnix_exp ./config/serving_exp 10.0 poisson 1.0 512 128 load 0 0 3 '512_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 10.0 poisson 1.0 512 128 load 1 1 3 '512_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 12.0 poisson 1.0 512 128 load 0 0 3 '512_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 12.0 poisson 1.0 512 128 load 1 1 3 '512_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 14.0 poisson 1.0 512 128 load 0 0 3 '512_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 14.0 poisson 1.0 512 128 load 1 1 3 '512_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 16.0 poisson 1.0 512 128 load 0 0 3 '512_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 16.0 poisson 1.0 512 128 load 1 1 3 '512_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 18.0 poisson 1.0 512 128 load 0 0 3 '512_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 18.0 poisson 1.0 512 128 load 1 1 3 '512_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 20.0 poisson 1.0 512 128 load 0 0 3 '512_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 20.0 poisson 1.0 512 128 load 1 1 3 '512_128/Llumnix'

bash ./llumnix_exp ./config/serving_exp 22.0 poisson 1.0 512 128 load 0 0 3 '512_128/INFaas++'
bash ./llumnix_exp ./config/serving_exp 22.0 poisson 1.0 512 128 load 1 1 3 '512_128/Llumnix'
