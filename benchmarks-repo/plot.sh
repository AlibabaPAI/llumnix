log_file1=/mnt/wencong.xwc/sunbiao/develop/vllm-exp/benchmarks-repo/results/qps7.7_prompts10000_poisson_cv1.0_zipf_imean256_zipf_omean256_migrate0_LCFS_local/disable_migrate_2023-12-01_05:34:24.log
log_file2=/mnt/wencong.xwc/sunbiao/develop/vllm-exp/benchmarks-repo/results/qps7.7_prompts10000_poisson_cv1.0_zipf_imean256_zipf_omean256_migrate1_LCFS_local_prefill1/enable_migrate_2023-12-01_06:03:22.log
log_file3=/mnt/wencong.xwc/sunbiao/develop/vllm-exp/benchmarks-repo/results/qps35.0_prompts10000_poisson_cv1.0_zipf_imean128_zipf_omean128_migrate1_LCFS_local_prefill1/latency_info_disable_migrate_2023-12-02_09:46:28.json

# python plot_utils.py --log-file1 $log_file1 \
#                      --log-file2 $log_file2

# python plot_utils.py --log-file1 $log_file2 \
#                      --log-file2 $log_file3

# python plot_utils.py --log-file1 $log_file3 \
#                      --log-file2 $log_file4

# python plot_utils.py --log-file1 $log_file1 \
#                      --log-file2 $log_file2 \
#                      --log-file3 $log_file3 \

python plot_utils.py --test --log-file1 $log_file1 --log-file2 $log_file2
