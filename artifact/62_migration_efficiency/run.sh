# script parameters

# 1: config
# 2: seq_len, 3: enable migration

# run 7b normal trace(without migration)
bash ./llumnix_exp ./config/overhead_7b 256 0 "7b/Normal/256"
bash ./llumnix_exp ./config/overhead_7b 512 0 "7b/Normal/512"
bash ./llumnix_exp ./config/overhead_7b 1024 0 "7b/Normal/1024"
bash ./llumnix_exp ./config/overhead_7b 2048 0 "7b/Normal/2048"
bash ./llumnix_exp ./config/overhead_7b 4096 0 "7b/Normal/4096"
bash ./llumnix_exp ./config/overhead_7b 8192 0 "7b/Normal/8192"

# run 7b migration trace
bash ./llumnix_exp ./config/overhead_7b 256 1 "7b/Migration/256"
bash ./llumnix_exp ./config/overhead_7b 512 1 "7b/Migration/512"
bash ./llumnix_exp ./config/overhead_7b 1024 1 "7b/Migration/1024"
bash ./llumnix_exp ./config/overhead_7b 2048 1 "7b/Migration/2048"
bash ./llumnix_exp ./config/overhead_7b 4096 1 "7b/Migration/4096"
bash ./llumnix_exp ./config/overhead_7b 8192 1 "7b/Migration/8192"

# run 30b normal trace(without migration)
bash ./llumnix_exp ./config/overhead_30b 256 0 "30b/Normal/256"
bash ./llumnix_exp ./config/overhead_30b 512 0 "30b/Normal/512"
bash ./llumnix_exp ./config/overhead_30b 1024 0 "30b/Normal/1024"
bash ./llumnix_exp ./config/overhead_30b 2048 0 "30b/Normal/2048"
bash ./llumnix_exp ./config/overhead_30b 4096 0 "30b/Normal/4096"
bash ./llumnix_exp ./config/overhead_30b 8192 0 "30b/Normal/8192"

# run 30b migration trace
bash ./llumnix_exp ./config/overhead_30b 256 1 "30b/Migration/256"
bash ./llumnix_exp ./config/overhead_30b 512 1 "30b/Migration/512"
bash ./llumnix_exp ./config/overhead_30b 1024 1 "30b/Migration/1024"
bash ./llumnix_exp ./config/overhead_30b 2048 1 "30b/Migration/2048"
bash ./llumnix_exp ./config/overhead_30b 4096 1 "30b/Migration/4096"
bash ./llumnix_exp ./config/overhead_30b 8192 1 "30b/Migration/8192"


# run recompute
bash ./llumnix_exp ./config/recompute_7b 256 0 "7b/Recompute/256"
bash ./llumnix_exp ./config/recompute_7b 512 0 "7b/Recompute/512"
bash ./llumnix_exp ./config/recompute_7b 1024 0 "7b/Recompute/1024"
bash ./llumnix_exp ./config/recompute_7b 2048 0 "7b/Recompute/2048"
bash ./llumnix_exp ./config/recompute_7b 4096 0 "7b/Recompute/4096"
bash ./llumnix_exp ./config/recompute_7b 8192 0 "7b/Recompute/8192"

bash ./llumnix_exp ./config/recompute_30b 256 0 "30b/Recompute/256"
bash ./llumnix_exp ./config/recompute_30b 512 0 "30b/Recompute/512"
bash ./llumnix_exp ./config/recompute_30b 1024 0 "30b/Recompute/1024"
bash ./llumnix_exp ./config/recompute_30b 2048 0 "30b/Recompute/2048"
bash ./llumnix_exp ./config/recompute_30b 4096 0 "30b/Recompute/4096"
bash ./llumnix_exp ./config/recompute_30b 8192 0 "30b/Recompute/8192"