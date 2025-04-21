#!/bin/bash
test_mode=$1

set -ex

pgrep -f llumnix.entrypoints.vllm.api_server | { while read pid; do kill -9 "$pid"; done; }
pgrep -f benchmark_serving.py | { while read pid; do kill -9 "$pid"; done; }

if [[ "$test_mode" == *"vllm"* ]]; then
    image="registry.cn-beijing.aliyuncs.com/llumnix/llumnix-dev:20250213_image_65d0368"
    install_command="make vllm_install"
    docker_options="--net host --ipc host"
elif [[ "$test_mode" == *"bladellm"* ]]; then
    image="registry.cn-beijing.aliyuncs.com/llumnix/llumnix-bladellm-base-dev:202504181632_386530_2b436d6"
    install_command="make bladellm_install"
    docker_options="--net host --ipc host --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm"
else
    echo "Error: Invalid test_mode '$test_mode'."
    exit 1
fi

nvidia-docker run --rm -t \
  -v ${PWD}:/test_workspace \
  -v /mnt:/mnt \
  -w /test_workspace \
  $docker_options \
  "$image" sh -c "$install_command > /dev/null && make $test_mode"
