#!/bin/bash
test_mode=$1

set -ex

if [[ "$test_mode" == *"vllm"* ]]; then
    image="registry.cn-beijing.aliyuncs.com/llumnix/llumnix-vllm-dev:202504241348_image_54729fe"
    install_command="make vllm_install"
    docker_options=""
elif [[ "$test_mode" == *"bladellm"* ]]; then
    image="registry.cn-beijing.aliyuncs.com/llumnix/llumnix-bladellm-dev:202504251128_image_5dc1604"
    install_command="make bladellm_install"
    docker_options="--device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/rdma_cm --ulimit memlock=-1"
else
    echo "Error: Invalid test_mode '$test_mode'."
    exit 1
fi

nvidia-docker run --rm -t \
  -v ${PWD}:/test_workspace \
  -v /mnt:/mnt \
  -w /test_workspace \
  $docker_options \
  --shm-size=512mb \
  "$image" sh -c "date && $install_command > /dev/null && date && make $test_mode"
