#!/bin/bash
test_mode=$1

set -ex

pkill -f llumnix.entrypoints.vllm.api_server
pkill -f benchmark_serving.py

nvidia-docker run --rm -t --net host --ipc host -v ${PWD}:/workspace -v /mnt:/mnt -w /workspace \
  registry.cn-beijing.aliyuncs.com/llumnix/llumnix-dev:20240909_action_678a439 \
  bash -c "pip install -e . > /dev/null && make $test_mode"
