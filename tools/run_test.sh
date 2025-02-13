#!/bin/bash
test_mode=$1

set -ex

pgrep -f llumnix.entrypoints.vllm.api_server | { while read pid; do kill -9 "$pid"; done; }
pgrep -f benchmark_serving.py | { while read pid; do kill -9 "$pid"; done; }

nvidia-docker run --rm -t --net host --ipc host -v ${PWD}:/workspace -v /mnt:/mnt -w /workspace \
  registry.cn-beijing.aliyuncs.com/llumnix/llumnix-dev:20250213_image_65d0368 \
  sh -c "make vllm_install > /dev/null && make $test_mode"
