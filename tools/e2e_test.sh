# #!/bin/bash
set -ex

pgrep -f llumnix.entrypoints.vllm.api_server | { while read pid; do kill -9 "$pid"; done; }

nvidia-docker run --rm -t --net host --ipc host -v ${PWD}:/workspace -v /mnt:/mnt -w /workspace \
  registry.cn-beijing.aliyuncs.com/llumnix/llumnix-dev:20240909_action_678a439 \
  bash -c "pip install -e . > /dev/null && make e2e_test"
