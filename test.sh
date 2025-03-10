export HEAD_NODE_IP='127.0.0.1'

HEAD_NODE=1 python -m llumnix.entrypoints.vllm.api_server \
                --config-file /mnt/sunbiao.sun/develop/llumnix-github/configs/vllm.yml \
                --model facebook/opt-125m \
                --worker-use-ray \
