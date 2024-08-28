# Installation

## Requirements

Llumnix requires python `3.8.1~3.10.0` and is currently built on top of vLLM (version 0.4.2). Therefore, the installation requirements are almost identical to those of vLLM. You can view the specific installation requirements for vLLM at the following link:

[vLLM Installation](https://docs.vllm.ai/en/v0.4.2/getting_started/installation.html)

### Install from Pypi

You can install Llumnix from pypi:
```
pip install llumnix
```

### Build from Source

You can build and install Llumnix from source:
```
git clone https://github.com/AlibabaPAI/llumnix.git
cd llumnix
make install
```

The default migration backend is RPC. If you want to use NCCL as the migration backend, run `make cupy-cuda` to install [cupy-cuda](https://pypi.org/search/?q=cupy-cuda) manually, as it is related to the CUDA version.

If you want to use Gloo as migration backend, **in addition to installing cupy-cuda**, please refer to [this link](https://github.com/ZeldaHuang/pygloo/blob/main/.github/workflows/ubuntu_basic.yml#L24C1-L26C1) to install [Bazel](https://github.com/bazelbuild/bazel) >= 5.1.0. Then, run `make pygloo` to install [pygloo](https://github.com/ZeldaHuang/pygloo).

Note: Using conda is not recommended, as it cannot properly handle pygloo's dependency on gcc libstdc++.so.6: version GLIBCXX_3.4.30.

After installation, you can follow this guide to use Llumnix for multi-instance LLM serving quickly.

# Deployment

## Migrating from Existing Deployments

Inference engines like vLLM provide an API server user interface, e.g., `python -m vllm.entrypoints.api_server`. To deploy multiple instances, people start multiple such API servers, each corresponding to one instance, on multiple nodes / containers / k8s pods.

Llumnix provides a similar user interface to enable seamless integration with such existing multi-instance deployments.
You only need two simple steps to migrate from a deployed vLLM service to Llumnix:

1. Replace the original vLLM server entrypoint with the Llumnix one.
```
python -m llumnix.entrypoints.vllm.api_server \
    --host $HOST \
    --port $PORT \
    # Llumnix arguments ...
    # vLLM arguments ...
    ...
```

Upon starting the server, Llumnix's components are automatically configured.
In addition to the server arguments provided above, it's necessary to specify both the Llumnix arguments and the vLLM arguments. For detailed configuration options, please consult the documentation for [Llumnix arguments](./Arguments.md) and [vLLM arguments](https://docs.vllm.ai/en/v0.4.2/models/engine_args.html).

2. Launch multiple servers and connect to the Llumnix cluster. Llumnix uses Ray to manage multiple vLLM servers and instances. You need to configure the following environment variables for Llumnix to correctly set up the cluster.
```
# Configure on all nodes.
export HEAD_NODE_IP=$HEAD_NODE_IP_ADDRESS

# Configure on head node.
export HEAD_NODE=1
```

During the execution of serving deployment, Llumnix will:
- Initiate the Ray cluster for distributed execution.
- Start Llumnix actor components, including LLMEngineManager, Llumlet, among others.
- Launch the vLLM engine instances.

Following these steps, Llumnix acts as the request scheduling layer situated behind the multiple frontend API servers and above the multiple backend vLLM engine instances. This positioning allows Llumnix to significantly enhance serving performance through its dynamic, fine-grained, and KV-cache-aware request scheduling and rescheduling across instances.

## Ray Cluster Notice
When you include the --launch-ray-cluster option in Llumnix's serving deployment command, Llumnix automatically builds a Ray cluster during the execution of serving deployment. This action will overwrite any existing Ray cluster. If this behavior is not desired, simply omit the --launch-ray-cluster option, and Llumnix will initiate its actor components within the current Ray cluster.


# Benchmarking
We provide a benchmarking example to help you get through the usage of Llumnix.
First, you should start the server to launch Llumnix and backend LLM engine instances:
```
HEAD_NODE=1 python -m llumnix.entrypoints.vllm.api_server \
                --host $HOST \
                --port $PORT \
                --initial-instances $INITIAL_INSTANCES \
                --launch-ray-cluster \
                --model $MODEL_PATH \
                --engine-use-ray \
                --worker-use-ray \
                --max-model-len 4096
```
`INITIAL_INSTANCES` determines the number of instances to be launched on the current node, while `MODEL_PATH` defines the location of your model.

Second, you can run the benchmark to evaluate the serving performance:

```
cd benchmark

python benchmark_serving.py \
    --ip_ports ${IP_PORTS[@]} \
    --tokenizer $MODEL_PATH \
    --random_prompt_count $NUM_PROMPTS \
    --dataset_type "sharegpt" \
    --dataset_path $DATASET_PATH \
    --qps $QPS \
    --distribution "poisson" \
    --log_latencies \
    --fail_on_response_failure \
```

`IP_PORTS` should follow the format of `IP_ADDRESS:PORT`, which is the IP address of the server started in the first step. `DATASET_PATH` indicates the location of the ShareGPT dataset you've downloaded, available at [ShareGPT](https://huggingface.co/datasets/shibing624/sharegpt_gpt4).