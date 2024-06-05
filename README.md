# Artifact for Llumnix (OSDI'24)

## Overview
This is the artifact for the OSDI'24 paper "Llumnix: Dynamic Scheduling for Large Language Model Serving".

You will be able to reproduce the main results in the paper following the instructions provided below in order.

Besides, we provide the code organization of Llumnix project at the end of this document. You can check the functionality of Llumnix by reviewing the codes.

## Environment Setup
Our experiments use a cluster with 4 nodes, each equipped with 4 A10 GPUs, as decribed in section 6.1 of the paper.

### For AE Reviewers (Using Pre-Launched Containers)

We have already setup the Docker containers and the Ray cluster on the GPU machines we provided (see HotCRP for how to access them). The reviewers just need to login to the head node and attach to our container:

```
docker exec -it llumnix-artifact bash
```

That's it! You can then jump to the instructions for running the experiments.

<!-- **We have already prepared the environment in the dockers we provided on the 4 nodes. Therefore, the artifact evaluators can skip the environment preparation steps below.**  -->

### Prepare Environment By Yourself

You can also setup the containers by yourself by using Dockerfile:

```
# on each node
docker build -t llumnix:osdi24ae .
nvidia-docker run -tid --name llumnix-artifact --net host --ipc host -v /mnt:/mnt llumnix:osdi24ae
```

Launch a Ray cluster:
```
# On the head node
ray start --head

# On the other three nodes
ray start --address="$head_address"
```

$head_address is shown in CLI after running 'ray start --head' on the head node.

## Getting Started Instructions (a hello world example)

To verify that the environment has been successfully built and to check the basic functionality of the artifact, we provided a hello word example that can be finished in less than 5 miniutes.

```
cd artifact/0_getting_started

# Serving 2000 requests within 5 miniutes
./llumnix_exp ./config/serving_exp_test 36.0 poisson 1.0 128 128 load 1 1 3 '128_128/Llumnix'
```

During the execution of the script, Llumnix continuously prints the number of finished requests and will print request statistics after all requests have been finished. In the end, you should be able to see logs like:

```
...
INFO xxx llm_engine_manager.py:314] self.num_finished_request 2000.
INFO:     ::1:36714 - "POST /generate_v2 HTTP/1.1" 200 OK
...
Latency:
bin_edges=array([...])
hist=array([...])
cumsum=array([...])
16.0 gpu/s
```

If the above command is executed successfully without getting stucked for a long time, it means that we have built the environment correctly and can go to the next detailed instructions section.

## Detailed Instuctions (to reproduce the main evaluation results)

**The estimated time of running all experiments below for once is about 4 days.**

**If running all the experiments is too lengthy for you, you can also use the history logs of all experiments we provided. The history logs are generated through our experiments during the preparation of AE (thus only run once). You can quickly verify the results through plotting the history logs. See the 'Plot the history logs' part in each experiment section.**

The procedure of the following experiments is very similar, and therefore we will only explain the procedure of 6.3 in detail, other experiments can reference 6.3.


### 6.3 Serving Performance (Figure 11) (Estimated Running Time: 47h)

* Download dataset
```
cd ./artifact/63_serving_performance
wget https://llumnix.oss-cn-beijing.aliyuncs.com/dataset/BurstGPT_GPT4-Conversation.csv
wget https://huggingface.co/datasets/shibing624/sharegpt_gpt4/blob/main/sharegpt_gpt4.jsonl
```
* Run script:
```
cd ./artifact/63_serving_performance
nohup bash run.sh &
```
We can get all the results in Figure 11 of the paper through running the run.sh.

The run.sh executes the llumnix_exp script for multiple times to collect all the data points in Figure 11. During each execution, it generates some log files in the corresponding sub-directory of ./log directory.

In the end of the llumnix_exp script, we run './plot/process_log.py' to obtain key metrics data (e.g: Requet P99/Mean, Prefill P99/Mean etc.). The metrics data is saved to the file that ends with .data in the corresponding log directory.

* After the run.sh finishes (47h), plot figure:
```
# In the 63_serving_performance directory
cd ./plot
python plot.py
```

The plot.py reads all the .data files generated in previous step to collect all evaluation data. Then, the plot.py plots figure11.png and generates figure11.claim in the plot directory. The .claim file shows the key improvements summarized from all evaluation data, as we claimed in our paper.

* Plot the history logs:

For AE reviewers, we also provide history logs in './log_his' directory on the GPU machines we provided. (you can also download them by running ```wget https://llumnix.oss-cn-beijing.aliyuncs.com/history_log.zip``` and then unzip to artifact directory ```unzip history_log.zip -d llumnix/artifact```). You can plot them and check the generated figure and claim file:
```
# In the 63_serving_performance directory
cd ./plot
python plot.py --log-path '../log_his'
```

* #### How to Check Reproducibility

To check the reproducibility, the figure11.png should be compared with the figure11_paper.pdf and the figure11.claim should be compared with the figure11_paper.claim.

* **Notes:**

**Result variance and multi-run.**
When doing the experiments for the paper, we noticed that some metrics in this experiment could have variance (especially the P99, as some fluctuation could be accumulated on the P99 metrics). Therefore, the numbers reported in the paper were averaged from multiple runs.
Due to the time constraints of the AE, we do understand that it might be difficult to replicate the experiments multiple times for the reviewers, as running all experiments once takes about 4 days.
As a result, if the experiments are not repeated many times, the absolute values of each evaluation metric might not be very close to the values reported in paper.
Despite the variance, the relative key improvements should be close to those shown in the paper. These relative key improvements can be quickly verified by reviewing the .claim file. We expect that the key improvements recorded in the .claim file will be close to the number shown in the paper through a few times of experiments.

Therefore, we recommend the artifact evaluators to run all the experiments once with 4 days. And check the overall reproducibility through comparing the figures and .claim files of each experiment. If you find that the reproducibility of some traces is not good enough, you can adjust the run.sh to run the corresponding traces multiple times and regenerate the .figure file and .claim file through running plot.py. The plot.py automatically collects and averages all the data generated after multiple runs.

**Manually kill and rerun experiments.**
During the experiments, we found a very low possibility that the experiments hung and we needed to kill them and rerun. If you find the log of an experiment were not updated, or the GPU utilization (through `nvidia-smi`) were zero for a long time, then the experiment could possibly have hung. You can kill the experiment by executing `bash ./artifact/kill_exp.sh` in the container on the head node, and after that you can re-launch the experiment.

### 6.4 Support for Priorities (Figure 13) (Estimated Running Time: 1h)

* Run script:
```
cd ./artifact/64_support_for_priorities
nohup bash run.sh &
```

* After the run.sh finished(1h), plot figure:
```
cd ./plot
python plot.py
```

The figure13.png and figure13.claim will be generated in the plot directory.

* Plot the history logs:
```
cd ./plot
python plot.py --log-path '../log_his'
```

* #### How to Check Reproducibility

To check the reproducibility, the figure13.png should be compared with the figure13_paper.pdf and the figure13.claim should be compared with the figure13_paper.claim.

### 6.5 Auto-scaling (Figure 14) (Estimated Running Time: 42h)

* Run script:
```
cd ./artifact/65_auto_scaling
nohup bash run.sh &
```

* After the run.sh finished(42h), plot figure:
```
cd ./plot
python plot.py
```

The figure14/15.png and figure14.claim will be generated in the plot directory.

* Plot the history logs:
```
cd ./plot
python plot.py --log-path '../log_his'
```

* #### How to Check Reproducibility

To check the reproducibility, the figure14/15.png should be compared with the figure14/15_paper.pdf and the figure14.claim should be compared with the figure14_paper.claim.


### 6.2 Migration Efficiency (Figure 10) (Estimated Running Time: 1h)

The initial implementation of our paper used vllm of version v0.1.7, which was used in all of the previous experiments. After the paper submission, we found a newer version (v0.2.0) could further reduce migration overhead. We have rerun the migration benchmark (Figure 10) using this new version.

Therefore, to reproduce the new results, you will need to re-install vllm.


* Install vllm==0.2.0:

**Important**: the command here should be executed from **outside** the container, i.e., on the host OS of head node. This command will re-install vllm on the four nodes automatically.

```
# on head node, outside of container
nohup bash ./artifact/set_vllm_newver.sh &
```

* **Go back to the container on the head node.** Run Script:
```
cd ./artifact/62_migration_efficiency
nohup bash run.sh &
```

* After the run.sh finished(1h), plot figure:
```
cd ./plot
python plot.py
```

* Plot the history logs:
```
cd ./plot
python plot.py --log-path '../log_his'
```

* #### How to Check Reproducibility

To check the reproducibility, the figure10.png should be compared with the figure10_paper.pdf and the figure10.claim should be compared with the figure10_paper.claim.

### Code Organization (to check functionality)
We list the key files that Llumnix creates or changes base on vLLM (0.1.7 scheduler, others 0.1.2).
```

Llumnix/
│
├── vllm/
│   │
│   ├── core/
│   │   ├── request_scheduler.py        # Global Scheduling Policy
│   │   ├── scheduler.py                # Local Scheduler
│   │   └── ...
│   │
│   ├── engine/
│   │   ├── async_llm_engine.py
│   │   ├── llm_engine_manager.py       # Global Scheduler
│   │   ├── llm_engine.py               # Llumlet
│   │   └── ...
│   │
│   ├── entrypoints/
│   │   ├── api_server.py
│   │   └── ...
│   │
│   ├── simulator/                      # Llumnix Simulator
│   │   └── ...
│   │
│   ├── worker/
│   │   ├── cache_engine.py             # Migration Executor
│   │   └── worker.py
│   │
│   ├── instance_info.py                # Instance Load
│   └── ...
│
└── ...
```