# Artifact for Llumnix (OSDI'24)

## Overview
This is the artifact for the paper "Llumnix: Dynamic Scheduling for Large Language Model Serving". We are going to reproduce the main results in the paper.
## Environment Setup
Install Llumnix by running
```
pip install -e .
```
in the root folder of `llumnix` project.

Lauch the ray runtime
```
ray start --head
```
connect to this ray runtime from another node
```
ray start --address="$head_address"
```

## Reproduce the Evaluation Results

### 6.3 Serving Performance (Figure 11)


* (Run):
```
cd ./artifact/63_serving_performance
nohup bash run.sh &
```
We can get all the results in figure11 through running the run.sh script.

The run.sh scipt runs the llumnix_exp script for multiple times; During each run, it collects a data point for figure11 and generates some log files in the coresponding sub-directory of ./log directory.

In the end of the llumnix_exp script, we runs ./plot/process_log.py to obtain evaluation data (e.g: Requet P99/Mean, Prefill P99/Mean etc.). The evaluation data is saved to the file ends with .data in the log directory.

* (Plot):
```
cd ./plot
python plot.py
```

The plot.py reads all the .data files generated in the previous step to collect all evaluation data. Then, the plot.py plots figure11.png and generates figure11.claim. The .claim file summarizes the key improvements obtained from all evaluation data, as we claimed in our paper.

To check the reproducibility, the figure11.png should be compared with the figure11_paper.pdf and the figure11.claim should be compared with the figure11_paper.claim.

Due to time constraints, it is challenging to replicate the experiments multiple times during the ae, as each run takes approximately two days. Additionally, the performance of serving is quite unstable (especialy the P99 metrics). As a result, if the experiments are not repeated many times, the abosulte values of each evaluation metric might not be very close to the values reported in paper, especially the monotonicity of the values. However, the relative key improvements should be close to those shown in the paper. These relative key improvements can be quickly verified by reviewing the .claim file. We expect that the key improvements recorded in the .claim file will be close to the number shown in the paper through serveral times of experiments.

### 6.4 Support for Priorities (Figure 13)

* (Run):
```
cd ./artifact/64_support_for_priorities
nohup bash run.sh &
```

* (Plot):
```
cd ./plot
python plot.py
```

### 6.5 Auto-scaling (Figure 14)

* (Run):
```
cd ./artifact/65_auto_scaling
nohup bash run.sh &
```

* (Plot):
```
cd ./plot
python plot.py
```
### 6.2 Migration Efficiency (Figure 10)
We use a newer vllm version(v0.2.0) to futher reduce migration overhead.

* Install vllm==0.2.0

run bash `nohup bash ./artifact/set_vllm_newver.sh &` or install manually

```
cd ./artifact/62_migration_efficiency/vllm_020/vllm
pip install -e.
```

* (Run):
```
cd ./artifact/62_migration_efficiency
nohup bash run.sh &
```

* (Plot):
```
cd ./plot
python plot.py
```