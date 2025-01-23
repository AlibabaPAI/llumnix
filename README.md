<h1 align="center">
Llumnix
</h1>

<h3 align="center">
Efficient and easy <i>multi-instance</i> LLM serving
</h3>

---

## 🔥 Latest News

- [2025.1] We updated vLLM to version v0.6.3.post1.
- [2024.11] Llumnix v0.1.0 launched!
- [2024.7] We officially released the first version of Llumnix.
- [2024.6] We released our OSDI '24 [research paper](https://arxiv.org/abs/2406.03243) on arxiv.

## 🚀 Why Llumnix

Llumnix is a cross-instance request scheduling layer built on top of LLM inference engines such as [vLLM](https://github.com/vllm-project/vllm).

Llumnix provides optimized multi-instance serving performance in terms of:

- *Low latency*
  - **Reduced time-to-first-token** (TTFT) and queuing delays with less memory fragmentation
  - **Reduced time-between-tokens** (TBT) and preemption stalls with better load balancing
- *High throughput*
  - Integration with state-of-the-art inference engines
  - Support for techniques like prefill-decoding disaggregation

Llumnix achieves this with:

- Dynamic, fine-grained, KV-cache-aware scheduling
- Continuous **rescheduling** across instances
  - Enabled by a KV cache migration mechanism with near-zero overhead
  - Exploited for continuous load balancing, de-fragmentation, and prefill-decoding disaggregation

Llumnix is easy to use with:

- Minimal code changes required for vanilla vLLM deployments

- Seamless integration with existing multi-instance deployment platforms

- Fault tolerance, elasticity, and high service availability

- Extensibility to more inference engines and scheduling policies


## Getting Started

If you are already utilizing vLLM for multi-instance LLM serving deployments, simply replace the vLLM serving deployment command `python -m entrypoints.vllm.api_server ...` for each instance with the command provided below:
```
python -m llumnix.entrypoints.vllm.api_server \
    --host $HOST \
    --port $PORT \
    ...
```
During the serving deployment execution, Llumnix will automatically configure itself and serve as the request scheduling layer on top of the multiple vLLM engine instances.

Visit our [documentation](./docs/) to get started:
- [Quick Start](./docs/Quickstart.md)
- [Supported Models](./docs/Supported_Models.md)
- [Fault Tolerance](./docs/Fault_Tolerance.md)
- [Simulator](./docs/Simulator.md)
- [Prefill-decoding Disaggregation](./docs/Prefill-decoding_Disaggregation.md)

## Performance
We evaluate the performance of the KV-cache-aware load-balancing scheduler and migration mechanism of Llumnix with 16 Qwen2.5-7B instances (each using an A10-24GB GPU) and 16 Llama2-13B instances (each using an A800-80GB GPU).

We use Poisson distributions with different request rates to generate request arrivals. For the input/output lengths of requests, we use ShareGPT dataset.

<div align=center>
<img src="./docs/v0.1.0_benchmark.png" align="center" width=80%/>
</div>

Llumnix outperforms a simple round-robin scheduler in TTFT (prefill) by up to 6.4x and 12.1x for mean and P99, and 12% for P99 TBT (decode). Llumnix also shows significantly shorter average preemption stalls (by two orders of magnitude).

With the KV-cache-aware load-balancing scheduler and the migration mechanism, Llumnix also outperforms a simple load balancing scheduler based on queue sizes in TTFT (prefill) by up to 4.6x and 9.1x for mean and P99, and 15% for P99 TBT (decode).

## Roadmap

Llumnix is currently in an alpha stage. Moving forward, we have work items planned including but not limited to:

- Architectural improvement: improving the scalability and efficiency of distributed serving and coordination;
- Policy optimization: better dispatching, migration, auto-scaling policies;
- New features: incorporating more inference engine features;
- Engineering: testing, CI/CD, etc.

## Publication

Please cite our paper if you use Llumnix in your research:

```bibtex
@inproceedings{sun2024llumnix,
  title={Llumnix: Dynamic Scheduling for Large Language Model Serving},
  author={Biao Sun and Ziming Huang and Hanyu Zhao and Wencong Xiao and Xinyi Zhang and Yong Li and Wei Lin},
  booktitle={18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
  year={2024}
}
```

## License

Llumnix is licensed under the Apache 2.0 License.