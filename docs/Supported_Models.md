# Supported Models

Llumnix serves as the request scheduling layer on top of the LLM backend engines. Therefore, all the models supported by the backend LLM engine should ideally be supported by Llumnix. We are also conducting full compatibility tests on different models.

Currently, Llumnix is developed on top of the vLLM (version 0.4.2), making its supported models identical to those of vLLM. Up to now, our primary testing of Llumnix has been conducted on Qwen and Llama models, including:

- Llama
- Llama2
- Llama3
- Qwen
- Qwen1.5
- Qwen2
- More models (not officially tested): [vLLM Supported Models](https://docs.vllm.ai/en/v0.6.3.post1/models/supported_models.html)

# Supported Backends

Currently, Llumnix supports vLLM as its backend LLM engine. However, Llumnix is designed for extensibility to various backend LLM inference engines. We will incorporate more inference engines in the future.

- vLLM (v0.6.3.post1)