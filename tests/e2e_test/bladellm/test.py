from blade_llm import LLM, SamplingParams
from blade_llm.protocol import GenerateStreamResponse

import ray

model = '/mnt/dataset/Qwen--Qwen1.5-7B-tiny-random'
sampling_params = {
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "top_k": 1,
        "max_new_tokens": 20,
    }
@ray.remote(num_gpus=1)
def run_bladellm(model, sampling_params):
    bladellm_output = {}
    raw_bladellm = LLM(model=model)
    outputs = raw_bladellm.submit_request("0", "Hello, my name is", SamplingParams(**sampling_params))

    for output in outputs:
        bladellm_output[output.prompt] = output.prompt + output.outputs[0].text

    return bladellm_output

bladellm_output = ray.get(run_bladellm.remote(model, sampling_params))
print(bladellm_output)