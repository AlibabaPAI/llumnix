#!/usr/bin/env python3


from transformers import AutoTokenizer
from typing import List, Callable
import argparse
import itertools
import random
import numpy as np
import heapq
import os
import matplotlib.pyplot as plt
import json
import queue

from vllm.engine.arg_utils import EngineManagerArgs
from vllm.simulator.llm_engine_manager import LLMEngineManager
from vllm.simulator.profiling import *
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from benchmark_throughput import (GenerationBackend, MeasureLatency, get_wait_time, calculate_cdf, plot_latency_slo,\
                                plot_len_cdf, calculate_slo, plot_latency_cdf, load_prompts, gen_random_prompts_return_lens, \
                                gen_random_response_lens, gen_random_session_id,save_all_latencies_npy, plot_instance)

class PriorityQueue:
    def __init__(self):
        self.store = []

    def put(self, value):
        heapq.heappush(self.store, value)

    def get(self):
        return heapq.heappop(self.store)

    def __len__(self):
        return len(self.store)

    def __bool__(self):
        return True if self.store else False

class TimedCoroutine:
    """A coroutine that will be woken up at specific time."""
    def __init__(self,
                 wake_up_time: float,
                 func: Callable,
                 *args,
                 **kwargs,):
        self.wake_up_time = wake_up_time
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        return self.func(*self.args, **self.kwargs)

    def __lt__(self, other):
        return self.wake_up_time < other.wake_up_time

    def __str__(self):
        if hasattr(self.func, "__name__"):
            name = self.func.__name__
        elif hasattr(self.func, "func"):
            name = self.func.func.__name__
        else:
            name = ""
        return f"TimedCoroutine(wake_up_time={self.wake_up_time}, func={name})"

class Request:
    def __init__(self, request_id, prompt, arrival_time, prompt_len, response_len, priority_type):
        self.request_id = request_id
        self.prompt = prompt
        self.arrival_time = arrival_time
        self.token_arrival_time = []
        self.inference_latency = None
        self.finished = False
        self.prompt_len = prompt_len
        self.response_len = response_len
        self.priority_type = priority_type
    
    def get_pertoken_latency(self):
        return self.get_req_latency()/self.response_len

    def get_prefill_latency(self):
        return (self.token_arrival_time[0] - self.arrival_time) * 1000
    
    def get_avg_decode_latency(self):
        decode_latencies = np.diff(self.token_arrival_time)
        mean_latency = 0 if len(decode_latencies) == 0 else np.mean(decode_latencies)
        return mean_latency * 1000

    def get_sum_decode_latency(self):
        decode_latencies = np.diff(self.token_arrival_time)
        decode_latency = 0 if len(decode_latencies) == 0 else np.sum(decode_latencies)
        return decode_latency * 1000

    def get_req_latency(self):
        return self.token_arrival_time[-1] - self.arrival_time
    
    def all_token_latency_pairs(self):
        token_latencies = np.diff(self.token_arrival_time)
        token_latencies *= 1000
        token_latencies = np.insert(token_latencies, 0, self.get_prefill_latency())
        return np.stack((self.token_arrival_time, token_latencies), axis=1)

def request_arrival_time_gen(num_request: int, qps: float, distribution="uniform", coefficient_variation: float = 0.0):
    arrival_time_list = [0]
    for _ in range(num_request - 1):
        arrival_time = arrival_time_list[-1]
        if distribution != "burst":
            arrival_time += get_wait_time(1.0 / qps, distribution, coefficient_variation)
        arrival_time_list.append(arrival_time)
    return arrival_time_list

def update_from_step(engine_manager, request_dict, timestamp, instance_id, post_process_args):
    outputs, instance_info = engine_manager.instances[instance_id].post_process(*post_process_args)
    engine_manager._update_instance_info(timestamp, instance_info)
    for output in outputs:
        request_id = output.request_id
        request = request_dict[request_id]
        request.token_arrival_time.append(timestamp)
        request.finished = output.finished
        request.inference_latency = output.total_inference_time

def calculate_throughput_sim(requests, dur_s, backend, tokenizer, median_token_latency, median_e2e_latency, median_inference_latency, 
                         all_e2e_latencies, all_per_token_latencies, all_inference_latencies, log_latencies, fail_on_response_failure):
    responses = requests
    prompt_token_count = sum([request.prompt_len for request in requests])
    response_token_count = sum([request.response_len for request in requests])

    all_prompt_lens = [request.prompt_len for request in requests]
    all_response_lens = [request.response_len for request in requests]
    all_total_tokens = [all_prompt_lens[i] + all_response_lens[i] for i in range(len(all_prompt_lens))]
    
    all_waiting_latencies = [all_e2e_latencies[i] - all_inference_latencies[i] for i in range(len(all_e2e_latencies))]

    throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    # print(f'throughput_tok_s {throughput_tok_s:.02f}')
    qps = len(responses) / dur_s
    msg1 = f'backend {backend} dur_s {dur_s:.04f} tokens_per_s {throughput_tok_s:.02f} qps {qps:.04f}\n'
    msg2 = f'successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}\n'
    msg3 = f'{median_token_latency=:.04f}, {median_e2e_latency=:.04f}, {median_inference_latency=:.04f}\n'
    msg = msg1 + msg2 + msg3
    if log_latencies:
        msg += f'{all_total_tokens=}\n{all_prompt_lens=}\n{all_response_lens=}\n'
        msg += f'{all_e2e_latencies=}\n{all_per_token_latencies=}\n{all_inference_latencies=}\n{all_waiting_latencies=}\n'
        msg += f'{all_request_ids=}\n{all_decode_latencies=}\n'
    print(msg)
    return throughput_tok_s

def plot_request_per_s(results_filename: str, request_arrival_time: List[float], interval: int=60):
    fig_filename = os.path.splitext(results_filename)[0] + "_latency.png"
    ts = np.arange(0, max(request_arrival_time),step=interval)
    hist, bin_edges = np.histogram(request_arrival_time, bins=len(ts))
    fig, ax = plt.subplots()
    ax.plot(ts, hist)
    ax.set_xlabel("timestamp(s)")
    ax.set_ylabel(f"reqeust per {interval}s)")
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def benchmark(
    engine_manager: LLMEngineManager,
    backend: GenerationBackend,
    tokenizer,
    prompts: List[str],
    allow_variable_generation_length: bool,
    verbose: bool,
    results_filename: str,
    port: int,
    distribution: str,
    qps: float,
    coefficient_variation: float,
    log_latencies: bool,
    fail_on_response_failure: bool,
    prefill_SLO: float,
    priority_ratio: float,
):
    m = MeasureLatency()
    m1 = MeasureLatency()

    if distribution == "burst":
        qps = float('inf')
    if distribution != "gamma":
        coefficient_variation = 0.0

    print(
        f'Starting with backend={backend}, num_prompts={len(prompts)}, allow_variable_generation_length={allow_variable_generation_length}')
    print(f'traffic distribution={distribution}, qps={qps}, coefficient_variation={coefficient_variation}')

    # async_prompts = async_request_gen(
    #     iter(prompts), qps=qps, distribution=distribution, coefficient_variation=coefficient_variation)
    request_arrival_time = request_arrival_time_gen(
        len(prompts), qps=qps, distribution=distribution, coefficient_variation=coefficient_variation)
    #Dict[request_id -> Request]
    request_dict = {}
    func_queue = PriorityQueue()
    for i in range(len(prompts)):
        prompt, prompt_len, expected_response_len, session_id = prompts[i]
        best_of = random.randint(1,1)
        use_beam_search = best_of > 1
        params_dict = {
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_k": -1 if use_beam_search else 1,
            "max_tokens": max(expected_response_len, 1),
            "ignore_eos": True,
        }
        sampling_params = SamplingParams(**params_dict)
        request_id = random_uuid()
        priority_type = 1 if random.random() < priority_ratio else 0
        # print(f"priority_ratio:{priority_ratio}, priority_type:{priority_type}")
        request_dict[request_id] = Request(request_id, prompt, request_arrival_time[i], prompt_len, expected_response_len, priority_type = priority_type)
        tc = TimedCoroutine(request_arrival_time[i], engine_manager.generate, session_id, prompt, sampling_params, request_id,\
         prompt_token_ids=None, arrival_time=request_arrival_time[i], priority_type = priority_type)
        #  if engine_manager.enable_migrate else 0

        func_queue.put(tc)
    instance_free_time = [0] * engine_manager.max_replicas
    print(f"arrival: {request_arrival_time}")
    # plot_request_per_s(results_filename, request_arrival_time)
    non_dispatch_request_queue = queue.Queue()
    while func_queue:
        tc = func_queue.get()
        if tc.func.__name__ == "generate":
            request_id = tc.args[-1]
            # print(f"request {request_id} generate")
            tc.run()
            step_tc = TimedCoroutine(tc.wake_up_time, engine_manager.step, tc.wake_up_time, request_id)
            func_queue.put(step_tc)
            tmp_queue = queue.Queue()
            while not non_dispatch_request_queue.empty():
                # print("!!!")
                # import pdb; pdb.set_trace()
                request_id = non_dispatch_request_queue.get()
                if request_id not in engine_manager.request_instance:
                    tmp_queue.put(request_id)
                    continue
                step_tc = TimedCoroutine(tc.wake_up_time, engine_manager.step, tc.wake_up_time, request_id)
                func_queue.put(step_tc)
            non_dispatch_request_queue = tmp_queue
        elif tc.func.__name__ == "step":
            request_id = tc.args[-1]
            # print(f"request {request_id} step")
            if request_id not in engine_manager.request_instance:
                # print(f"!!!!!! push request {request_id} to non_dispatch_request_queue")
                # import pdb; pdb.set_trace()
                non_dispatch_request_queue.put(request_id)
                continue
            instance_id = engine_manager.request_instance[request_id]
            next_timestamp = 0
            if request_dict[request_id].finished:
                continue
            elif tc.wake_up_time >= instance_free_time[instance_id]:
                post_process_args = tc.run()
                next_timestamp = post_process_args[-1]
                instance_free_time[instance_id] = next_timestamp
                update_tc = TimedCoroutine(next_timestamp, update_from_step, engine_manager, request_dict, next_timestamp, instance_id, post_process_args)
                func_queue.put(update_tc)
            instance_id = engine_manager.request_instance[request_id]
            # if migration happened, next_step_time may change
            next_step_time = max(instance_free_time[instance_id], next_timestamp) + 0.004
            step_tc = TimedCoroutine(next_step_time, engine_manager.step, next_step_time, request_id)
            func_queue.put(step_tc)
        elif tc.func.__name__ == "update_from_step":
            instance_info = tc.args[2]
            tc.run()
            tmp_queue = queue.Queue()
            while not non_dispatch_request_queue.empty():
                # print("!!!")
                # import pdb; pdb.set_trace()
                request_id = non_dispatch_request_queue.get()
                if request_id not in engine_manager.request_instance:
                    tmp_queue.put(request_id)
                    continue
                step_tc = TimedCoroutine(tc.wake_up_time, engine_manager.step, tc.wake_up_time, request_id)
                func_queue.put(step_tc)
            non_dispatch_request_queue = tmp_queue
    low_priority_list=[]
    high_priority_list=[]
    for request in request_dict.values():
        if request.priority_type == 0:
            low_priority_list.append(request)
            m._prefill_token_latencies.append(request.get_prefill_latency())
            m._per_token_latencies.append(request.get_pertoken_latency())
            m._latencies.append(request.get_req_latency())
            m._decode_token_latencies.append(request.get_avg_decode_latency())
            m._decode_latencies.append(request.get_sum_decode_latency())
            m._inference_latencies.append(request.inference_latency)
            m._all_latencies.append(request.all_token_latency_pairs())
        else:
            high_priority_list.append(request)
            m1._prefill_token_latencies.append(request.get_prefill_latency())
            m1._per_token_latencies.append(request.get_pertoken_latency())
            m1._latencies.append(request.get_req_latency())
            m1._decode_token_latencies.append(request.get_avg_decode_latency())
            m1._decode_latencies.append(request.get_sum_decode_latency())
            m1._inference_latencies.append(request.inference_latency)
            m1._all_latencies.append(request.all_token_latency_pairs())
    print(f"0 len:{len(m._latencies)} 1 len:{len(m1._latencies)}")
    # m._prefill_token_latencies = [request.get_prefill_latency() for request in request_dict.values()]
    # m._per_token_latencies = [request.get_pertoken_latency() for request in request_dict.values()]
    # m._latencies = [request.get_req_latency() for request in request_dict.values()]
    # m._decode_token_latencies = [request.get_avg_decode_latency() for request in request_dict.values()]
    # m._inference_latencies = [request.inference_latency for request in request_dict.values()]
    # m._all_latencies=[request.all_token_latency_pairs() for request in request_dict.values()]
    dur_s = np.max([request.token_arrival_time[-1] for request in request_dict.values()])

    median_token_latency = np.median(m._per_token_latencies)
    median_e2e_latency = np.median(m._latencies)
    median_inference_latency = np.median(m._inference_latencies)

    throughput = calculate_throughput_sim(high_priority_list, dur_s, backend, tokenizer, median_token_latency, median_e2e_latency,
                         median_inference_latency,
                         m1._latencies, m1._per_token_latencies, m1._inference_latencies, log_latencies, fail_on_response_failure)
    calculate_cdf(m._latencies)
    plot_latency_cdf(m._latencies, m._prefill_token_latencies, m._decode_token_latencies, results_filename, prefill_SLO)
    plot_latency_cdf(m1._latencies, m1._prefill_token_latencies, m1._decode_token_latencies, results_filename, prefill_SLO, 1)

    calculate_slo(m._latencies, m._inference_latencies)
    plot_latency_slo(m._latencies, m._inference_latencies, results_filename)
    plot_latency_slo(m1._latencies, m1._inference_latencies, results_filename, 1)
    save_all_latencies_npy(m._all_latencies, results_filename)
    avg_instance_num = plot_instance(results_filename)

    return throughput, (m._prefill_token_latencies, m1._prefill_token_latencies), (m._decode_token_latencies, m1._decode_token_latencies), \
        (m._inference_latencies, m1._inference_latencies), avg_instance_num, (m._latencies, m1._latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--backend', type=GenerationBackend,
                        choices=[e.name for e in GenerationBackend], required=True)
    parser.add_argument('--results_filename', type=str, default='log')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--random_prompt_lens_mean', type=int)
    parser.add_argument('--random_prompt_lens_range', type=int)
    parser.add_argument('--variable_prompt_lens_distribution', choices=[
                        "uniform", "exponential", "capped_exponential", "zipf"], default="uniform")
    parser.add_argument('--random_prompt_count', type=int)

    parser.add_argument(
        '--distribution', choices=["burst", "uniform", "poisson", "gamma"], default="burst")
    parser.add_argument('--qps', type=float, default=4.0)
    parser.add_argument('--coefficient_variation', type=float, default=0.0)
    parser.add_argument('--log_latencies', action="store_true",
                        help="Whether or not to write all latencies to the log file.")
    parser.add_argument('--fail_on_response_failure', action="store_true",
                        help="Whether or not to fail the benchmarking script if any request fails")

    parser.add_argument('--variable_response_lens_mean', type=int)
    parser.add_argument('--variable_response_lens_range', type=int)
    parser.add_argument('--variable_response_lens_distribution', choices=[
                        "uniform", "exponential", "capped_exponential", "zipf"], default="uniform")

    parser.add_argument('--gen_random_session_id', action='store_true')
    parser.add_argument('--session_0_ratio', type=float, default=0.5)
    parser.add_argument('--new_session_ratio', type=float, default=0.2)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompts_filename', type=str)
    group.add_argument('--gen_random_prompts', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--allow_variable_generation_length',
                       action='store_true')
    group.add_argument('--fixed_max_tokens', type=int)

    parser.add_argument('--print-generation-lens-and-exit',
                        action='store_true')

    parser.add_argument('--prefill_SLO', type=float, default=10.0)

    parser.add_argument('--priority_ratio', type=float, default=0.0)

    # parser = AsyncEngineArgs.add_cli_args(parser)
    # args = parser.parse_args()

    # engine_args = AsyncEngineArgs.from_cli_args(args)
    # engine = AsyncLLMEngine.from_engine_args(engine_args)

    parser = EngineManagerArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = EngineManagerArgs.from_cli_args(args)
    engine_manager = LLMEngineManager.from_engine_args(engine_args)
    # parser.add_argument('--calculate_begin_ratio', type=float, default=0.5)
    # parser.add_argument('--calculate_end_ratio', type=float, default=0.8)

    if args.gen_random_prompts:
        assert args.random_prompt_count is not None

    backend = GenerationBackend[args.backend]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(tokenizer)

    if args.prompts_filename:
        prompts = load_prompts(args.prompts_filename)
        prompt_lens = itertools.repeat(-1)
        num_prompts = len(prompts)
    elif args.gen_random_prompts:
        num_prompts = args.random_prompt_count
        random.seed(0xCADE)
        np.random.seed(0xCADE)
        prompts, prompt_lens = gen_random_prompts_return_lens(
            tokenizer,
            distribution=args.variable_prompt_lens_distribution,
            len_mean=args.random_prompt_lens_mean,
            len_range=args.random_prompt_lens_range,
            num_prompts=num_prompts,
            vocab_ids_to_exclude=tokenizer.all_special_ids,
        )
    else:
        raise ValueError("unknown prompts")

    if args.allow_variable_generation_length:
        response_lens = gen_random_response_lens(
            args.variable_response_lens_distribution, args.variable_response_lens_mean, args.variable_response_lens_range, num_prompts=num_prompts)
        args.fixed_max_tokens = -1
    else:
        response_lens = [args.fixed_max_tokens for _ in range(num_prompts)]

    for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
        total = prompt_len + gen_len
        # @@@
        # if total > 2048:
        #     print(f'truncating long prompt+gen_len {prompt_len=} {gen_len=}')
        #     gen_len = 2048 - prompt_len
        if total > 32768:
            print(f'truncating long prompt+gen_len {prompt_len=} {gen_len=}')
            gen_len = 32768 - prompt_len
        response_lens[i] = gen_len

    if args.print_generation_lens_and_exit:
        print(f'{prompt_lens=}')
        print(f'{response_lens=}')
        print('Exiting...')
        return

    if args.verbose or True:
        print('prompt lens', sorted(list(prompt_lens)))
        print('response lens', sorted(list(response_lens)))

        total_tokens = []
        for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
            total_tokens.append(prompt_len + gen_len)

        print('total tokens', sorted(list(total_tokens)))
    
    plot_len_cdf(prompt_lens, response_lens, total_tokens, args.results_filename)
    
    if args.gen_random_session_id:
        session_ids = gen_random_session_id(args.new_session_ratio, args.session_0_ratio, num_prompts=num_prompts)
    else:
        session_ids = [-1] * num_prompts

    prompts = list(zip(prompts, prompt_lens, response_lens, session_ids))

    throughput, prefill_token_latencies, decode_token_latencies, inference_latencies, avg_instance_num, request_latencies, request_ids, decode_latencies = benchmark(
        engine_manager,
        backend,
        tokenizer,
        prompts,
        args.allow_variable_generation_length,
        args.verbose,
        args.results_filename,
        args.port,
        args.distribution,
        args.qps,
        args.coefficient_variation,
        args.log_latencies,
        args.fail_on_response_failure,
        args.prefill_SLO,
        args.priority_ratio,
    )
    file_name = os.path.splitext(args.results_filename)[0] + "_latency_info.json"
    results = []
    
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = prefix_dir + f"/latency_info_{'enable_migrate' if args.enable_migrate else 'disable_migrate'}_{current_time}.json"
    try:
        with open(file_name, 'r') as f:
            results = json.load(f)
    except json.decoder.JSONDecodeError:
        pass
    except FileNotFoundError:
        os.mknod(file_name)

    with open(file_name, 'w') as f:
        results.append({"qps": args.qps, "cv": args.coefficient_variation, "scale_up_threshold":args.scale_up_threshold,
                        "request_latencies": request_latencies[0], "inference_latencies": inference_latencies[0],
                        "prefill_latencies": prefill_token_latencies[0], "decode_latencies": decode_token_latencies[0],
                        "throughput": throughput, "instance_num": avg_instance_num, 
                        "priority_request_latencies": request_latencies[1], "priority_inference_latencies": inference_latencies[1],
                        "priority_prefill_latencies": prefill_token_latencies[1], "priority_decode_latencies": decode_token_latencies[1]})
        json.dump(results, f)

if __name__ == '__main__':
    main()
