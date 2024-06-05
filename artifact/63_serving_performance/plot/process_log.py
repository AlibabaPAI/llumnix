import argparse
import os
import json
import numpy as np
import re
import pandas as pd
from collections import defaultdict
from vllm.simulator.profiling import *


def _pad_to_alignment(x, multiple_of):
    return x + ((-1*x) % multiple_of)

def get_req_data(json_filename, key):
    with open(json_filename, 'r') as f:
        latency_info = json.load(f)[0]
    req_data = latency_info[key]
    if key in ['prefill_latencies', 'decode_latencies']:
        req_data = [i / 1000 for i in req_data]

    return req_data

def get_mean_and_p99_latency(latencies):
    latencies = list(latencies)
    mean_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    return mean_latency, p99_latency

def parse_dir_name(log_filename):
    dir_path = os.path.dirname(log_filename)
    dir_name = os.path.basename(dir_path)

    qps_match = re.search(r"qps(\d+(\.\d+)?)", dir_name)
    imean_match = re.search(r"imean(\d+)", dir_name)
    omean_match = re.search(r"omean(\d+)", dir_name)
    migrate_match = re.search(r"migrate(\d+)", dir_name)
    defrag_match = re.search(r"defrag(\d+)", dir_name)

    qps = float(qps_match.group(1)) if qps_match else None
    imean = int(imean_match.group(1)) if imean_match else None
    omean = int(omean_match.group(1)) if omean_match else None
    migrate = int(migrate_match.group(1)) if migrate_match else None
    defrag = int(defrag_match.group(1)) if defrag_match else None

    trace = str(imean) + '_' + str(omean)

    dispatch = None
    if 'load' in dir_name:
        dispatch = 'local'
    elif 'balanced' in dir_name:
        dispatch = 'balanced'
    else:
        assert dispatch is not None, "Only support balanced/load dispatch"

    method = None
    if dispatch == 'local':
        if migrate == 0:
            method = 'INFaaS++'
        else:
            if defrag == 1:
                method = 'Llumnix'
            else:
                assert method is not None, "Only test Round-Robin/InFaaS++/Llumnix"
    else:
        method = 'Round-Robin'

    return qps, trace, method

def get_preemption_loss(log_filename):
    df = pd.read_csv(log_filename + "_req.csv").drop_duplicates()
    df = df.sort_values(by='timestamp')
    preempted_request_set = set()
    request_num = len(df["req_id"].drop_duplicates())
    preemption_loss_sum = 0
    last_killed_time = defaultdict(lambda: 0.0)
    last_killed_len = defaultdict(lambda: 0.0)

    database = ProfilingDatabase("/mnt/huangziming/artifact/vllm_017/vllm/vllm/simulator/profiling_result_new.pkl", False)
    profiling_result = database.get("llama-7b")
    sim_parallel_config = SimParallelConfig(1, 1)
    latency_mem = profiling_result.para_dict[sim_parallel_config]

    for _, row in df.iterrows():
        req_id = row["req_id"]
        if row["event"] == "prefill" and last_killed_time[req_id]:
            preemption_loss_sum += row["timestamp"] - last_killed_time[req_id]
            prompt_len = last_killed_len[req_id]
            prompt_len = _pad_to_alignment(prompt_len, 8)
            preemption_loss_sum += (latency_mem.prefill_latency[(1, prompt_len)][0] - latency_mem.decode_latency[(8, last_killed_len[req_id])][0]) / 1000
            preempted_request_set.add(req_id)
        elif row["event"] == "killed":
            last_killed_time[req_id] = row["timestamp"]
            last_killed_len[req_id] = row["output_len"]
    preemption_loss = preemption_loss_sum / request_num
    return preemption_loss

def get_evaluation_data(log_filename):
    # dir, json
    # trace_key, method(llumnix, infaas++, round-robin), qps
    # request/prefill/decode mean/p99, preemption loss

    # read dir
    qps, trace, method = parse_dir_name(log_filename)

    data_file.write("Trace: {}\n".format(trace))
    data_file.write("Method: {}\n".format(method))
    data_file.write("QPS: {:.2f}\n".format(qps))
    json_filename = os.path.splitext(log_filename)[0] + "_latency_info.json"
    data_keys = ['request_latencies', 'prefill_latencies', 'decode_latencies']
    key2metric = {'request_latencies': 'Request', 'prefill_latencies': 'Prefill', 'decode_latencies': 'Decode'}
    for data_key in data_keys:
        latencies = get_req_data(json_filename, data_key)
        mean_latency, p99_latency = get_mean_and_p99_latency(latencies)
        mean_metric_name = key2metric[data_key] + ' ' + 'Mean'
        p99_metric_name = key2metric[data_key] + ' ' + 'P99'
        data_file.write("{}: {:.4f}\n".format(p99_metric_name, p99_latency))
        data_file.write("{}: {:.4f}\n".format(mean_metric_name, mean_latency))

    preemption_loss = get_preemption_loss(log_filename)
    data_file.write("Preemption Loss: {:.4f}".format(preemption_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-filename', type=str)
    args = parser.parse_args()

    prefix, _ = os.path.splitext(args.log_filename)
    data_filename = prefix + '.data'
    data_file = open(data_filename, 'w')


    try:
        get_evaluation_data(args.log_filename)
    except Exception as e:
        print(e)
        data_file.write("Invalid Log!\n")
    finally:
        data_file.close()
