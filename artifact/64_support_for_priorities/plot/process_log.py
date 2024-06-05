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
    if key in ['prefill_latencies', 'decode_latencies','priority_prefill_latencies', 'priority_decode_latencies']:
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

    cv_match = re.search(r"cv(\d+(\.\d+)?)", dir_name)
    imean_match = re.search(r"imean(\d+)", dir_name)
    omean_match = re.search(r"omean(\d+)", dir_name)
    migrate_match = re.search(r"migrate(\d+)", dir_name)
    defrag_match = re.search(r"defrag(\d+)", dir_name)

    cv = float(cv_match.group(1)) if cv_match else None
    imean = int(imean_match.group(1)) if imean_match else None
    omean = int(omean_match.group(1)) if omean_match else None
    migrate = int(migrate_match.group(1)) if migrate_match else None
    defrag = int(defrag_match.group(1)) if defrag_match else None

    trace = str(imean) + '_' + str(omean)

    if "priority0" in dir_name:
        method="Llumnix-base"
    else:
        method="Llumnix"

    return cv, trace, method


def get_evaluation_data(log_filename, priority_type):
    # dir, json
    # trace_key, method(llumnix, infaas++, round-robin), qps
    # request/prefill/decode mean/p99, preemption loss

    # read dir
    cv, trace, method = parse_dir_name(log_filename)

    data_file.write("Trace: {}\n".format(trace))
    data_file.write("Method: {}\n".format(method))
    data_file.write("CV: {:.2f}\n".format(cv))
    json_filename = os.path.splitext(log_filename)[0] + "_latency_info.json"
    if priority_type == 0:
        data_keys = ['request_latencies', 'prefill_latencies', 'decode_latencies']
        key2metric = {'request_latencies': 'Request', 'prefill_latencies': 'Prefill', 'decode_latencies': 'Decode'}
    else:
        data_keys = ['priority_request_latencies', 'priority_prefill_latencies', 'priority_decode_latencies']
        key2metric = {'priority_request_latencies': 'Request', 'priority_prefill_latencies': 'Prefill', 'priority_decode_latencies': 'Decode'}
    for data_key in data_keys:
        latencies = get_req_data(json_filename, data_key)
        mean_latency, p99_latency = get_mean_and_p99_latency(latencies)
        mean_metric_name = key2metric[data_key] + ' ' + 'Mean'
        p99_metric_name = key2metric[data_key] + ' ' + 'P99'
        data_file.write("{}: {:.4f}\n".format(p99_metric_name, p99_latency))
        data_file.write("{}: {:.4f}\n".format(mean_metric_name, mean_latency))

    decode_time = np.mean(get_req_data(json_filename, "inference_latencies" if priority_type == 0 else "priority_inference_latencies"))
    data_file.write("Decode Execution Time: {:.4f}".format(decode_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-filename', type=str)
    parser.add_argument('--priority-type', type=int)
    args = parser.parse_args()

    prefix, _ = os.path.splitext(args.log_filename)
    data_filename = prefix + f'_{args.priority_type}.data'
    data_file = open(data_filename, 'w')

    # for test
    # get_evaluation_data(args.log_filename)

    try:
        get_evaluation_data(args.log_filename, args.priority_type)
    except Exception as e:
        print(e)
        data_file.write("Invalid Log!\n")
    finally:
        data_file.close()
