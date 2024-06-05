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

    cv_match = re.search(r"cv(\d+(\.\d+)?)", dir_name)
    qps_match = re.search(r"qps(\d+(\.\d+)?)", dir_name)
    imean_match = re.search(r"imean(\d+)", dir_name)
    omean_match = re.search(r"omean(\d+)", dir_name)
    migrate_match = re.search(r"migrate(\d+)", dir_name)
    defrag_match = re.search(r"defrag(\d+)", dir_name)
    scale_match = re.search(r"scale(\d+)", dir_name)

    cv = float(cv_match.group(1)) if cv_match else None
    qps = float(qps_match.group(1)) if qps_match else None
    imean = int(imean_match.group(1)) if imean_match else None
    omean = int(omean_match.group(1)) if omean_match else None
    migrate = int(migrate_match.group(1)) if migrate_match else None
    defrag = int(defrag_match.group(1)) if defrag_match else None
    scale_match = int(scale_match.group(1)) if scale_match else None

    trace = str(imean) + '_' + str(omean)

    if migrate:
        method="Llumnix"
    else:
        method="INFaaS++"

    return qps, cv, scale_match, trace, method


def get_evaluation_data(log_filename):
    # dir, json
    # trace_key, method(llumnix, infaas++, round-robin), qps
    # request/prefill/decode mean/p99, preemption loss

    # read dir
    qps, cv, scale, trace, method = parse_dir_name(log_filename)

    data_file.write("Trace: {}\n".format(trace))
    data_file.write("Method: {}\n".format(method))
    data_file.write("Scale: {}\n".format(scale))
    if cv > 1:
        data_file.write("CV: {:.2f}\n".format(cv))
    else :
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

    data_file.write("Avg Instance Num: {:.4f}".format(get_req_data(json_filename, "instance_num")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-filename', type=str)
    args = parser.parse_args()

    prefix, _ = os.path.splitext(args.log_filename)
    data_filename = prefix + f'_.data'
    data_file = open(data_filename, 'w')

    try:
        get_evaluation_data(args.log_filename)
    except Exception as e:
        print(e)
        data_file.write("Invalid Log!\n")
    finally:
        data_file.close()
