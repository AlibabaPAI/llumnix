import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import argparse


def get_avg_latency(results_filename, start_time , stop_time, add_begin_time = False):
    df = pd.read_csv(results_filename).drop_duplicates()
    df = df[df["inference_type"]!="prefill"]
    df = df[df["instance_id"]==0]
    latency_list = []
    first_row = df.iloc[0]
    begin_time = first_row["timestamp"]
    if add_begin_time:
        start_time += begin_time
        stop_time += begin_time

    for idx, row in df.iterrows():
        if row["timestamp"] > start_time and row["timestamp"] < stop_time:
            latency_list.append(row["latency"])
    # print(f"avg:{np.mean(latency_list)}")
    return np.mean(latency_list), begin_time

def get_migration_timestamp(results_filename):
    df = pd.read_csv(results_filename).drop_duplicates()
    df = df.sort_values("timestamp", ascending=True)
    stage0_begin_timestamp = None
    stage1_begin_timestamp = None
    stage1_end_timestamp = None
    for idx, row in df.iterrows():
        if row["event"] == "migrate_out_stage_0" and not stage0_begin_timestamp:
            stage0_begin_timestamp = row["timestamp"]
        if row["event"] == "migrate_out_stage_1" and not stage1_begin_timestamp:
            stage1_begin_timestamp = row["timestamp"]
        if row["event"] == "migrate_in" and not stage1_end_timestamp:
            stage1_end_timestamp = row["timestamp"]

    return stage0_begin_timestamp, stage1_begin_timestamp, stage1_end_timestamp

def get_migration_stage_data(log_path, len_list):
    stage1_latency = []
    stage0_latency = []
    decode_latency = []
    migration_begin_offset = []
    for seq_len in len_list:
        for root, dirs, files in os.walk(log_path):
            if root.endswith(str(seq_len)):
                for file in files:
                    if file.endswith('req.csv'):
                        filepath = os.path.join(root, file)
                        stage0_begin_timestamp, stage1_begin_timestamp, stage1_end_timestamp = get_migration_timestamp(filepath)
                        stage0_latency.append(stage1_begin_timestamp-stage0_begin_timestamp)
                        stage1_latency.append(stage1_end_timestamp - stage1_begin_timestamp)
                for file in files:
                    if file.endswith('instance.csv'):
                        filepath = os.path.join(root, file)
                        avg_decode_latency_migration, begin_time = get_avg_latency(filepath, stage0_begin_timestamp, stage1_begin_timestamp)
                        decode_latency.append(avg_decode_latency_migration)
                        migration_begin_offset.append(stage0_begin_timestamp - begin_time)

    return stage0_latency, stage1_latency, decode_latency, migration_begin_offset

def get_recompute_data(log_path, len_list):
    latency_list = []
    for seq_len in len_list:
        for root, dirs, files in os.walk(log_path):
            if root.endswith(str(seq_len)):
                for file in files:
                    if file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r') as file:
                            data_dict = json.load(file)
                            latency_list.append(data_dict[0]["inference_latencies"][0])
    return latency_list

def get_normal_decode_latency(log_path, len_list,  migration_begin_offset, migration_time):
    latency_list = []
    idx = 0
    for seq_len in len_list:
        for root, dirs, files in os.walk(log_path):
            if root.endswith(str(seq_len)):
                for file in files:
                    if file.endswith('instance.csv'):
                        filepath = os.path.join(root, file)
                        avg_decode_latency_migration, _ = get_avg_latency(filepath, migration_begin_offset[idx],\
                                                                    migration_begin_offset[idx] + migration_time[idx],True)
                        latency_list.append(avg_decode_latency_migration)
                        idx+=1
    return latency_list

def plot_migration_microbenchmark(log_path):
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(2*4.2, 3.4))
    plt.subplots_adjust(bottom=0.15)

    plt.rcParams.update({'font.size': 10})
    len_list = [256,512,1024,2048,4096,8192]
    seq_len7b_x = np.log2(len_list)
    seq_len30b_x = np.log2(len_list)
    # profiled recompute latency
    log_path_7b = os.path.join(log_path,"7b")
    log_path_30b = os.path.join(log_path,"30b")
    # recompute_7b = [58,110,220,436,940,1995]
    recompute_7b =get_recompute_data(os.path.join(log_path_7b, "Recompute"),len_list)
    recompute_30b =get_recompute_data(os.path.join(log_path_30b, "Recompute"),len_list)
    # recompute_30b = [115,235,443,829,1712,3546]

    # stage1_7b = [13,12,15,16,22,26]
    # stage1_30b = [33,37,32,33,37,35]

    # blocking_7b = [68,129,226,451,909,1682]
    # blocking_30b = [118,181,272,604,1203,1730]

    # migration_decode_7b = [47.36,44.56,43.99,43.48,43.59,44.18]
    # migration_decode_30b = [66.11,62.79,59.70,60.03,61.82,66.66]

    # decode_7b = [46.80, 44.39, 43.30, 42.85, 42.90, 43.62]
    # decode_30b = [65.74,61.76,59.15,58.91,60.64,65.76]
    stage1_7b = []
    stage1_30b = []

    blocking_7b = []
    blocking_30b = []

    migration_decode_7b = []
    migration_decode_30b = []

    decode_7b = []
    decode_30b = []

    blocking_7b, stage1_7b, migration_decode_7b, migration_begin_offset = get_migration_stage_data(os.path.join(log_path_7b, "Migration"),len_list)
    decode_7b = get_normal_decode_latency(os.path.join(log_path_7b, "Normal"),len_list,migration_begin_offset, blocking_7b)
    blocking_30b, stage1_30b, migration_decode_30b, migration_begin_offset = get_migration_stage_data(os.path.join(log_path_30b, "Migration"),len_list)
    decode_30b = get_normal_decode_latency(os.path.join(log_path_30b, "Normal"),len_list,migration_begin_offset, blocking_30b)
    # print(max((np.array(migration_decode_7b)-np.array(decode_7b))/np.array(decode_7b)))

    ax1.plot(seq_len7b_x, np.array(stage1_7b)*1000, label=f"Migration(7B)",color="tab:blue",marker="o")
    ax1.plot(seq_len30b_x, np.array(stage1_30b)*1000, label=f"Migration(30B)",color="tab:blue",marker="o",linestyle='--')

    ax1.plot(seq_len7b_x, np.array(blocking_7b)*1000, label=f"Blocking copy(7B)",color="tab:orange",marker="v")
    ax1.plot(seq_len30b_x, np.array(blocking_30b)*1000, label=f"Blocking copy(30B)",color="tab:orange",marker="v",linestyle='--')

    ax1.plot(seq_len7b_x, np.array(recompute_7b)*1000, label=f"Recompute(7B)",color="tab:green",marker="s")
    ax1.plot(seq_len30b_x, np.array(recompute_30b)*1000, label=f"Recompute(30B)",color="tab:green",marker="s",linestyle='--')

    ax2.plot(seq_len7b_x,migration_decode_7b, label=f"Migration(7B)",color="tab:blue",marker="o")
    ax2.plot(seq_len30b_x,migration_decode_30b, label=f"Migration(30B)",color="tab:blue",marker="o",linestyle='--')

    ax2.plot(seq_len7b_x,decode_7b, label=f"Normal(7B)",color="tab:orange",marker="v")
    ax2.plot(seq_len30b_x,decode_30b, label=f"Normal(30B)",color="tab:orange",marker="v",linestyle='--')

    decode_7b = np.array(decode_7b)
    migration_decode_7b = np.array(migration_decode_7b)
    decode_30b = np.array(decode_30b)
    migration_decode_30b = np.array(migration_decode_30b)

    out_filename = "figure10.claim"
    with open(out_filename, 'w+') as out_file:
        out_file.write("Llumnix overhead\n")
        out_file.write("7B model overhead up to {:.2f}\n".format(max(migration_decode_7b/decode_7b - 1)))
        out_file.write("30B model overhead up to {:.2f}\n".format(max(migration_decode_30b/decode_30b - 1)))

    def thousands_formatter(x, pos):
        if 2**x >=1000:
            return '%dk' % (2**x / 1000)
        else:
            return int(2 ** x)

    lable_fontsize = 12
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax1.legend(loc='best')
    ax1.set_xlabel('Sequence Length',fontsize=lable_fontsize)
    ax1.set_ylabel('Downtime (ms)',fontsize=lable_fontsize)

    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax2.legend(loc='best')
    ax2.set_xlabel('Sequence Length',fontsize=lable_fontsize)
    ax2.set_ylabel('Decode Latency (ms)',fontsize=lable_fontsize)

    fig_filename = "figure10.png"
    fig.savefig(fig_filename)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', default='../log', type=str)
    args = parser.parse_args()
    plot_migration_microbenchmark(args.log_path)