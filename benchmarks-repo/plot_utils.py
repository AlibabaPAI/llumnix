import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import csv
from xformers import ops as xops
import numpy as np
import time
from scipy.interpolate import interp1d
import argparse
from process import read_data_from_log_file
from collections import defaultdict
from vllm.simulator.profiling import *
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter


def _pad_to_alignment(x, multiple_of):
    return x + ((-1*x) % multiple_of)

def save_inference_csv(results_filename, model_name, tp, pp, inference_type, bs_tuple_list, latency_list):
    inference_info_file = open(results_filename+f'_{inference_type}_inference.csv','w')
    inference_info_csv = csv.writer(inference_info_file)
    inference_info_csv.writerow([
        'ModelName',
        "TP",
        "PP",
        "inference_type",
        "BS",
        "len_sum",
        "StageLatencies(ms)",
        ])
    for idx, (bs, len_sum) in enumerate(bs_tuple_list):
        inference_info_csv.writerow([
            model_name,
            tp,
            pp,# now pp=1
            inference_type,
            bs,
            len_sum,
            [latency_list[idx]],
        ])
    inference_info_file.flush()

def plot_inference_latency(results_filename, inference_type="prefill"):
    df = pd.read_csv(results_filename+"_instance.csv")
    df = df[df['inference_type']==inference_type].drop_duplicates()
    x = df['bs'].to_numpy()
    x = _pad_to_alignment(x, 8)
    ts = df['timestamp'].to_numpy()
    y = df['latency'].to_numpy()
    seq_lens_list = []
    for idx, row in df.iterrows():
        seq_lens = list(map(float, row["seq_lens"].strip("[]").split(",")))
        seq_lens_list.append(seq_lens)
    latency_dict = {}
    max_seq_len_dict = {}
    for i in range(len(x)):
        if x[i] not in latency_dict:
            latency_dict[x[i]] = [y[i]]
            max_seq_len_dict[x[i]] = [np.sum(seq_lens_list[i])]
        else:
            latency_dict[x[i]].append(y[i])
            max_seq_len_dict[x[i]].append(np.sum(seq_lens_list[i]))
    df = df[df['latency'] > 100]
    # print(df.to_csv("outlier.csv"))
    bs_list = []
    new_x = []
    new_y = []
    if inference_type == 'prefill':
        new_x = np.arange(start=8, stop=16384, step=8)
        coeff = np.polyfit(x, y, 1)
        for bs in new_x:
            new_y.append(coeff[0] * bs + coeff[1])
        for bs in latency_dict.keys():
            if np.max(latency_dict[bs]) - (coeff[0] * bs + coeff[1]) > 50:
                print(np.max(latency_dict[bs]))
            # print(f'bs={key}, max_latency={np.max(latency_dict[key])}, min_latency={np.min(latency_dict[key])}, average={np.average(latency_dict[key])}')
        fig, ax = plt.subplots()
        ax.scatter(x, y, marker='o')
        ax.plot(new_x, new_y, color='red')
        bs_list = [1]*len(new_x)
        ax.set_xlabel('batch size')
        ax.set_ylabel('latency(ms)')
    else:
        fig, ax = plt.subplots()
        for bs in latency_dict.keys():
            ax.scatter(max_seq_len_dict[bs], latency_dict[bs], label=f"bs={bs}")
            coeff = np.polyfit(max_seq_len_dict[bs], latency_dict[bs], 1)
            for seq_len in range(16384):
                new_y.append(coeff[0] * seq_len + coeff[1])
            new_x.extend(range(16384))
            bs_list.extend([bs]*16384)
            ax.plot(range(16384), new_y[-16384:], color='red')
        # ax.scatter(ts, y)
        ax.legend(loc='upper left')
        ax.set_xlabel('sum(seq_len)')
        ax.set_ylabel('latency(ms)')

    fig_filename = os.path.splitext(results_filename)[0] + f"_{inference_type}.png"
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)
    save_inference_csv(results_filename, 'llama-7b',1,1,inference_type, list(zip(bs_list, new_x)), new_y)

def find_ins_diff(results_filename_0, results_filename_1):
    df_0 = pd.read_csv(results_filename_0+"_instance.csv")
    df_1 = pd.read_csv(results_filename_1+"_instance.csv")
    df_0['timestamp'] -= np.min(df_0['timestamp'].to_numpy())
    df_1['timestamp'] -= np.min(df_1['timestamp'].to_numpy())
    df_0_ins0 = df_0[df_0["instance_id"]==0]
    df_1_ins0 = df_1[df_1["instance_id"]==0]
    for i in range(400):
        info_0 = df_0_ins0[df_0_ins0["step_id"]==i]
        info_1 = df_1_ins0[df_1_ins0["step_id"]==i]
        if info_0['max_tot_tokens'].tolist()[0] != info_1['max_tot_tokens'].tolist()[0]:
            print(f'step_{i}')
            print(info_0.drop_duplicates())
            print(info_1.drop_duplicates())
            break

def plot_mem_v0(results_filename):
    df = pd.read_csv(results_filename+"_instance.csv")
    instance_0 = df[df["instance_id"] == 0]
    instance_1 = df[df["instance_id"] == 1]
    instance_0_timestamp = instance_0["timestamp"].to_numpy()
    instance_0_load = instance_0["gpu_cache_usage"].to_numpy()
    instance_1_timestamp = instance_1["timestamp"].to_numpy()
    instance_1_load = instance_1["gpu_cache_usage"].to_numpy()
    for i in range(1,len(instance_0_timestamp)):
        if instance_0_timestamp[i]-instance_0_timestamp[i-1] > 10:
            import datetime
            print(instance_0_timestamp[i-1])
            print(datetime.datetime.fromtimestamp(instance_0_timestamp[i-1]))
    time_begin = min(instance_0_timestamp[0],instance_1_timestamp[0])
    fig, ax = plt.subplots()
    ax.plot(instance_0_timestamp - time_begin, instance_0_load*100, color="red", label="instance_0")
    ax.plot(instance_1_timestamp - time_begin, instance_1_load*100, color="blue", label="instance_1")

    ax.legend(loc='upper left')
    ax.set_xlabel('timestamp(s)')
    fig_filename = os.path.splitext(results_filename)[0] + "_mem.png"
    ax.set_ylabel('gpu usage(%)')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def plot_mem_v1(results_filename, show_migrate=True, show_killed=False):
    df = pd.read_csv(results_filename+"_instance.csv")
    instance_0 = df[df["instance_id"] == 0]
    instance_1 = df[df["instance_id"] == 1]
    instance_0_timestamp = instance_0["timestamp"].to_numpy()
    instance_0_gpu = instance_0["gpu_cache_usage"].to_numpy()
    instance_1_timestamp = instance_1["timestamp"].to_numpy()
    instance_1_gpu = instance_1["gpu_cache_usage"].to_numpy()
    time_begin = min(instance_0_timestamp[0],instance_1_timestamp[0])
    fig, ax = plt.subplots()
    instance_0_timestamp -= time_begin
    instance_0_timestamp = np.round(instance_0_timestamp,2)
    instance_1_timestamp = np.round(instance_1_timestamp,2)
    x_0 = np.linspace(min(instance_0_timestamp), max(instance_0_timestamp), 10000)
    f_0 = interp1d(instance_0_timestamp, instance_0_gpu*100)
    instance_1_timestamp -= time_begin
    x_1 = np.linspace(min(instance_1_timestamp), max(instance_1_timestamp), 10000)
    f_1 = interp1d(instance_1_timestamp, instance_1_gpu*100)
    ax.plot(x_0, f_0(x_0), color="red", label="instance_0")
    ax.plot(x_1, f_1(x_1), color="blue", label="instance_1")
    # ax.axhline(np.mean(instance_0_gpu)*100, linestyle='--', color='red')
    # ax.axhline(np.mean(instance_1_gpu)*100, linestyle='--', color='blue')
    # print(np.mean(instance_0_gpu)*100,np.mean(instance_1_gpu)*100)
    ax.legend(loc='upper left')
    ax.set_xlabel('timestamp(s)')
    fig_filename = os.path.splitext(results_filename)[0] + "_mem.png"
    ax.set_ylabel('gpu usage(%)')
    if show_migrate==True:
        df = pd.read_csv(results_filename+"_req.csv")
        migrate_in_0 = df[(df['event']=="migrate_in") & (df['instance_id']==0)]
        migrate_in_1 = df[(df['event']=="migrate_in") & (df['instance_id']==1)]
        timestamp_0 = migrate_in_0['timestamp'].to_numpy() - time_begin
        timestamp_1 = migrate_in_1['timestamp'].to_numpy() - time_begin
        timestamp_0 = np.round(timestamp_0,2)
        timestamp_1 = np.round(timestamp_1,2)
        ax.scatter(timestamp_0, f_0(timestamp_0), s=16, label='migrate in')
        ax.scatter(timestamp_1, f_1(timestamp_1), s=16, label='migrate in')
        ax.legend(loc='upper left')
    if show_killed==True:
        df = pd.read_csv(results_filename+"_req.csv")
        migrate_in_0 = df[(df['event']=="killed") & (df['instance_id']==0)]
        migrate_in_1 = df[(df['event']=="killed") & (df['instance_id']==1)]
        timestamp_0 = migrate_in_0['timestamp'].to_numpy() - time_begin
        timestamp_1 = migrate_in_1['timestamp'].to_numpy() - time_begin
        timestamp_0 = np.round(timestamp_0,2)
        timestamp_1 = np.round(timestamp_1,2)
        ax.scatter(timestamp_0, f_0(timestamp_0), s=16, label='killed')
        ax.scatter(timestamp_1, f_1(timestamp_1), s=16, label='killed')
        ax.legend(loc='upper left')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def plot_mem_v2(results_filename, instance_num=2):
    df = pd.read_csv(results_filename+"_instance.csv").drop_duplicates()
    tot_steps = 0
    time_begin = np.inf
    time_end = 0
    instance_timestamp_list = []
    instance_gpu_list = []
    instance_steps_list = []
    for instance_id in range(instance_num):
        instance = df[df["instance_id"] == instance_id]
        instance_steps = len(instance)
        tot_steps = max(tot_steps, instance_steps)
        instance_steps_list.append(instance_steps)
        instance_timestamp = instance["timestamp"].to_numpy()
        instance_gpu = instance["gpu_cache_usage"].to_numpy()
        instance_timestamp_list.append(instance_timestamp)
        instance_gpu_list.append(instance_gpu)
        if len(instance_timestamp):
            time_begin = min(time_begin, instance_timestamp[0])
            time_end = max(time_end,instance_timestamp[-1])
    
    fig, ax = plt.subplots()
    tot_avg_gpu_usage = 0
    for i in range(instance_num):
        instance_timestamp = instance_timestamp_list[i]
        instance_gpu = instance_gpu_list[i]
        instance_timestamp -= time_begin
        instance_timestamp = np.round(instance_timestamp, 2)
        step_precent = np.round(instance_steps_list[i]/tot_steps*100, 2)
        area = np.trapz(instance_gpu, instance_timestamp)
        avg_gpu_usage = area/(time_end - time_begin)*100
        tot_avg_gpu_usage += avg_gpu_usage
        ax.plot(instance_timestamp, instance_gpu, label=f"instance_{i}:{np.round(avg_gpu_usage,2)}%", linewidth=0.5)
    tot_avg_gpu_usage/=instance_num
    print(f"tot avg gpu usage: {np.round(tot_avg_gpu_usage,2)}%")

    ax.legend(loc='upper left')
    ax.set_xlabel('timestamp(s)')
    ax.set_ylabel('gpu usage(%)')

    df = pd.read_csv(results_filename+"_req.csv")
    migrate_in_df = df[(df['event']=="migrate_in")]
    migrate_ts = migrate_in_df['timestamp'].to_numpy() - time_begin
    ax.scatter(migrate_ts, np.zeros_like(migrate_ts), c='green')

    fig_filename = os.path.splitext(results_filename)[0] + "_mem.png"
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def plot_load(results_filename):
    df = pd.read_csv(results_filename+"_instance.csv")
    instance_0 = df[df["instance_id"] == 0]
    instance_1 = df[df["instance_id"] == 1]
    instance_0_timestamp = instance_0["timestamp"].to_numpy()
    instance_0_load = instance_0["instance_load"].to_numpy()
    instance_1_timestamp = instance_1["timestamp"].to_numpy()
    instance_1_load = instance_1["instance_load"].to_numpy()
    time_begin = min(instance_0_timestamp[0],instance_1_timestamp[0])
    fig, ax = plt.subplots()
    ax.plot(instance_0_timestamp - time_begin, np.clip(instance_0_load,a_min=0, a_max=10), color="red", label="instance_0")
    ax.plot(instance_1_timestamp - time_begin, np.clip(instance_1_load,a_min=0, a_max=10), color="blue", label="instance_1")

    ax.legend(loc='upper left')
    ax.set_xlabel('timestamp(s)')
    fig_filename = os.path.splitext(results_filename)[0] + "_load.png"
    ax.set_ylabel('load')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def plot_max_tokens(results_filename):
    df = pd.read_csv(results_filename+"_instance.csv")
    instance_0 = df[df["instance_id"] == 0]
    instance_1 = df[df["instance_id"] == 1]
    instance_0_timestamp = instance_0["timestamp"].to_numpy()
    instance_0_tokens = instance_0["max_tot_tokens"].to_numpy()
    instance_1_timestamp = instance_1["timestamp"].to_numpy()
    instance_1_tokens = instance_1["max_tot_tokens"].to_numpy()
    time_begin = min(instance_0_timestamp[0],instance_1_timestamp[0])
    fig, ax = plt.subplots()

    ax.plot(instance_0_timestamp - time_begin, instance_0_tokens, color="red", label="instance_0")
    ax.plot(instance_1_timestamp - time_begin, instance_1_tokens, color="blue", label="instance_1")
    ax.axhline(13000, linestyle='--')
    ax.legend(loc='upper left')
    ax.set_xlabel('timestamp(s)')
    fig_filename = os.path.splitext(results_filename)[0] + "_maxtoken.png"
    ax.set_ylabel('max_tot_tokens')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def get_migrate_killed(results_filename):
    df = pd.read_csv(results_filename+"_req.csv")
    req_ids = df[df['event']=="migrate_in"]["req_id"].drop_duplicates()
    print(f'migrate reqs:{len(req_ids)}')
    cnt = 0
    for req_id in req_ids:
        events = df[df["req_id"] == req_id]["event"].tolist()
        print(events)
        for event in reversed(events):
            if event == "migrate_in":
                break
            if event == "killed":
                cnt+=1
                break
    print(f'killed after migrate {cnt}')
    
def check_output():
    gt = {}
    migrated = {}
    with open("gt.jso", 'r') as f:
        gt = json.load(f)
    with open("migrate_async.json", 'r') as f:
        migrated = json.load(f)

    for key in gt.keys():
        if gt[key][:50] != migrated[key][:50]:
            print(key)
            print(migrated[key][:100])
            print("*"*50)
            print(gt[key][:100])
            print("-"*50)
    
def plot_scaling_p99latency_compare(latency_enable_migrate, latency_disable_migrate, x_label="qps"):
    enable_list = None
    disable_list = None
    with open(latency_enable_migrate,'r') as f:
        enable_list = json.load(f)
    with open(latency_disable_migrate,'r') as f:
        disable_list = json.load(f)
    fig, (ax_thr, ax_prefill, ax_decode, ax_instance) = plt.subplots(1,4, figsize=(4*7, 4.8))
    x = []
    enable_list = sorted(enable_list, key=lambda x: x[x_label])
    disable_list = sorted(disable_list, key=lambda x: x[x_label])
    throughput_list_enable = []
    throughput_list_disable = []
    prefill_p99_list_enable = []
    prefill_p99_list_disable = []
    decode_p99_list_enable = []
    decode_p99_list_disable = []
    instance_list_enable = []
    instance_list_disable = []
    for i in range(len(enable_list)):
        x.append(enable_list[i][x_label])
        throughput_enable =  np.percentile(enable_list[i]["throughput"], 99)
        throughput_list_enable.append(throughput_enable)
        throughput_disable =  np.percentile(disable_list[i]["throughput"], 99)
        throughput_list_disable.append(throughput_disable)
        prefill_p99_enable =  np.percentile(enable_list[i]["prefill_latencies"], 99)
        prefill_p99_list_enable.append(prefill_p99_enable)
        prefill_p99_disable =  np.percentile(disable_list[i]["prefill_latencies"], 99)
        prefill_p99_list_disable.append(prefill_p99_disable)
        decode_p99_enable =  np.percentile(enable_list[i]["decode_latencies"], 99)
        decode_p99_list_enable.append(decode_p99_enable)
        decode_p99_disable =  np.percentile(disable_list[i]["decode_latencies"], 99)
        decode_p99_list_disable.append(decode_p99_disable)
        instance_enable =  enable_list[i]["instance_num"]
        instance_list_enable.append(instance_enable)
        instance_disable =  disable_list[i]["instance_num"]
        instance_list_disable.append(instance_disable)
    ax_thr.plot(x, throughput_list_enable, 'o-', color='r', label="enable migrate")
    ax_thr.plot(x, throughput_list_disable, 's-', color='b', label="disable migrate")
    ax_prefill.plot(x, prefill_p99_list_enable, 'o-', color='r', label="enable migrate")
    ax_prefill.plot(x, prefill_p99_list_disable, 's-', color='b', label="disable migrate")
    ax_decode.plot(x, decode_p99_list_enable, 'o-', color='r', label="enable migrate")
    ax_decode.plot(x, decode_p99_list_disable, 's-', color='b', label="disable migrate")
    ax_instance.plot(x, instance_list_enable, 'o-', color='r', label="enable migrate")
    ax_instance.plot(x, instance_list_disable, 's-', color='b', label="disable migrate")
    ax_thr.set_xlabel(x_label)
    ax_thr.set_ylabel('throughput(token/s)')
    ax_prefill.set_xlabel(x_label)
    ax_prefill.set_ylabel('prefill p99 lantency(ms)')
    ax_decode.set_xlabel(x_label)
    ax_decode.set_ylabel('decode p99 lantency(ms)')
    ax_instance.set_xlabel(x_label)
    ax_instance.set_ylabel('avg instance num(gpu/s)')
    ax_thr.legend(loc='upper left')
    ax_decode.legend(loc='upper left')
    ax_instance.legend(loc='upper left')
    ax_prefill.legend(loc='upper left')
    ax_decode.set_ylim(ymin=0)
    ax_instance.set_ylim(ymin=1)
    ax_prefill.set_ylim(ymin=0)
    fig_filename = os.path.splitext(latency_enable_migrate)[0] + "_scaling_p99_compare.png"
    fig.savefig(fig_filename)

def plot_scaling_meanlatency_compare(latency_enable_migrate, latency_disable_migrate, x_label="qps"):
    enable_list = None
    disable_list = None
    with open(latency_enable_migrate,'r') as f:
        enable_list = json.load(f)
    with open(latency_disable_migrate,'r') as f:
        disable_list = json.load(f)
    fig, (ax_thr, ax_prefill, ax_decode, ax_instance) = plt.subplots(1,4, figsize=(4*7, 4.8))
    x = []
    if x_label!="scale_up_threshold":
        enable_list = sorted(enable_list, key=lambda x: x[x_label])
        disable_list = sorted(disable_list, key=lambda x: x[x_label])
    throughput_list_enable = []
    throughput_list_disable = []
    prefill_mean_list_enable = []
    prefill_mean_list_disable = []
    decode_mean_list_enable = []
    decode_mean_list_disable = []
    instance_list_enable = []
    instance_list_disable = []
    for i in range(len(enable_list)):
        if x_label=="scale_up_threshold":
            x.append(5+20*i)
        else:
            x.append(enable_list[i][x_label])
        throughput_enable =  np.mean(enable_list[i]["throughput"])
        throughput_list_enable.append(throughput_enable)
        throughput_disable =  np.mean(disable_list[i]["throughput"])
        throughput_list_disable.append(throughput_disable)
        prefill_mean_enable =  np.mean(enable_list[i]["prefill_latencies"])
        prefill_mean_list_enable.append(prefill_mean_enable)
        prefill_mean_disable =  np.mean(disable_list[i]["prefill_latencies"])
        prefill_mean_list_disable.append(prefill_mean_disable)
        decode_mean_enable =  np.mean(enable_list[i]["decode_latencies"])
        decode_mean_list_enable.append(decode_mean_enable)
        decode_mean_disable =  np.mean(disable_list[i]["decode_latencies"])
        decode_mean_list_disable.append(decode_mean_disable)
        instance_enable =  enable_list[i]["instance_num"]
        instance_list_enable.append(instance_enable)
        instance_disable =  disable_list[i]["instance_num"]
        instance_list_disable.append(instance_disable)
    ax_thr.plot(x, throughput_list_enable, 'o-', color='r', label="enable migrate")
    ax_thr.plot(x, throughput_list_disable, 's-', color='b', label="disable migrate")
    ax_prefill.plot(x, prefill_mean_list_enable, 'o-', color='r', label="enable migrate")
    ax_prefill.plot(x, prefill_mean_list_disable, 's-', color='b', label="disable migrate")
    ax_decode.plot(x, decode_mean_list_enable, 'o-', color='r', label="enable migrate")
    ax_decode.plot(x, decode_mean_list_disable, 's-', color='b', label="disable migrate")
    ax_instance.plot(x, instance_list_enable, 'o-', color='r', label="enable migrate")
    ax_instance.plot(x, instance_list_disable, 's-', color='b', label="disable migrate")
    ax_thr.set_xlabel(x_label)
    ax_thr.set_ylabel('throughput(token/s)')
    ax_prefill.set_xlabel(x_label)
    ax_prefill.set_ylabel('prefill mean lantency(ms)')
    ax_decode.set_xlabel(x_label)
    ax_decode.set_ylabel('decode mean lantency(ms)')
    ax_instance.set_xlabel(x_label)
    ax_instance.set_ylabel('avg instance num(gpu/s)')
    ax_thr.legend(loc='upper left')
    ax_decode.legend(loc='upper left')
    ax_instance.legend(loc='upper left')
    ax_prefill.legend(loc='upper left')
    ax_decode.set_ylim(ymin=0)
    ax_instance.set_ylim(ymin=1)
    ax_prefill.set_ylim(ymin=0)
    fig_filename = os.path.splitext(latency_enable_migrate)[0] + "_scaling_mean_compare.png"
    fig.savefig(fig_filename)

def plot_interval_compare(enable_migrate_npy_file, disable_migrate_npy_file):
    fig_filename = os.path.splitext(enable_migrate_npy_file)[0] + "_interval.png"
    fig, (ax_lat, ax_throughput) = plt.subplots(1, 2, figsize=(2*7, 4.8))

    def plot_all_latencies(ax_lat, ax_throughput, is_migrate, all_lat_pairs,  interval = 100):
        prev_idx = 0
        ts = [0]
        interval_lat = [all_lat_pairs[0][1]]
        all_p99_interval_lat = []
        all_interval_throughput = []
        for idx in range(1, len(all_lat_pairs)):
            ts.append(all_lat_pairs[idx][0]-all_lat_pairs[0][0])
            interval_lat.append(all_lat_pairs[idx][1])
            if all_lat_pairs[idx][0]-all_lat_pairs[prev_idx][0] > interval or idx+1 == len(all_lat_pairs):
                throughput = len(interval_lat)/(all_lat_pairs[idx][0]-all_lat_pairs[prev_idx][0])
                p99 = np.percentile(interval_lat, 99)
                for _ in range(prev_idx, idx+1):
                    all_p99_interval_lat.append(p99)
                    all_interval_throughput.append(throughput)
                interval_lat = []
                prev_idx = idx + 1
        ax_lat.plot(ts, all_p99_interval_lat, color="red" if is_migrate else 'blue', label='enable migrate' if is_migrate else 'disable migrate')
        ax_throughput.plot(ts, all_interval_throughput, color="red" if is_migrate else 'blue', label='enable migrate' if is_migrate else 'disable migrate')
    
    all_lat_pairs = np.load(enable_migrate_npy_file)
    plot_all_latencies(ax_lat, ax_throughput,True, all_lat_pairs)

    all_lat_pairs = np.load(disable_migrate_npy_file)
    plot_all_latencies(ax_lat, ax_throughput,False, all_lat_pairs)

    ax_throughput.legend(loc='upper left')
    ax_throughput.set_xlabel('timestamp (s)')
    ax_throughput.set_ylabel('token/s')
    ax_lat.legend(loc='upper left')
    ax_lat.set_xlabel('timestamp (s)')
    ax_lat.set_ylabel('p99 lat(ms)')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def plot_len_latency_diff(results_filename1: str,
                          results_filename2: str, 
                          len_key: str = 'prompt'):
    all_e2e_latencies1, all_inference_latencies1, all_waiting_latencies1, \
        all_total_tokens1, all_prompt_lens1, all_response_lens1 = read_data_from_log_file(results_filename1)
    all_e2e_latencies2, all_inference_latencies2, all_waiting_latencies2, \
        all_total_tokens2, all_prompt_lens2, all_response_lens2 = read_data_from_log_file(results_filename2)

    assert all_total_tokens1 == all_total_tokens2 and all_prompt_lens1 == all_prompt_lens2 and all_response_lens1 == all_response_lens2

    fig, (ax_waiting, ax_inference, ax_e2e) = plt.subplots(1, 3, figsize=(3*7, 4.8))

    def plot_single(ax, len_key, latency_key):
        latencies_diff = []
        lens = []
        if latency_key == 'e2e':
            latencies_diff = [all_e2e_latencies2[i] - all_e2e_latencies1[i] for i in range(len(all_e2e_latencies2))]
            y_label_str = 'e2e lantency diff'
        elif latency_key == 'inference':
            latencies_diff = [all_inference_latencies2[i] - all_inference_latencies1[i] for i in range(len(all_inference_latencies2))]
            y_label_str = 'inference lantency diff'
        elif latency_key == 'waiting':
            latencies_diff = [all_waiting_latencies2[i] - all_waiting_latencies1[i] for i in range(len(all_waiting_latencies2))]
            y_label_str = 'waiting lantency diff'

        if len_key == 'total':
            lens = all_total_tokens1
            x_label_str = 'total token'
        elif len_key == 'prompt':
            lens = all_prompt_lens1
            x_label_str = 'prompt len'
        elif len_key == 'response':
            lens = all_response_lens1
            x_label_str = 'response len'

        mean_latency_diff = np.mean(latencies_diff)

        lens_latencies_diff_dict = defaultdict(list)
        for i in range(len(lens)):
            lens_latencies_diff_dict[lens[i]].append(latencies_diff[i])
        lens = []
        latencies_diff = []
        for (l, diffs) in lens_latencies_diff_dict.items():
            lens.append(l)
            latencies_diff.append(np.mean(diffs))
    
        ax.scatter(lens, latencies_diff, s=8)
        ax.axhline(y=mean_latency_diff, color='red', linestyle='--')
        ax.text(ax.get_xlim()[1], mean_latency_diff, f"{mean_latency_diff:.2f}", va='bottom', ha='right', color='blue')
        ax.grid(True)
        ax.set_xlabel(x_label_str)
        ax.set_ylabel(y_label_str)
    
    plot_single(ax_waiting, len_key, latency_key='waiting')
    plot_single(ax_inference, len_key, latency_key='inference')
    plot_single(ax_e2e, len_key, latency_key='e2e')

    fig_filename1 = os.path.splitext(results_filename1)[0]
    index1 = fig_filename1.rfind('/')
    index2 = fig_filename1.rfind('/', 0, index1)
    fig_filename1_title = fig_filename1[index2 + 1:]
    fig_filename2 = os.path.splitext(results_filename2)[0] + "_" + len_key + "_latency_diff.png"
    index1 = fig_filename2.rfind('/')
    index2 = fig_filename2.rfind('/', 0, index1)
    fig_filename2_title = fig_filename2[index2 + 1:]
    fig_filename_title = fig_filename1_title + '_' + fig_filename2_title
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename2)

def plot_len_latency(results_filename1: str,
                     len_key: str = 'prompt'):
    all_e2e_latencies1, all_inference_latencies1, all_waiting_latencies1, \
        all_total_tokens1, all_prompt_lens1, all_response_lens1 = read_data_from_log_file(results_filename1)

    fig, (ax_waiting, ax_inference, ax_e2e) = plt.subplots(1, 3, figsize=(3*7, 4.8))

    def plot_single(ax, len_key, latency_key):
        latencies = []
        lens = []
        if latency_key == 'e2e':
            latencies = [all_e2e_latencies1[i] for i in range(len(all_e2e_latencies1))]
            y_label_str = 'e2e lantency'
        elif latency_key == 'inference':
            latencies = [all_inference_latencies1[i] for i in range(len(all_inference_latencies1))]
            y_label_str = 'inference lantency'
        elif latency_key == 'waiting':
            latencies = [all_waiting_latencies1[i] for i in range(len(all_waiting_latencies1))]
            y_label_str = 'waiting lantency'

        if len_key == 'total':
            lens = all_total_tokens1
            x_label_str = 'total token'
        elif len_key == 'prompt':
            lens = all_prompt_lens1
            x_label_str = 'prompt len'
        elif len_key == 'response':
            lens = all_response_lens1
            x_label_str = 'response len'

        mean_latency = np.mean(latencies)
    
        ax.scatter(lens, latencies, s=8)
        ax.axhline(y=mean_latency, color='red', linestyle='--')
        ax.text(ax.get_xlim()[1], mean_latency, f"{mean_latency:.2f}", va='bottom', ha='right', color='blue')
        ax.grid(True)
        ax.set_xlabel(x_label_str)
        ax.set_ylabel(y_label_str)
    
    plot_single(ax_waiting, len_key, latency_key='waiting')
    plot_single(ax_inference, len_key, latency_key='inference')
    plot_single(ax_e2e, len_key, latency_key='e2e')

    fig_filename1 = os.path.splitext(results_filename1)[0] + "_" + len_key + "_latency.png"
    index1 = fig_filename1.rfind('/')
    index2 = fig_filename1.rfind('/', 0, index1)
    fig_filename1_title = fig_filename1[index2 + 1:]
    plt.suptitle(fig_filename1_title, fontsize=6)
    fig.savefig(fig_filename1)

def plot_len_latency_diff_v2(results_filename1: str,
                             results_filename2: str, 
                             len_key: str = 'prompt'):
    all_e2e_latencies1, all_inference_latencies1, all_waiting_latencies1, \
        all_total_tokens1, all_prompt_lens1, all_response_lens1 = read_data_from_log_file(results_filename1)
    all_e2e_latencies2, all_inference_latencies2, all_waiting_latencies2, \
        all_total_tokens2, all_prompt_lens2, all_response_lens2 = read_data_from_log_file(results_filename2)

    assert all_total_tokens1 == all_total_tokens2 and all_prompt_lens1 == all_prompt_lens2 and all_response_lens1 == all_response_lens2

    fig, (ax_waiting, ax_inference, ax_e2e) = plt.subplots(1, 3, figsize=(3*7, 4.8))

    def plot_single(ax, len_key, latency_key):
        latencies_diff = []
        lens = []
        if latency_key == 'e2e':
            latencies_diff = [all_e2e_latencies2[i] - all_e2e_latencies1[i] for i in range(len(all_e2e_latencies2))]
            y_label_str = 'e2e lantency diff'
        elif latency_key == 'inference':
            latencies_diff = [all_inference_latencies2[i] - all_inference_latencies1[i] for i in range(len(all_inference_latencies2))]
            y_label_str = 'inference lantency diff'
        elif latency_key == 'waiting':
            latencies_diff = [all_waiting_latencies2[i] - all_waiting_latencies1[i] for i in range(len(all_waiting_latencies2))]
            y_label_str = 'waiting lantency diff'

        if len_key == 'total':
            lens = all_total_tokens1
            x_label_str = 'total token'
        elif len_key == 'prompt':
            lens = all_prompt_lens1
            x_label_str = 'prompt len'
        elif len_key == 'response':
            lens = all_response_lens1
            x_label_str = 'response len'

        mean_latency_diff = np.mean(latencies_diff)
    
        ax.scatter(lens, latencies_diff, s=8)
        ax.axhline(y=mean_latency_diff, color='red', linestyle='--')
        ax.text(ax.get_xlim()[1], mean_latency_diff, f"{mean_latency_diff:.2f}", va='bottom', ha='right', color='blue')
        ax.grid(True)
        ax.set_xlabel(x_label_str)
        ax.set_ylabel(y_label_str)
    
    plot_single(ax_waiting, len_key, latency_key='waiting')
    plot_single(ax_inference, len_key, latency_key='inference')
    plot_single(ax_e2e, len_key, latency_key='e2e')

    fig_filename1 = os.path.splitext(results_filename1)[0]
    index1 = fig_filename1.rfind('/')
    index2 = fig_filename1.rfind('/', 0, index1)
    fig_filename1_title = fig_filename1[index2 + 1:]
    fig_filename2 = os.path.splitext(results_filename2)[0] + "_" + len_key + "_latency_diff.png"
    index1 = fig_filename2.rfind('/')
    index2 = fig_filename2.rfind('/', 0, index1)
    fig_filename2_title = fig_filename2[index2 + 1:]
    fig_filename_title = fig_filename1_title + '_' + fig_filename2_title
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename2)

def plot_waiting(results_filename, instance_num=2):
    df = pd.read_csv(results_filename + "_instance.csv").drop_duplicates()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(4*7, 4.8))

    def plot_single(ax, key):
        if key == 'num_block_first_waiting_request':
            y_column_str = 'num_block_first_waiting_request'
            y_label_str = 'num block first waiting request'
        elif key == 'num_block_all_waiting_request':
            y_column_str = 'num_block_all_waiting_request'
            y_label_str = 'num block all waiting request'
        elif key == 'waiting_time_first_waiting_request':
            y_column_str = 'waiting_time_first_waiting_request'
            y_label_str = 'waiting time first waiting request'
        elif key == 'num_waiting_request':
            y_column_str = 'num_waiting_request'
            y_label_str = 'num waiting request'

        time_begin = np.inf
        time_end = 0
        instance_timestamp_list = []
        instance_y_list = []
        for instance_id in range(instance_num):
            instance = df[df["instance_id"] == instance_id]
            instance_timestamp = instance["timestamp"].to_numpy()
            instance_y = instance[y_column_str].to_numpy()
            instance_timestamp_list.append(instance_timestamp)
            instance_y_list.append(instance_y)
            
            if len(instance_timestamp):
                time_begin = min(time_begin, instance_timestamp[0])
                time_end = max(time_end, instance_timestamp[-1])
        
        for i in range(instance_num):
            instance_timestamp = instance_timestamp_list[i]
            instance_y = instance_y_list[i]
            instance_timestamp -= time_begin
            instance_timestamp = np.round(instance_timestamp, 2)
            ax.plot(instance_timestamp, instance_y, linestyle='-', label=f"instance_{i}", linewidth=0.5)

        ax.legend(loc='upper left')
        ax.set_xlabel('timestamp(s)')
        ax.set_ylabel(y_label_str)
    
    plot_single(ax1, "num_block_first_waiting_request")
    plot_single(ax2, "num_block_all_waiting_request")
    plot_single(ax3, "waiting_time_first_waiting_request")
    plot_single(ax4, "num_waiting_request")

    fig_filename = os.path.splitext(results_filename)[0] + "_waiting" + ".png"
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def plot_platency_compare(json_file_list, priority_type = 0):
    latency_info_list = []
    for json_file in json_file_list:
        with open(json_file, 'r') as f:
            latency_info_list.append(json.load(f)[0])

    results_filename = json_file_list[-1]

    fig, (ax_request, ax_prefill, ax_decode) = plt.subplots(1, 3, figsize=(3*7, 4.8))

    def plot_single(ax, key):
        # mean, p50, p80, p95, p99
        if key == 'request':
            column_name = "request_latencies"
            y_label = 'latency(s)'
        elif key == 'prefill':
            column_name = "prefill_latencies"
            y_label = 'latency(ms)'
        elif key == 'decode':
            column_name = "decode_latencies"
            y_label = 'latency(ms)'
        if priority_type == 1:
            column_name = "priority_"+column_name
        data = []
        for latency_info in latency_info_list:
            latencies = latency_info[column_name]
            data.append([np.mean(latencies), np.percentile(latencies, 50),
                        np.percentile(latencies, 80), np.percentile(latencies, 95),
                        np.percentile(latencies, 99), np.percentile(latencies, 99.9)])
        # 绘制柱状图
        x = np.arange(len(stats))
        width = 0.25  # 柱子的宽度
        for i in range(len(data)):
            ax.bar(x + i * width, data[i], width, label=stats[i])
        ax.set_ylabel(y_label)
        ax.set_xticks(x + width)  # 设置刻度位置
        ax.set_xticklabels(stats)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.2)  # 将y轴上限增加为原来的1.2倍
        # 添加标签
        for i in range(len(data)):
            for j in range(len(stats)):
                value = data[i][j]
                value_label = "{:.2f}".format(value)  # 保留两位小数
                ax.text(x[j] + i * width, value + 0.01 * (ymax - ymin), value_label, ha='center', va='bottom', rotation=90)
        # 计算每个柱子相对于第一个柱子的提高百分比
        percent_diff = []
        for i in range(len(stats)):
            mean_diff = (data[-1][i] - data[0][i]) / data[-1][i] * 100
            percent_diff.append(mean_diff)

        # 添加百分比标签
        for i in range(len(stats)):
            label = f'{percent_diff[i]:.2f}%'
            ax.text(x[i] + width * (len(data) - 1) / 2, -0.13 * (ymax - ymin), label, ha='center', va='bottom')

    stats = ['mean', 'P50', 'P80', 'P95', 'P99', 'P99.9']
    plot_single(ax_request, 'request')
    plot_single(ax_prefill, 'prefill')
    plot_single(ax_decode, 'decode')
    ax_request.set_title('request latency')
    ax_prefill.set_title('prefill latency')
    ax_decode.set_title('decode latency')

    fig_filename = os.path.splitext(results_filename)[0] + "_platency_compare_{}".format(priority_type) + ".png"
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

def get_priority_latency_json(json_file_list, priority_type=0,filename="",cv=0):
    latency_info_list = []
    for json_file in json_file_list:
        with open(json_file, 'r') as f:
            latency_info_list.append(json.load(f)[0])
        
    results_filename = json_file_list[-1]
    
    fig, (ax_request, ax_prefill, ax_decode) = plt.subplots(1, 3, figsize=(3*7, 4.8))
    output_dict_disable = {}
    output_dict_enable = {}
    def plot_single(ax, key):
        # mean, p50, p80, p95, p99
        if key == 'Request':
            column_name = "request_latencies"
            y_label = 'latency(s)'
        elif key == 'Prefill':
            column_name = "prefill_latencies"
            y_label = 'latency(ms)'
        elif key == 'Decode':
            column_name = "decode_latencies"
            y_label = 'latency(ms)'
        elif key == "Inference":
            column_name = "inference_latencies"
        data = []
        if priority_type == 1:
            column_name = "priority_"+column_name
        for idx, latency_info in enumerate(latency_info_list):
            latencies = latency_info[column_name]
            # data.append([np.mean(latencies), np.percentile(latencies, 50),
            #             np.percentile(latencies, 80), np.percentile(latencies, 95),
            #             np.percentile(latencies, 99), np.percentile(latencies, 99.9)])
            if idx==0:
                output_dict_disable[key]=[np.percentile(latencies, 99),np.mean(latencies)]\
                if key != "Inference" else [np.mean(latencies)]
            else:
                output_dict_enable[key]=[np.percentile(latencies, 99),np.mean(latencies)]\
                if key != "Inference" else [np.mean(latencies)]

        # # 绘制柱状图
        # x = np.arange(len(stats))
        # width = 0.25  # 柱子的宽度
        # for i in range(len(data)):
        #     ax.bar(x + i * width, data[i], width, label=stats[i])
        # ax.set_ylabel(y_label)
        # ax.set_xticks(x + width)  # 设置刻度位置
        # ax.set_xticklabels(stats)
        # ymin, ymax = ax.get_ylim()
        # ax.set_ylim(ymin, ymax * 1.2)  # 将y轴上限增加为原来的1.2倍
        # # 添加标签
        # for i in range(len(data)):
        #     for j in range(len(stats)):
        #         value = data[i][j]
        #         value_label = "{:.2f}".format(value)  # 保留两位小数
        #         ax.text(x[j] + i * width, value + 0.01 * (ymax - ymin), value_label, ha='center', va='bottom', rotation=90)
        # # 计算每个柱子相对于第一个柱子的提高百分比
        # percent_diff = []
        # for i in range(len(stats)):
        #     mean_diff = (data[-1][i] - data[0][i]) / data[-1][i] * 100
        #     percent_diff.append(mean_diff)

        # # 添加百分比标签
        # for i in range(len(stats)):
        #     label = f'{percent_diff[i]:.2f}%'
        #     ax.text(x[i] + width * (len(data) - 1) / 2, -0.13 * (ymax - ymin), label, ha='center', va='bottom')

    stats = ['mean', 'P50', 'P80', 'P95', 'P99', 'P99.9']
    plot_single(ax_request, 'Request')
    plot_single(ax_prefill, 'Prefill')
    plot_single(ax_decode, 'Decode')
    plot_single(ax_decode, 'Inference')
    results = {}
    try:
        with open(filename+f"_enable{priority_type}.json", 'r') as f:
            results = json.load(f)
    except json.decoder.JSONDecodeError:
        pass
    except FileNotFoundError:
        os.mknod(filename+f"_enable{priority_type}.json")

    with open(filename+f"_enable{priority_type}.json", 'w') as f:
        results[cv] = output_dict_enable
        json.dump(results, f, indent=4)
    
    try:
        with open(filename+f"_disable{priority_type}.json", 'r') as f:
            results = json.load(f)
    except json.decoder.JSONDecodeError:
        pass
    except FileNotFoundError:
        os.mknod(filename+f"_disable{priority_type}.json")

    with open(filename+f"_disable{priority_type}.json", 'w') as f:
        results[cv] = output_dict_disable
        json.dump(results, f, indent=4)
    # ax_request.set_title('request latency')
    # ax_prefill.set_title('prefill latency')
    # ax_decode.set_title('decode latency')

    # fig_filename = os.path.splitext(results_filename)[0] + "_platency_compare"+f"_{priority_type}" + ".png"
    # index1 = fig_filename.rfind('/')
    # index2 = fig_filename.rfind('/', 0, index1)
    # fig_filename_title = fig_filename[index2 + 1:]
    # plt.suptitle(fig_filename_title, fontsize=6)
    # fig.savefig(fig_filename)

def get_lantency_metric_json(latency_enable_migrate, latency_disable_migrate, x_label="qps",filename=""):
    enable_list = None
    disable_list = None
    with open(latency_enable_migrate,'r') as f:
        enable_list = json.load(f)
    with open(latency_disable_migrate,'r') as f:
        disable_list = json.load(f)
    fig, (ax_thr, ax_prefill, ax_decode, ax_instance) = plt.subplots(1,4, figsize=(4*7, 4.8))
    x = []
    if x_label!="scale_up_threshold":
        enable_list = sorted(enable_list, key=lambda x: x[x_label])
        disable_list = sorted(disable_list, key=lambda x: x[x_label])
    throughput_list_enable = []
    throughput_list_disable = []
    prefill_p99_list_enable = []
    prefill_p99_list_disable = []
    decode_p99_list_enable = []
    decode_p99_list_disable = []
    request_p99_list_enable = []
    request_p99_list_disable = []

    instance_list_enable = []
    instance_list_disable = []
    prefill_mean_list_enable = []
    prefill_mean_list_disable = []
    decode_mean_list_enable = []
    decode_mean_list_disable = []
    request_mean_list_enable = []
    request_mean_list_disable = []
    output_dict_disable = {}
    output_dict_enable = {}
    for i in range(len(enable_list)):
        if x_label=="scale_up_threshold":
            x.append(5+20*i)
        else:
            x.append(enable_list[i][x_label])
        throughput_enable =  np.percentile(enable_list[i]["throughput"], 99)
        throughput_list_enable.append(throughput_enable)
        throughput_disable =  np.percentile(disable_list[i]["throughput"], 99)
        throughput_list_disable.append(throughput_disable)
        request_p99_enable = np.percentile(enable_list[i]["request_latencies"], 99)
        # request_p99_enable_list.append(request_p99_enable)
        request_p99_disable = np.percentile(disable_list[i]["request_latencies"], 99)
        # request_p99_disable_list.append(request_p99_disable)


        prefill_p99_enable =  np.percentile(enable_list[i]["prefill_latencies"], 99)
        prefill_p99_list_enable.append(prefill_p99_enable)
        prefill_p99_disable =  np.percentile(disable_list[i]["prefill_latencies"], 99)
        prefill_p99_list_disable.append(prefill_p99_disable)
        decode_p99_enable =  np.percentile(enable_list[i]["decode_latencies"], 99)
        decode_p99_list_enable.append(decode_p99_enable)
        decode_p99_disable =  np.percentile(disable_list[i]["decode_latencies"], 99)
        decode_p99_list_disable.append(decode_p99_disable)
        instance_enable =  enable_list[i]["instance_num"]
        instance_list_enable.append(instance_enable)
        instance_disable =  disable_list[i]["instance_num"]
        instance_list_disable.append(instance_disable)

        request_mean_enable = np.mean(enable_list[i]["request_latencies"])
        # request_mean_enable_list.append(request_mean_enable)

        request_mean_disable = np.mean(disable_list[i]["request_latencies"])
        # request_mean_disable_list.append(request_mean_disable)

        prefill_mean_enable =  np.mean(enable_list[i]["prefill_latencies"])
        prefill_mean_list_enable.append(prefill_mean_enable)
        prefill_mean_disable =  np.mean(disable_list[i]["prefill_latencies"])
        prefill_mean_list_disable.append(prefill_mean_disable)
        decode_mean_enable =  np.mean(enable_list[i]["decode_latencies"])
        decode_mean_list_enable.append(decode_mean_enable)
        decode_mean_disable =  np.mean(disable_list[i]["decode_latencies"])
        decode_mean_list_disable.append(decode_mean_disable)

        output_dict_enable[x[i]] = {
            "Request":[request_p99_enable,request_mean_enable],
            "Prefill":[prefill_p99_enable,prefill_mean_enable],
            "Decode":[decode_p99_enable,decode_mean_enable],
            "Instance":[instance_enable]
        }

        output_dict_disable[x[i]] = {
            "Request":[request_p99_disable,request_mean_disable],
            "Prefill":[prefill_p99_disable,prefill_mean_disable],
            "Decode":[decode_p99_disable,decode_mean_disable],
            "Instance":[instance_disable]
        }

    with open(filename+"_enable.json", 'w') as f:
        json.dump(output_dict_enable, f, indent=4)  # 添加4个空格的缩进
    
    with open(filename+"_disable.json", 'w') as f:
        json.dump(output_dict_disable, f, indent=4) 

def check_kv_cache():
    for i in range(2, 10):
        for j in range(30):
            tensor_send = torch.load(
                f"/mnt/wencong.xwc/huangziming/tensors/send_{i}key_{j}.pt", map_location="cpu")
            tensor_recv = torch.load(
                f"/mnt/wencong.xwc/huangziming/tensors/recv_{i}key_{j}.pt", map_location="cpu")
            print(torch.max(tensor_recv-tensor_send))

def get_avg_req_preemption_loss(results_filename):
    df = pd.read_csv(results_filename + "_req.csv").drop_duplicates()
    df = df.sort_values(by='timestamp')
    request_num = len(df["req_id"].drop_duplicates())
    preemption_time_sum = 0
    last_killed_time = defaultdict(lambda: 0.0)
    last_killed_len = defaultdict(lambda: 0.0)

    database = ProfilingDatabase("/mnt/wencong.xwc/huangziming/vllm/vllm/simulator/profiling_result_new.pkl", False)
    profiling_result = database.get("llama-7b")
    sim_parallel_config = SimParallelConfig(1,1)
    latency_mem = profiling_result.para_dict[sim_parallel_config]
    
    for idx, row in df.iterrows():
        req_id = row["req_id"]
        if row["event"] == "prefill" and last_killed_time[req_id]:
            preemption_time_sum += row["timestamp"] - last_killed_time[req_id]
            prompt_len = last_killed_len[req_id]
            prompt_len = _pad_to_alignment(prompt_len, 8)
            preemption_time_sum += latency_mem.prefill_latency[(1, prompt_len)][0] - latency_mem.decode_latency[(8, last_killed_len[req_id])][0]        
        elif row["event"] == "killed":
            last_killed_time[req_id] = row["timestamp"]
            last_killed_len[req_id] = row["output_len"]
    avg_preemption_loss = preemption_time_sum / request_num * 1000 # ms
    print(f"avg_preemption loss:{avg_preemption_loss}")
    return avg_preemption_loss

def get_req_preemption_loss(results_filename):
    df = pd.read_csv(results_filename + "_req.csv").drop_duplicates()
    df = df.sort_values(by='timestamp')
    req_preemption_loss = defaultdict(lambda: 0.0)
    last_killed_time = defaultdict(lambda: 0.0)
    last_killed_len = defaultdict(lambda: 0.0)

    database = ProfilingDatabase("/mnt/wencong.xwc/huangziming/vllm/vllm/simulator/profiling_result_new.pkl", False)
    profiling_result = database.get("llama-7b")
    sim_parallel_config = SimParallelConfig(1,1)
    latency_mem = profiling_result.para_dict[sim_parallel_config]
    
    for idx, row in df.iterrows():
        req_id = row["req_id"]
        if row["event"] == "prefill" and last_killed_time[req_id]:
            req_preemption_loss[req_id] += row["timestamp"] - last_killed_time[req_id]
            prompt_len = last_killed_len[req_id]
            prompt_len = _pad_to_alignment(prompt_len, 8)
            req_preemption_loss[req_id] += (latency_mem.prefill_latency[(1, prompt_len)][0] - latency_mem.decode_latency[(8, last_killed_len[req_id])][0]) / 1000
        elif row["event"] == "killed":
            last_killed_time[req_id] = row["timestamp"]
            last_killed_len[req_id] = row["output_len"]
        
    return req_preemption_loss

def get_req_decode_latency_sum(results_filename):
    json_filename = os.path.splitext(results_filename)[0] + "_latency_info.json"
    with open(json_filename, 'r') as f:
        latency_info = json.load(f)[0]
    req_ids = latency_info['request_ids']
    decode_latencies_sum = latency_info['decode_latencies_sum']
    req_decode_latency_sum = {}
    for i in range(len(req_ids)):
        req_decode_latency_sum[req_ids[i]] = decode_latencies_sum[i] / 1000
        
    return req_decode_latency_sum

def get_req_decode_latency_token(results_filename):
    json_filename = os.path.splitext(results_filename)[0] + "_latency_info.json"
    with open(json_filename, 'r') as f:
        latency_info = json.load(f)[0]
    req_ids = latency_info['request_ids']
    decode_latencies = latency_info['decode_latencies']
    req_decode_latency_token = {}
    for i in range(len(req_ids)):
        req_decode_latency_token[req_ids[i]] = decode_latencies[i]
        
    return req_decode_latency_token

def get_req_latency(results_filename):
    json_filename = os.path.splitext(results_filename)[0] + "_latency_info.json"
    with open(json_filename, 'r') as f:
        latency_info = json.load(f)[0]
    req_ids = latency_info['request_ids']
    req_latencies = latency_info['request_latencies']
    req_latency = {}
    for i in range(len(req_ids)):
        req_latency[req_ids[i]] = req_latencies[i] 
        
    return req_latency

def get_req_decode_len(results_filename):
    json_filename = os.path.splitext(results_filename)[0] + "_latency_info.json"
    with open(json_filename, 'r') as f:
        latency_info = json.load(f)[0]
    req_ids = latency_info['request_ids']
    decode_lens = latency_info['request_lens']
    req_decode_len = {}
    for i in range(len(req_ids)):
        req_decode_len[req_ids[i]] = decode_lens[i] 
        
    return req_decode_len
    
def get_avg_req_waiting_loss(results_filename,results_json_file):
    all_e2e_latencies, all_inference_latencies, all_waiting_latencies, \
        all_total_tokens, all_prompt_lens, all_response_lens = read_data_from_log_file(results_filename)

    with open(results_json_file,'r') as f:
        results_json = json.load(f)
    prefill_latencies = results_json[0]["prefill_latencies"]
    database = ProfilingDatabase("/mnt/wencong.xwc/huangziming/vllm/vllm/simulator/profiling_result_new.pkl", False)
    profiling_result = database.get("llama-7b")
    sim_parallel_config = SimParallelConfig(1,1)
    latency_mem = profiling_result.para_dict[sim_parallel_config]
    all_waiting_loss = []
    for i in range(len(prefill_latencies)):
        prompt_len = all_prompt_lens[i]
        prompt_len = _pad_to_alignment(prompt_len, 8)
        inference_latency = latency_mem.prefill_latency[(1, prompt_len)][0]
        all_waiting_loss.append(prefill_latencies[i] - inference_latency)
    avg_waiting_loss = np.mean(all_waiting_loss) # ms
    print(f"avg_waiting_loss:{avg_waiting_loss}")
    return avg_waiting_loss

    # TODO(woosuk): Unnecessary copy. Optimize.
    # output.copy_(out.squeeze(0))
    t1 = time.time()
    print(f'time{(t1 - t0)*1000}')

def get_avg_latency(results_filename, start_time , stop_time):
    df = pd.read_csv(results_filename + "_instance.csv").drop_duplicates()
    latency_list = []
    for idx, row in df.iterrows():
        if row["timestamp"] > start_time and row["timestamp"] < stop_time:
            latency_list.append(row["latency"])
    print(latency_list)
    print(np.max(latency_list))
    print(f"avg:{np.mean(latency_list)}")

def get_migrate_time(results_filename):
    df = pd.read_csv(results_filename + "_req.csv").drop_duplicates()
    df = df.sort_values(by='timestamp')
    tot_migrate_time=0
    for idx, row in df.iterrows():
        if row["event"]=="migrate_in":
            tot_migrate_time += 0.1
    # tot_migrate_time /= 16
    print(tot_migrate_time)
    print(f"avg migrate precent:{tot_migrate_time/(df['timestamp'].iloc[-1]-df['timestamp'].iloc[0])}")

def plot_mem_v2(results_filename, instance_num=2):
    df = pd.read_csv(results_filename+"_instance.csv").drop_duplicates()
    tot_steps = 0
    time_begin = np.inf
    time_end = 0
    instance_timestamp_list = []
    instance_gpu_list = []
    instance_steps_list = []
    for instance_id in range(instance_num):
        instance = df[df["instance_id"] == instance_id]
        instance_steps = len(instance)
        tot_steps = max(tot_steps, instance_steps)
        instance_steps_list.append(instance_steps)
        instance_timestamp = instance["timestamp"].to_numpy()
        instance_gpu = instance["gpu_cache_usage"].to_numpy()
        instance_timestamp_list.append(instance_timestamp)
        instance_gpu_list.append(instance_gpu)
        if len(instance_timestamp):
            time_begin = min(time_begin, instance_timestamp[0])
            time_end = max(time_end,instance_timestamp[-1])
    
    fig, ax = plt.subplots()
    tot_avg_gpu_usage = 0
    for i in range(instance_num):
        instance_timestamp = instance_timestamp_list[i]
        instance_gpu = instance_gpu_list[i]
        instance_timestamp -= time_begin
        instance_timestamp = np.round(instance_timestamp, 2)
        step_precent = np.round(instance_steps_list[i]/tot_steps*100, 2)
        area = np.trapz(instance_gpu, instance_timestamp)
        avg_gpu_usage = area/(time_end - time_begin)*100
        tot_avg_gpu_usage += avg_gpu_usage
        ax.plot(instance_timestamp, instance_gpu, label=f"instance_{i}:{np.round(avg_gpu_usage,2)}%", linewidth=0.5)
    tot_avg_gpu_usage/=instance_num
    print(f"tot avg gpu usage:{np.round(tot_avg_gpu_usage,2)}%")
    ax.legend(loc='upper left')
    ax.set_xlabel('timestamp(s)')
    ax.set_ylabel('gpu usage(%)')
    df = pd.read_csv(results_filename+"_req.csv")
    migrate_in_df = df[(df['event']=="migrate_in")]
    migrate_ts = migrate_in_df['timestamp'].to_numpy() - time_begin
    ax.scatter(migrate_ts, np.zeros_like(migrate_ts), c='green')
    fig_filename = os.path.splitext(results_filename)[0] + "_mem.png"
    fig.savefig(fig_filename)

def plot_priority(results_filename, instance_num=2):
    df = pd.read_csv(results_filename+"_instance.csv").drop_duplicates()
    tot_steps = 0
    time_begin = np.inf
    time_end = 0
    instance_timestamp_list = []
    instance_priority_list = []
    instance_steps_list = []
    gpu_usage_list = []
    gpu_usage_list_low = []
    for instance_id in range(instance_num):
        instance = df[df["instance_id"] == instance_id]
        instance_steps = len(instance)
        tot_steps = max(tot_steps, instance_steps)
        instance_steps_list.append(instance_steps)
        instance_timestamp = instance["timestamp"].to_numpy()
        instance_priority = instance["num_priority_request"].to_numpy()
        instance_timestamp_list.append(instance_timestamp)
        instance_priority_list.append(instance_priority)
        if len(instance_timestamp):
            time_begin = min(time_begin, instance_timestamp[0])
            time_end = max(time_end,instance_timestamp[-1])
    for idx, row in df.iterrows():
        instance_priority = row["num_priority_request"]
        num_running_request = row["num_running_request"]
        gpu_usage_list.extend(instance_priority*[row["gpu_cache_usage"]])
        gpu_usage_list_low.extend(num_running_request*[row["gpu_cache_usage"]])
    fig, ax = plt.subplots()
    for i in range(instance_num):
        instance_timestamp = instance_timestamp_list[i]
        instance_priority_num = instance_priority_list[i]
        instance_timestamp -= time_begin
        instance_timestamp = np.round(instance_timestamp, 2)
        mean_priority_num = np.round(np.mean(instance_priority_num),2)
        print(f"instance {i}:{mean_priority_num}")
        ax.plot(instance_timestamp, instance_priority_num, label=f"instance_{i}:{mean_priority_num}", linewidth=0.5)
    print(f"avg usage:{np.mean(gpu_usage_list)}")
    print(f"avg usage:{np.mean(gpu_usage_list_low)}")
    ax.legend(loc='upper left')
    ax.set_xlabel('timestamp(s)')
    ax.set_ylabel('num priority reqs')
    fig_filename = os.path.splitext(results_filename)[0] + "_priority_num.png"
    fig.savefig(fig_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--log-file1', type=str)
    parser.add_argument('--log-file2', type=str)
    parser.add_argument('--log-file3', type=str)
    parser.add_argument('--log-file4', type=str)
    args = parser.parse_args()

    log_file_list = []
    if args.log_file1:
        log_file_list.append(args.log_file1)
    if args.log_file2:
        log_file_list.append(args.log_file2)
    if args.log_file3:
        log_file_list.append(args.log_file3)
    if args.log_file4:
        log_file_list.append(args.log_file3)
    
    if args.test:
        pass
    elif len(log_file_list) == 1:
        # plot_mem_v0(args.log_file1)
        # plot_mem_v1(args.log_file1, show_killed=True, show_migrate=True)
        plot_mem_v2(args.log_file1, instance_num=16)
        plot_len_latency(args.log_file1, len_key='prompt')
        plot_len_latency(args.log_file1, len_key='response')
        plot_len_latency(args.log_file1, len_key='total')
        # plot_waiting(args.log_file1, instance_num=1)
    elif len(log_file_list) == 2:
        # json
        # plot_scaling_p99latency_compare(args.log_file1, args.log_file2)
        # npy
        # plot_interval_compare(args.log_file1, args.log_file2)
        # plot_len_latency_diff(args.log_file1, args.log_file2, len_key='prompt')
        # plot_len_latency_diff(args.log_file1, args.log_file2, len_key='response')
        # plot_len_latency_diff(args.log_file1, args.log_file2, len_key='total')
        # log
        plot_len_latency_diff_v2(args.log_file1, args.log_file2, len_key='prompt')
        plot_len_latency_diff_v2(args.log_file1, args.log_file2, len_key='response')
        plot_len_latency_diff_v2(args.log_file1, args.log_file2, len_key='total')
    else:
        # json
        plot_platency_compare(log_file_list)

if __name__ == '__main__':
    main()
