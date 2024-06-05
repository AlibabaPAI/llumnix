import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def get_avg_latency(results_filename, start_time , stop_time, begin_time = 0):
    df = pd.read_csv(results_filename + "_instance.csv").drop_duplicates()
    latency_list = []

    start_time += begin_time
    stop_time += begin_time
    for idx, row in df.iterrows():
        if row["timestamp"] > start_time and row["timestamp"] < stop_time:
            latency_list.append(row["latency"])
    print(f"avg:{np.mean(latency_list)}")
    return np.mean(latency_list)

def get_migration_timestamp(results_filename):
    df = pd.read_csv(results_filename + "_req.csv").drop_duplicates()
    df = df.sort_values("timestamp", ascending=True)
    stage0_begin_timestamp = None
    stage1_begin_timestamp = None
    stage1_end_timestamp = None
    for idx, row in df.iterrows():
        if row["event"] == "migrate_out_stage_0" and not stage0_begin_timestamp:
            stage0_begin_timestamp = row["timestamp"]
        if row["event"] == "migrate_out_stage_a" and not stage1_begin_timestamp:
            stage1_begin_timestamp = row["timestamp"]
        if row["event"] == "migrate_in" and not stage1_end_timestamp:
            stage1_end_timestamp = row["timestamp"]

    return stage0_begin_timestamp, stage1_begin_timestamp, stage1_end_timestamp



def plot_migration_microbenchmark(results_file_list_7b, results_file_list_30b):
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(2*4.2, 3.4))
    plt.subplots_adjust(bottom=0.15)  # 调整边距

    plt.rcParams.update({'font.size': 10})

    seq_len7b_x = np.log2([256,512,1024,2048,4096,8192])
    seq_len30b_x = np.log2([256,512,1024,2048,4096,8192])
    # profiled recompute latency
    recompute_7b = [58,110,220,436,940,1995]
    recompute_30b = [115,235,443,829,1712,3546]

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

    for migration_7b, normal_7b in results_file_list_7b:
        # stage 0 end time==stage 1 begin time
        stage0_begin_timestamp, stage1_begin_timestamp, stage1_end_timestamp = get_migration_timestamp(migration_7b)
        blocking_7b.append(stage1_begin_timestamp-stage0_begin_timestamp)
        stage1_7b.append(stage1_end_timestamp - stage1_begin_timestamp)
        avg_decode_latency_migration = get_avg_latency(migration_7b, stage0_begin_timestamp, stage1_begin_timestamp)
        migration_decode_7b.append(avg_decode_latency_migration)
        avg_decode_latency_normal = get_avg_latency(normal_7b, stage0_begin_timestamp, stage1_begin_timestamp)
        decode_7b.append(avg_decode_latency_normal)

    for migration_30b, normal_30b in results_file_list_30b:
        # stage 0 end time==stage 1 begin time
        stage0_begin_timestamp, stage1_begin_timestamp, stage1_end_timestamp = get_migration_timestamp(migration_30b)
        blocking_30b.append(stage1_begin_timestamp-stage0_begin_timestamp)
        stage1_30b.append(stage1_end_timestamp - stage1_begin_timestamp)
        avg_decode_latency_migration = get_avg_latency(migration_30b, stage0_begin_timestamp, stage1_begin_timestamp)
        migration_decode_30b.append(avg_decode_latency_migration)
        avg_decode_latency_normal = get_avg_latency(normal_30b, stage0_begin_timestamp, stage1_begin_timestamp)
        decode_30b.append(avg_decode_latency_normal)



    # print(max((np.array(migration_decode_7b)-np.array(decode_7b))/np.array(decode_7b)))

    ax1.plot(seq_len7b_x, stage1_7b, label=f"Migration(7B)",color="tab:blue",marker="o")
    # ax1.plot(seq_len30b_x, stage1_30b, label=f"Migration(30B)",color="tab:blue",marker="o",linestyle='--')

    ax1.plot(seq_len7b_x, blocking_7b, label=f"Blocking copy(7B)",color="tab:orange",marker="v")
    # ax1.plot(seq_len30b_x, blocking_30b, label=f"Blocking copy(30B)",color="tab:orange",marker="v",linestyle='--')

    ax1.plot(seq_len7b_x, recompute_7b, label=f"Recompute(7B)",color="tab:green",marker="s")
    # ax1.plot(seq_len30b_x, recompute_30b, label=f"Recompute(30B)",color="tab:green",marker="s",linestyle='--')

    ax2.plot(seq_len7b_x,migration_decode_7b, label=f"Migration(7B)",color="tab:blue",marker="o")
    # ax2.plot(seq_len30b_x,migration_decode_30b, label=f"Migration(30B)",color="tab:blue",marker="o",linestyle='--')

    ax2.plot(seq_len7b_x,decode_7b, label=f"Normal(7B)",color="tab:orange",marker="v")
    # ax2.plot(seq_len30b_x,decode_30b, label=f"Normal(30B)",color="tab:orange",marker="v",linestyle='--')

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

    fig_filename = "microbenchmark1.pdf"
    fig.savefig(fig_filename)
if __name__ == '__main__':

    plot_migration_microbenchmark()