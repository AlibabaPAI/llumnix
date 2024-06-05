import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import colorsys
import argparse


request_P99_improved_ratios = []
request_mean_improved_ratios = []
prefill_P99_improved_ratios = []
prefill_mean_improved_ratios = []
decode_P99_improved_ratios = []
decode_mean_improved_ratios = []
preemption_loss_reduced_ratios = []
preemption_loss_reduced_values = []


def get_colors():
    # blue
    red, green, blue = 68, 114, 196
    color1 = (red / 255.0, green / 255.0, blue / 255.0)
    # orange
    red, green, blue = 237, 125, 49
    color2 = (red / 255.0, green / 255.0, blue / 255.0)
    # green
    red, green, blue = 112, 173, 71
    color3 = (red / 255.0, green / 255.0, blue / 255.0)

    colors = [color1, color2, color3]

    scale_factor = 10.0
    scaled_colors = []
    for color in colors:
        hsv = colorsys.rgb_to_hsv(*color)
        scaled_hsv = (hsv[0], min(hsv[1] * scale_factor, 1.0), hsv[2])
        scaled_rgb = colorsys.hsv_to_rgb(*scaled_hsv)
        scaled_colors.append(scaled_rgb)
    colors = scaled_colors

    return colors


def get_trace_data(log_path, trace):
    if trace == 'ShareGPT':
        log_trace_path = os.path.join(log_path, "sharegpt")
    elif trace == 'BurstGPT':
        log_trace_path = os.path.join(log_path, "burstgpt-conversation")
    else:
        log_trace_path = os.path.join(log_path, trace.replace('-', '_'))

    if trace == 'ShareGPT':
        qps_list = [7.00, 7.25, 7.50, 7.75, 8.00]
    if trace == 'BurstGPT':
        qps_list = [7.5, 7.75, 8.0, 8.25, 8.5]
    if trace == "128-128":
        qps_list = [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0]
    elif trace == "256-256":
        qps_list = [7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0]
    elif trace == "512-512":
        qps_list = [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]
    elif trace == "128-512":
        qps_list = [4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40]
    elif trace == "512-128":
        qps_list = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
    else:
        assert("figure11 onlys contains 7 traces: ShareGPT, BurstGPT, 128-128, 256-256, 512-512, 128-512, 512-128")
    trace_data = {}
    for qps in qps_list:
        trace_data[qps] = {
            'INFaaS++' : {
                'Request P99': [],
                'Request Mean': [],
                'Prefill P99': [],
                'Prefill Mean': [],
                'Decode P99': [],
                'Decode Mean': [],
                'Preemption Loss': []
            },
            'Llumnix' : {
                'Request P99': [],
                'Request Mean': [],
                'Prefill P99': [],
                'Prefill Mean': [],
                'Decode P99': [],
                'Decode Mean': [],
                'Preemption Loss': []
            },
            'Round-Robin' : {
                'Request P99': [],
                'Request Mean': [],
                'Prefill P99': [],
                'Prefill Mean': [],
                'Decode P99': [],
                'Decode Mean': [],
                'Preemption Loss': []
            }
        }
    metrics = ['Request P99', 'Request Mean', 'Prefill P99', 'Prefill Mean', 'Decode P99', 'Decode Mean', 'Preemption Loss']
    for root, dirs, files in os.walk(log_trace_path):
        for file in files:
            if file.endswith('.data'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file:
                    file_data = {}
                    valid = True
                    for line in file:
                        parts = line.split(':')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value_part = parts[1].strip()
                            if key in ['Trace', 'Method']:
                                value = value_part
                            else:
                                value = float(value_part)
                            file_data[key] = value
                        else:
                            print("{} is invalid log data, skip it!".format(filepath))
                            valid = False
                            break
                    if valid:
                        if file_data['QPS'] in qps_list:
                            for metric in metrics:
                                trace_data[file_data['QPS']][file_data['Method']][metric].append(file_data[metric])
    
    return qps_list, trace_data


def get_figure11_data(log_path, trace):
    qps_list, trace_data = get_trace_data(log_path, trace)
    # latency_dict: QPS->'Request'/'Prefill'/'Decode'/'Preemption'->[P99, Mean]
    infaaspp_latency_dict = {}
    llumnix_latency_dict = {}
    round_robin_latency_dict = {}
    for latency_dict in [infaaspp_latency_dict, llumnix_latency_dict, round_robin_latency_dict]:
        for qps in qps_list:
            latency_dict[qps] = {
                'Request': [0.0, 0.0], 
                'Prefill': [0.0, 0.0], 
                'Decode': [0.0, 0.0], 
                'Preemption': [0.0]
            }
    methods = ['INFaaS++', 'Llumnix', 'Round-Robin']

    for qps in qps_list:
        for method in methods:
            if method == 'INFaaS++':
                latency_dict = infaaspp_latency_dict
            elif method == 'Llumnix':
                latency_dict = llumnix_latency_dict
            else: # method == 'Round-Robin'
                latency_dict = round_robin_latency_dict
            if len(trace_data[qps][method]['Request P99']) != 0:
                data_dict = trace_data[qps][method]
                latency_dict[qps]['Request'][0] = sum(data_dict['Request P99']) / len(data_dict['Request P99'])
                latency_dict[qps]['Request'][1] = sum(data_dict['Request Mean']) / len(data_dict['Request Mean'])
                latency_dict[qps]['Prefill'][0] = sum(data_dict['Prefill P99']) / len(data_dict['Prefill P99'])
                latency_dict[qps]['Prefill'][1] = sum(data_dict['Prefill Mean']) / len(data_dict['Prefill Mean'])
                latency_dict[qps]['Decode'][0] = sum(data_dict['Decode P99']) / len(data_dict['Decode P99'])
                latency_dict[qps]['Decode'][1] = sum(data_dict['Decode Mean']) / len(data_dict['Decode Mean'])
                latency_dict[qps]['Preemption'][0] = sum(data_dict['Preemption Loss']) / len(data_dict['Preemption Loss'])

    return qps_list, infaaspp_latency_dict, llumnix_latency_dict, round_robin_latency_dict


def collect_figure11_claims_data(metric_key, metric_index, infaaspp_values, llumnix_values):
    if metric_key == 'Request':
        # P99
        if metric_index == 0:
            curr_request_P99_improved_ratios = [infaaspp_values[i] / llumnix_values[i] if llumnix_values[i] != 0.0 else -1.0 for i in range(len(infaaspp_values))]
            request_P99_improved_ratios.extend(curr_request_P99_improved_ratios)
        # mean
        else:
            curr_request_mean_improved_ratios = [infaaspp_values[i] / llumnix_values[i] if llumnix_values[i] != 0.0 else -1.0 for i in range(len(infaaspp_values))]
            request_mean_improved_ratios.extend(curr_request_mean_improved_ratios)
    if metric_key == 'Prefill':
        # P99
        if metric_index == 0:
            curr_prefill_P99_improved_ratios = [infaaspp_values[i] / llumnix_values[i] if llumnix_values[i] != 0.0 else -1.0 for i in range(len(infaaspp_values))]
            prefill_P99_improved_ratios.extend(curr_prefill_P99_improved_ratios)
        # mean
        else:
            curr_prefill_mean_improved_ratios = [infaaspp_values[i] / llumnix_values[i] if llumnix_values[i] != 0.0 else -1.0 for i in range(len(infaaspp_values))]
            prefill_mean_improved_ratios.extend(curr_prefill_mean_improved_ratios)
    if metric_key == 'Decode':
        # P99
        if metric_index == 0:
            curr_decode_P99_improved_ratios = [infaaspp_values[i] / llumnix_values[i] if llumnix_values[i] != 0.0 else -1.0 for i in range(len(infaaspp_values))]
            decode_P99_improved_ratios.extend(curr_decode_P99_improved_ratios)
        # mean
        else:
            curr_decode_mean_improved_ratios = [infaaspp_values[i] / llumnix_values[i] if llumnix_values[i] != 0.0 else -1.0 for i in range(len(infaaspp_values))]
            decode_mean_improved_ratios.extend(curr_decode_mean_improved_ratios)
    if metric_key == 'Preemption':
        curr_preemption_loss_reduced_values = [infaaspp_values[i] - llumnix_values[i] if llumnix_values[i] != 0.0 else -1.0 for i in range(len(infaaspp_values))]
        preemption_loss_reduced_values.extend(curr_preemption_loss_reduced_values)
        curr_preemption_loss_reduced_raios = [(infaaspp_values[i] - llumnix_values[i]) / infaaspp_values[i] if llumnix_values[i] != 0.0 else -1.0 for i in range(len(infaaspp_values))]
        preemption_loss_reduced_ratios.extend(curr_preemption_loss_reduced_raios)


def validate_figure11_claims_paper():
    out_filename = "figure11.claim"
    with open(out_filename, 'w+') as out_file:
        for i in range(2):
            if i == 0:
                out_file.write("Real dataset\n")
                out_file.write("Llumnix vs. Round-Robin\n")
                start = 0
                end = 2 * 5
            else:
                out_file.write("Generated dataset\n")
                out_file.write("Llumnix vs. INFaaS++\n")
                start = 2 * 5
                end = 7 * (5 + 2)
            out_file.write("Request Latency(s):\n")
            out_file.write("Request Mean outperforms up to: {:.2f}x\n".format(max(request_mean_improved_ratios[start:end])))
            out_file.write("Request P99 outperforms up to: {:.2f}x\n".format(max(request_P99_improved_ratios[start:end])))
            out_file.write("Prefill Latency(s):\n")
            out_file.write("Prefill Mean outperforms up to: {:.2f}x\n".format(max(prefill_mean_improved_ratios[start:end])))
            out_file.write("Prefill P99 outperforms up to: {:.2f}x\n".format(max(prefill_P99_improved_ratios[start:end])))
            out_file.write("Decode Latency(s):\n")
            out_file.write("Decode Mean outperforms up to: {:.2f}x\n".format(max(decode_mean_improved_ratios[start:end])))
            out_file.write("Decode P99 outperforms up to: {:.2f}x\n".format(max(decode_P99_improved_ratios[start:end])))
            out_file.write("Preemption Loss(s):\n")
            out_file.write("Preemption Loss average reduction ratio: {:.2f}%\n".format(np.mean(preemption_loss_reduced_ratios[start:end]) * 100))
            out_file.write("Preemption Loss average reduction values: {:.2f}s\n\n".format(np.mean(preemption_loss_reduced_values[start:end])))


def plot_one_trace(trace, axs, qps_list, infaaspp_latency_dict, llumnix_latency_dict, round_robin_latency_dict=None):
    def plot_one_metric(ax, metric_key, metric_index):
        infaaspp_values = [infaaspp_latency_dict[qps][metric_key][metric_index] for qps in qps_list]
        llumnix_values = [llumnix_latency_dict[qps][metric_key][metric_index] for qps in qps_list]

        colors = get_colors()

        ax.plot(qps_list, infaaspp_values, marker='o', linestyle='--', color=colors[0], label='INFaaS++')
        ax.plot(qps_list, llumnix_values, marker='s', linestyle='-', color=colors[1], label='Llumnix')

        if trace in ['ShareGPT', 'BurstGPT']:
            round_robin_values = [round_robin_latency_dict[qps][metric_key][metric_index] for qps in qps_list]
            ax.plot(qps_list, round_robin_values, marker='s', linestyle=':', color=colors[2], label='Round-Robin')

        if trace in ['ShareGPT', 'BurstGPT']:
            collect_figure11_claims_data(metric_key, metric_index, round_robin_values, llumnix_values)
        else:
            collect_figure11_claims_data(metric_key, metric_index, infaaspp_values, llumnix_values)

        fontsize=15

        trace2ylabel = {
            'ShareGPT': 'ShareGPT',
            'BurstGPT': 'BurstGPT',
            '128-128': 'S-S',
            '256-256': 'M-M',
            '512-512': 'L-L',
            '128-512': 'S-L',
            '512-128': 'L-S'
        }

        if metric_key == 'Request' and metric_index == 0:
            ax.set_ylabel(trace2ylabel[trace] + '\n\n' + 'Latency (s)', fontsize=fontsize)

        if metric_index == 1:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0 - ymax * 0.1, ymax + ymax * 0.1)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0 - ymax * 0.1, ymax + ymax * 0.1)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        if trace == "ShareGPT":
            if metric_key == 'Request' and metric_index == 0:
                ax.set_title('Request P99', fontsize=fontsize)
            elif metric_key == 'Request' and metric_index == 1:
                ax.set_title('Request Mean', fontsize=fontsize)
            elif metric_key == 'Prefill' and metric_index == 0:
                ax.set_title('Prefill P99', fontsize=fontsize)
            elif metric_key == 'Prefill' and metric_index == 1:
                ax.set_title('Prefill Mean', fontsize=fontsize)
            elif metric_key == 'Decode' and metric_index == 0:
                ax.set_title('Decode P99', fontsize=fontsize)
            elif metric_key == 'Decode' and metric_index == 1:
                ax.set_title('Decode Mean', fontsize=fontsize)
            elif metric_key == 'Preemption':
                ax.set_title('Preemption Loss', fontsize=fontsize)
        if trace == "512-128":
            ax.set_xlabel("Request Rate", fontsize=fontsize)
        ax.grid(True)

    metric_keys = ['Request', 'Prefill', 'Decode', 'Preemption']
    for i in range(3 * 2 + 1):
        metric_key = metric_keys[int(i / 2)]
        metric_index = i % 2
        plot_one_metric(axs[i], metric_key, metric_index)


def plot_figure11_paper(log_path):
    traces = ["ShareGPT", "BurstGPT", "128-128", "256-256", "512-512", "128-512", "512-128"]

    wspace = 0.35
    hspace = 0.3
    nrows = 7
    ncols = 7
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.0 * ncols + wspace * (ncols - 1), 2.2 * nrows + hspace * (nrows - 1)))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs_dict = {}
    for row in range(len(traces)):
        axs_dict[traces[row]] = axs[row]

    for trace in traces:
        qps_list, infaaspp_latency_dict, llumnix_latency_dict, round_robin_latency_dict = get_figure11_data(log_path, trace)
        plot_one_trace(trace, axs_dict[trace], qps_list, infaaspp_latency_dict, llumnix_latency_dict, round_robin_latency_dict)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    new_handles = [handles[1], handles[0], handles[2]]
    new_labels = [labels[1], labels[0], labels[2]]
    fontsize=15
    fig.legend(new_handles, new_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.905), fontsize=fontsize)

    fig.savefig("figure11.png", bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', default='../log', type=str)
    args = parser.parse_args()

    plot_figure11_paper(args.log_path)
    validate_figure11_claims_paper()
