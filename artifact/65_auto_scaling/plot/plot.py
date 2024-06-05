import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse

def plot_trace2(trace_key, axs, qps_list, non_migrate_latency_dict, prefill_migrate_latency_dict, naive_dispatch_latency_dict=None):
    def plot_single_metric(ax, metric_key, metric_index):
        if metric_key == 'Preemption':
            non_migrate_values = [non_migrate_latency_dict[qps][metric_key][0] * non_migrate_latency_dict[qps][metric_key][1] / 10000
                                for qps in qps_list]
            prefill_migrate_values = [prefill_migrate_latency_dict[qps][metric_key][0] * prefill_migrate_latency_dict[qps][metric_key][1] / 10000
                                    for qps in qps_list]
        else:
            non_migrate_values = [non_migrate_latency_dict[qps][metric_key][metric_index] for qps in qps_list]
            prefill_migrate_values = [prefill_migrate_latency_dict[qps][metric_key][metric_index] for qps in qps_list]
        non_migrate_values = np.array(non_migrate_values)
        prefill_migrate_values = np.array(prefill_migrate_values)

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
        import colorsys
        scaled_colors = []
        for color in colors:
            hsv = colorsys.rgb_to_hsv(*color)
            scaled_hsv = (hsv[0], min(hsv[1] * scale_factor, 1.0), hsv[2])
            scaled_rgb = colorsys.hsv_to_rgb(*scaled_hsv)
            scaled_colors.append(scaled_rgb)
        colors = scaled_colors

        # if metric_key!="Instance" and metric_key!="Request" and metric_key!="Inference":
        #     non_migrate_values /= 1000
        #     prefill_migrate_values /= 1000
        ax.plot(qps_list, non_migrate_values, marker='o', linestyle='--', color=colors[0], label='Llumnix-base')
        ax.plot(qps_list, prefill_migrate_values, marker='s', linestyle='-', color=colors[1], label='Llumnix')

        # if trace_key == '256-256':
        #     naive_dispatch_values = [naive_dispatch_latency_dict[qps][metric_key][metric_index] for qps in qps_list]
        #     ax.plot(qps_list, naive_dispatch_values, marker='s', linestyle='-', color='green', label='Naive')

        fontsize=15

        if metric_key == 'Request' and metric_index == 0:
            ax.set_ylabel(trace_key + '\n\n' + 'Latency (s)', fontsize=fontsize)
        elif metric_key == 'Instance':
            ax.set_ylabel("Avg Instance Num", fontsize=fontsize)

        if metric_index == 1:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0 - ymax * 0.1, ymax + ymax * 0.1)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0 - ymax * 0.1, ymax + ymax * 0.1)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        if trace_key == "Poisson":
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
                # ax.text(1.05, 0.5, "\n\n", transform=ax.transAxes, rotation=-90, va="center")
            elif metric_key == 'Instance':
                ax.set_title('Resource Cost', fontsize=fontsize)
        # if metric_key == "Instance":
        #     ax.yaxis.set_label_position('right')
        #     ax.yaxis.tick_right()

        if trace_key == "Poisson":
            ax.set_xlabel("Request Rate", fontsize=fontsize)
        elif trace_key == "Gamma":
            ax.set_xlabel("CV", fontsize=fontsize)

        ax.grid(True)
        ax.set_xticks(qps_list)

    metric_keys = ['Request', 'Prefill', 'Decode', 'Instance']
    for i in range(3 * 2 + 1):
        metric_key = metric_keys[int(i / 2)]
        metric_index = i % 2
        plot_single_metric(axs[i], metric_key, metric_index)

def get_evaluation2_data(trace_key, log_trace_path ,key_list):
    trace_data = {}
    for key in key_list:
        trace_data[key] = {
            'Request': [],
            'Prefill': [],
            'Decode': [],
            'Instance': [],
        }
    for root, dirs, files in os.walk(log_trace_path):
    # import pdb; pdb.set_trace()
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
                        if trace_key=="Gamma":
                            key = "CV"
                        elif trace_key=="Poisson":
                            key = "QPS"
                        else:
                            key = "Scale"
                        key=file_data[key]
                        trace_data[key]['Request'].append(file_data["Request P99"])
                        trace_data[key]['Request'].append(file_data["Request Mean"])
                        trace_data[key]['Prefill'].append(file_data["Prefill P99"])
                        trace_data[key]['Prefill'].append(file_data["Prefill Mean"])
                        trace_data[key]['Decode'].append(file_data["Decode P99"])
                        trace_data[key]['Decode'].append(file_data["Decode Mean"])
                        trace_data[key]['Instance'].append(file_data["Avg Instance Num"])

    return trace_data
def plot_figure14_paper(log_path):
    wspace = 0.35
    hspace = 0.45
    fig, axs = plt.subplots(nrows=2, ncols=3 * 2 + 1, figsize=(3.0 * 7 + wspace * 6, 1.6 * 2 + hspace * 4))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs_dict = {}
    trace_keys = ["Poisson", "Gamma"]
    cv_list = [2,3,4,5,6]
    qps_list = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

    for row in range(len(trace_keys)):
        axs_dict[trace_keys[row]] = axs[row]
    for trace_key in trace_keys:
        key_list = cv_list if trace_key=="Gamma" else qps_list
        trace_path = os.path.join(log_path, trace_key)
        trace_data_llumnix = get_evaluation2_data(trace_key, os.path.join(trace_path,'Llumnix'),key_list)
        trace_data_baseline = get_evaluation2_data(trace_key, os.path.join(trace_path,'INFaas++'),key_list)
        non_migrate_latency_dict, prefill_migrate_latency_dict, naive_dispatch_latency_dict = trace_data_baseline, trace_data_llumnix, {}
        plot_trace2(trace_key, axs_dict[trace_key], key_list, non_migrate_latency_dict, prefill_migrate_latency_dict, naive_dispatch_latency_dict)
        request_mean_improved_ratios = [trace_data_baseline[key]["Request"][1]/trace_data_llumnix[key]["Request"][1] for key in key_list]
        request_P99_improved_ratios = [trace_data_baseline[key]["Request"][0]/trace_data_llumnix[key]["Request"][0] for key in key_list]
        prefill_mean_improved_ratios = [trace_data_baseline[key]["Prefill"][1]/trace_data_llumnix[key]["Prefill"][1] for key in key_list]
        prefill_P99_improved_ratios = [trace_data_baseline[key]["Prefill"][0]/trace_data_llumnix[key]["Prefill"][0] for key in key_list]
        decode_mean_improved_ratios = [trace_data_baseline[key]["Decode"][1]/trace_data_llumnix[key]["Decode"][1] for key in key_list]
        decode_P99_improved_ratios = [trace_data_baseline[key]["Decode"][0]/trace_data_llumnix[key]["Decode"][0] for key in key_list]
        resource_save_ratios = [trace_data_baseline[key]["Instance"][0]/trace_data_llumnix[key]["Instance"][0] - 1 for key in key_list]
        out_filename = "figure14.claim"
        with open(out_filename, 'w+') as out_file:
            out_file.write(f"Trace:{trace_key}:\n")
            out_file.write("Request Latency(s):\n")
            out_file.write("Request Mean outperforms up to: {:.2f}x\n".format(max(request_mean_improved_ratios)))
            out_file.write("Request P99 outperforms up to: {:.2f}x\n".format(max(request_P99_improved_ratios)))
            out_file.write("Prefill Latency(s):\n")
            out_file.write("Prefill Mean outperforms up to: {:.2f}x\n".format(max(prefill_mean_improved_ratios)))
            out_file.write("Prefill P99 outperforms up to: {:.2f}x\n".format(max(prefill_P99_improved_ratios)))
            out_file.write("Decode Latency(s):\n")
            out_file.write("Decode Mean outperforms up to: {:.2f}x\n".format(max(decode_mean_improved_ratios)))
            out_file.write("Decode P99 outperforms up to: {:.2f}x\n".format(max(decode_P99_improved_ratios)))
            out_file.write("Avg Instance Num:\n")
            out_file.write("Save Resource outperforms up to: {:.2f}%\n".format(max(resource_save_ratios)*100))
    handles, labels = fig.axes[0].get_legend_handles_labels()

    new_handles = [handles[0], handles[1]]
    new_labels = [labels[0], labels[1]]

    fontsize=15
    fig.legend(new_handles, new_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.93), fontsize=fontsize)

    fig.savefig("figure14.png", bbox_inches="tight")

def plot_figure15_paper(log_path):
    prefill_p99_list_enable = []
    prefill_p99_list_disable = []
    instance_num_list_enable = []
    instance_num_list_disable = []
    # prefill_p99_list_enable = [11.425, 4.047, 3.660, 3.508, 2.954]
    # prefill_p99_list_disable = [40.063, 21.507, 14.330, 4.587, 3.422]
    # instance_num_list_enable = [10.44, 11.39, 12.3, 13.3, 14.06]
    # instance_num_list_disable = [11.61, 12.61, 13.33, 14.25, 15.1]
    trace_path = os.path.join(log_path, "Threshold")
    threshold_list = [5,25,45,65,85]
    trace_data_llumnix = get_evaluation2_data("Scale", os.path.join(trace_path,'Llumnix'),threshold_list)
    trace_data_baseline = get_evaluation2_data("Scale", os.path.join(trace_path,'INFaas++'),threshold_list)

    for key in trace_data_baseline.keys():
        prefill_p99_list_disable.append(trace_data_baseline[key]["Prefill"][0])
        instance_num_list_disable.append(trace_data_baseline[key]["Instance"])

    for key in trace_data_llumnix.keys():
        prefill_p99_list_enable.append(trace_data_llumnix[key]["Prefill"][0])
        instance_num_list_enable.append(trace_data_llumnix[key]["Instance"])

    fontsize=10

    fig, (ax) = plt.subplots(1, 1, figsize=(3, 2.5))
    plt.subplots_adjust(bottom=0.15)
    plt.rcParams.update({'font.size': fontsize})
    prefill_p99_list_disable = np.array(prefill_p99_list_disable)
    prefill_p99_list_enable = np.array(prefill_p99_list_enable)

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
    import colorsys
    scaled_colors = []
    for color in colors:
        hsv = colorsys.rgb_to_hsv(*color)
        scaled_hsv = (hsv[0], min(hsv[1] * scale_factor, 1.0), hsv[2])
        scaled_rgb = colorsys.hsv_to_rgb(*scaled_hsv)
        scaled_colors.append(scaled_rgb)
    colors = scaled_colors

    ymin, ymax = ax.get_ylim()

    ax.plot(instance_num_list_disable, prefill_p99_list_disable, marker='x', linestyle='-.', color=colors[0], label='INFaaS++')
    ax.plot(instance_num_list_enable, prefill_p99_list_enable, marker='o', linestyle='-.', color=colors[1], label='Llumnix')
    # plt.axhline(y=4.5, color='r', linestyle='--')
    # ax.text(14.0, 4.5 + 3.8 * (ymax - ymin), f'Cost Saving: 36.49%', verticalalignment='top', horizontalalignment='right', color='red', fontsize=10)

    ax.set_xlabel("Avg Instance Num")
    ax.set_ylabel("P99 Prefill Latency (s)")
    # print(prefill_p99_list_enable,prefill_p99_list_disable)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    ax.grid(True)

    fig.savefig("figure15.png", bbox_inches="tight")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', default='../log', type=str)
    args = parser.parse_args()

    plot_figure14_paper(args.log_path)
    plot_figure15_paper(args.log_path)
