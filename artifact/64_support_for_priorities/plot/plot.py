import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import colorsys
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

        if trace_key == "Normal":
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
            elif metric_key == 'Inference':
                ax.set_title('Decode Execution Time', fontsize=fontsize)
        # if metric_key == "Instance":
        #     ax.yaxis.set_label_position('right')
        #     ax.yaxis.tick_right()

        # if trace_key == "Possion":
        #     ax.set_xlabel("Request Rate", fontsize=fontsize)
        # elif trace_key == "Gamma":
        #     ax.set_xlabel("CV", fontsize=fontsize)
        elif trace_key=="High Priority":
            ax.set_xlabel("CV", fontsize=fontsize)
        ax.grid(True)
        ax.set_xticks(qps_list)

    metric_keys = ['Request', 'Prefill', 'Decode', 'Inference']
    for i in range(3 * 2 + 1):
        metric_key = metric_keys[int(i / 2)]
        metric_index = i % 2
        plot_single_metric(axs[i], metric_key, metric_index)

def plot_figure13_paper(log_path):
    log_path = os.path.join(log_path, '128-128')
    wspace = 0.35
    hspace = 0.45
    fig, axs = plt.subplots(nrows=2, ncols=3 * 2 + 1, figsize=(3.0 * 7 + wspace * 6, 1.6 * 2 + hspace * 4))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs_dict = {}
    trace_keys = ["Normal", "High Priority"]
    cv_list=[2,4,6,8]
    def get_evaluation2_data(trace_key, log_trace_path):
        trace_data = {}
        for cv in cv_list:
            trace_data[cv] = {
                'Request': [],
                'Prefill': [],
                'Decode': [],
                'Inference': [],
            }
        for root, dirs, files in os.walk(log_trace_path):
            for file in files:
                if file.endswith('0.data' if trace_key=="Normal" else '1.data'):
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
                            cv=int(file_data["CV"])
                            trace_data[cv]['Request'].append(file_data["Request P99"])
                            trace_data[cv]['Request'].append(file_data["Request Mean"])
                            trace_data[cv]['Prefill'].append(file_data["Prefill P99"])
                            trace_data[cv]['Prefill'].append(file_data["Prefill Mean"])
                            trace_data[cv]['Decode'].append(file_data["Decode P99"])
                            trace_data[cv]['Decode'].append(file_data["Decode Mean"])
                            trace_data[cv]['Inference'].append(file_data["Decode Execution Time"])

        return trace_data

    for row in range(len(trace_keys)):
        axs_dict[trace_keys[row]] = axs[row]
    for trace_key in trace_keys:
        trace_data_llumnix = get_evaluation2_data(trace_key, os.path.join(log_path,'Llumnix'))
        trace_data_baseline = get_evaluation2_data(trace_key, os.path.join(log_path,'Llumnix-base'))
        non_migrate_latency_dict, prefill_migrate_latency_dict, naive_dispatch_latency_dict = trace_data_baseline, trace_data_llumnix, {}
        plot_trace2(trace_key, axs_dict[trace_key], cv_list, non_migrate_latency_dict, prefill_migrate_latency_dict, naive_dispatch_latency_dict)
        if trace_key == "Normal":
            request_mean_increase_ratios = [trace_data_llumnix[cv]["Request"][1]/trace_data_baseline[cv]["Request"][1] for cv in cv_list]
            request_P99_increase_ratios = [trace_data_llumnix[cv]["Request"][0]/trace_data_baseline[cv]["Request"][0] for cv in cv_list]
            prefill_mean_increase_ratios = [trace_data_llumnix[cv]["Prefill"][1]/trace_data_baseline[cv]["Prefill"][1] for cv in cv_list]
            prefill_P99_increase_ratios = [trace_data_llumnix[cv]["Prefill"][0]/trace_data_baseline[cv]["Prefill"][0] for cv in cv_list]
            decode_mean_increase_ratios = [trace_data_llumnix[cv]["Decode"][1]/trace_data_baseline[cv]["Decode"][1] for cv in cv_list]
            decode_P99_increase_ratios = [trace_data_llumnix[cv]["Decode"][0]/trace_data_baseline[cv]["Decode"][0] for cv in cv_list]
            inference_mean_increase_ratios = [trace_data_llumnix[cv]["Inference"][0]/trace_data_baseline[cv]["Inference"][0] for cv in cv_list]
        else:
            request_mean_improved_ratios = [trace_data_baseline[cv]["Request"][1]/trace_data_llumnix[cv]["Request"][1] for cv in cv_list]
            request_P99_improved_ratios = [trace_data_baseline[cv]["Request"][0]/trace_data_llumnix[cv]["Request"][0] for cv in cv_list]
            prefill_mean_improved_ratios = [trace_data_baseline[cv]["Prefill"][1]/trace_data_llumnix[cv]["Prefill"][1] for cv in cv_list]
            prefill_P99_improved_ratios = [trace_data_baseline[cv]["Prefill"][0]/trace_data_llumnix[cv]["Prefill"][0] for cv in cv_list]
            decode_mean_improved_ratios = [trace_data_baseline[cv]["Decode"][1]/trace_data_llumnix[cv]["Decode"][1] for cv in cv_list]
            decode_P99_improved_ratios = [trace_data_baseline[cv]["Decode"][0]/trace_data_llumnix[cv]["Decode"][0] for cv in cv_list]
            inference_mean_improved_ratios = [trace_data_baseline[cv]["Inference"][0]/trace_data_llumnix[cv]["Inference"][0] for cv in cv_list]
    handles, labels = fig.axes[0].get_legend_handles_labels()

    new_handles = [handles[0], handles[1]]
    new_labels = [labels[0], labels[1]]

    fontsize=15
    fig.legend(new_handles, new_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.93), fontsize=fontsize)

    fig.savefig("figure13.png", bbox_inches="tight")
    out_filename = "figure13.claim"

    with open(out_filename, 'w+') as out_file:
        out_file.write("Llumnix vs. Llumnix-base\n")
        out_file.write("Normal Reqeust:\n")
        out_file.write("Request Mean increase up to: {:.2f}x\n".format(max(request_mean_increase_ratios)))
        out_file.write("Request P99 increase up to: {:.2f}x\n".format(max(request_P99_increase_ratios)))
        out_file.write("Prefill Latency(s):\n")
        out_file.write("Prefill Mean increase up to: {:.2f}x\n".format(max(prefill_mean_increase_ratios)))
        out_file.write("Prefill P99 increase up to: {:.2f}x\n".format(max(prefill_P99_increase_ratios)))
        out_file.write("Decode Latency(s):\n")
        out_file.write("Decode Mean increase up to: {:.2f}x\n".format(max(decode_mean_increase_ratios)))
        out_file.write("Decode P99 increase up to: {:.2f}x\n".format(max(decode_P99_increase_ratios)))
        out_file.write("Decode Execution Time(s):\n")
        out_file.write("Decode Execution Time(s) increase up to: {:.2f}x\n".format(max(inference_mean_increase_ratios)))
        out_file.write("High Priority Reqeust:\n")
        out_file.write("Request Latency(s):\n")
        out_file.write("Request Mean outperforms up to: {:.2f}x\n".format(max(request_mean_improved_ratios)))
        out_file.write("Request P99 outperforms up to: {:.2f}x\n".format(max(request_P99_improved_ratios)))
        out_file.write("Prefill Latency(s):\n")
        out_file.write("Prefill Mean outperforms up to: {:.2f}x\n".format(max(prefill_mean_improved_ratios)))
        out_file.write("Prefill P99 outperforms up to: {:.2f}x\n".format(max(prefill_P99_improved_ratios)))
        out_file.write("Decode Latency(s):\n")
        out_file.write("Decode Mean outperforms up to: {:.2f}x\n".format(max(decode_mean_improved_ratios)))
        out_file.write("Decode P99 outperforms up to: {:.2f}x\n".format(max(decode_P99_improved_ratios)))
        out_file.write("Decode Execution Time(s):\n")
        out_file.write("Decode Execution Time(s) outperforms up to: {:.2f}x\n".format(max(inference_mean_improved_ratios)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', default='../log', type=str)
    args = parser.parse_args()

    plot_figure13_paper(args.log_path)