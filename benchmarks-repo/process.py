import sys


def calculate_slo(all_e2e_latencies, all_inference_latencies, all_response_lens):
    slos = [all_e2e_latencies[i] / all_inference_latencies[i] for i in range(len(all_e2e_latencies))]
    for i in range(len(slos)):
        slo = slos[i]
        if slo > 5:
            print(f"idx: {i}")
            print(f"slo: {slo}")
            print(f"response_len: {all_response_lens[i]}")
            print(f"e2e_latency: {all_e2e_latencies[i]}")
            print(f"inference_latency: {all_inference_latencies[i]}")
    return slos

def read_data_from_log_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("all_e2e_latencies=["):
                latency_str = line[len("all_e2e_latencies=["):].strip()[:-1]
                all_e2e_latencies = [float(latency) for latency in latency_str.split(",")]
            elif line.startswith("all_inference_latencies=["):
                latency_str = line[len("all_inference_latencies=["):].strip()[:-1]
                all_inference_latencies = [float(latency) for latency in latency_str.split(",")]
            elif line.startswith("all_waiting_latencies=["):
                latency_str = line[len("all_waiting_latencies=["):].strip()[:-1]
                all_waiting_latencies = [float(latency) for latency in latency_str.split(",")]
            elif line.startswith("all_total_tokens=["):
                len_str = line[len("all_total_tokens=["):].strip()[:-1]
                all_total_tokens = [int(len) for len in len_str.split(",")]
            elif line.startswith("all_prompt_lens=["):
                len_str = line[len("all_prompt_lens=["):].strip()[:-1]
                all_prompt_lens = [int(len) for len in len_str.split(",")]
            elif line.startswith("all_response_lens=["):
                len_str = line[len("all_response_lens=["):].strip()[:-1]
                all_response_lens = [int(len) for len in len_str.split(",")]
    
    return all_e2e_latencies, all_inference_latencies, all_waiting_latencies, all_total_tokens, all_prompt_lens, all_response_lens


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python script.py log_file')
        sys.exit(1)
    
    log_file = sys.argv[1]

    all_e2e_latencies, all_inference_latencies, all_waiting_latencies, all_total_tokens, all_prompt_lens, all_response_lens = read_data_from_log_file(log_file)
    
    slos = calculate_slo(all_e2e_latencies, all_inference_latencies, all_response_lens)
    # print(slos)