# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def parse_launch_mode(launch_mode: str):
    # 'eief' means that enable init instance by manager and enable fixed node init instance, and so on.
    if launch_mode == 'eief':
        disable_init_instance_by_manager = False
        disable_fixed_node_init_instance = False
    elif launch_mode == 'eidf':
        disable_init_instance_by_manager = False
        disable_fixed_node_init_instance = True
    elif launch_mode == 'dief':
        disable_init_instance_by_manager = True
        disable_fixed_node_init_instance = False
    else:
        disable_init_instance_by_manager = True
        disable_fixed_node_init_instance = True
    return disable_init_instance_by_manager, disable_fixed_node_init_instance

def generate_launch_command(result_filename: str = "", launch_ray_cluster: bool = True, HEAD_NODE_IP: str = "127.0.0.1",
                            ip: str = "127.0.0.1", port: int = 37000, instances_num = 1, dispatch_policy: str = "load",
                            migration_backend = "gloo", model = "facebook/opt-125m", max_model_len: int = 2048,
                            launch_mode: str = 'eief', log_instance_info: bool = False,
                            request_migration_policy: str = 'SR'):
    disable_init_instance_by_manager, disable_fixed_node_init_instance = parse_launch_mode(launch_mode)
    command = (
        f"RAY_DEDUP_LOGS=0 HEAD_NODE_IP={HEAD_NODE_IP} HEAD_NODE=1 "
        f"nohup python -u -m llumnix.entrypoints.vllm.api_server "
        f"--host {ip} "
        f"--port {port} "
        f"{'--disable-init-instance-by-manager ' if disable_init_instance_by_manager else ''}"
        f"{'--disable-fixed-node-init-instance ' if disable_fixed_node_init_instance else ''}"
        f"--initial-instances {instances_num} "
        f"{'--log-filename manager ' if log_instance_info else ''}"
        f"{'--log-instance-info ' if log_instance_info else ''}"
        f"--enable-migration "
        f"--model {model} "
        f"--engine-use-ray "
        f"--worker-use-ray "
        f"--max-model-len {max_model_len} "
        f"--dispatch-policy {dispatch_policy} "
        f"--trust-remote-code "
        f"--request-migration-policy {request_migration_policy} "
        f"--migration-backend {migration_backend} "
        f"--migration-buffer-blocks 32 "
        f"--migration-internal-buffer-num 2 "
        f"--tensor-parallel-size 1 "
        f"--request-output-queue-port {1234+port} "
        f"{'--launch-ray-cluster ' if launch_ray_cluster else ''}"
        f"{'> instance_'+result_filename if len(result_filename)> 0 else ''} 2>&1 &"
    )
    return command

def generate_bench_command(ip_ports: str, model: str, num_prompts: int, dataset_type: str, dataset_path: str,
                           qps: int, results_filename: str = "", query_distribution: str = "poisson",
                           coefficient_variation: float = 1.0, priority_ratio: float = 0.0):
    command = (
        f"python -u ./benchmark/benchmark_serving.py "
        f"--ip_ports {ip_ports} "
        f"--backend vLLM "
        f"--tokenizer {model} "
        f"--trust_remote_code "
        f"--log_filename bench_{ip_ports.split(':')[1]} "
        f"--random_prompt_count {num_prompts} "
        f"--dataset_type {dataset_type} "
        f"--dataset_path {dataset_path} "
        f"--qps {qps} "
        f"--distribution {query_distribution} "
        f"--coefficient_variation {coefficient_variation} "
        f"--priority_ratio {priority_ratio} "
        f"--log_latencies "
        f"--fail_on_response_failure "
        f"{'> bench_'+results_filename if len(results_filename)> 0 else ''}"
    )
    return command

def shutdown_llumnix_service():
    try:
        subprocess.run('pkill -f llumnix.entrypoints.vllm.api_server', shell=True, check=True)
        subprocess.run('pkill -f benchmark_serving.py', shell=True, check=True)
    # pylint: disable=broad-except
    except Exception:
        pass

def clear_ray_state():
    named_actors = ray.util.list_named_actors(True)
    for actor in named_actors:
        try:
            actor_handle = ray.get_actor(actor['name'], namespace=actor['namespace'])
        # pylint: disable=bare-except
        except:
            continue

        try:
            ray.kill(actor_handle)
        # pylint: disable=bare-except
        except:
            continue
    ray.shutdown()

def to_markdown_table(data):
    headers = data[0]
    rows = data[1:]

    col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

    header_row = " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(headers))
    separator_row = " | ".join('-' * col_widths[i] for i in range(len(headers)))

    data_rows = []
    for row in rows:
        data_row = " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row))
        data_rows.append(data_row)

    table = f"{header_row}\n{separator_row}\n" + "\n".join(data_rows) + "\n\n"
    return table

def backup_instance_log():
    dest_directory = os.path.expanduser(f'/mnt/error_log/{random_uuid()}/')
    os.makedirs(dest_directory, exist_ok=True)

    src_directory = os.getcwd()

    for filename in os.listdir(src_directory):
        if filename.startswith("instance_"):
            src_file_path = os.path.join(src_directory, filename)
            shutil.copy(src_file_path, dest_directory)
            print(f"Copied instance log: {src_file_path} to {dest_directory}")

        if filename.startswith("bench_"):
            src_file_path = os.path.join(src_directory, filename)
            shutil.copy(src_file_path, dest_directory)
            print(f"Copied bench log: {src_file_path} to {dest_directory}")
