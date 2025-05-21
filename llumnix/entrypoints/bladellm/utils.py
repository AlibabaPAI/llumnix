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

import sys
import os

import ray

from llumnix.entrypoints.setup import connect_to_ray_cluster
from llumnix.logging.logger import init_logger

logger = init_logger('llumnix.entrypoints.bladellm.server')


@ray.remote(num_cpus=1)
def remote_launch_task(module, args, main_func):
    # set sys.argv
    sys.argv = [module] + args

    # avoid "RuntimeError: No CUDA GPUs are available"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main_func()

def get_current_node_resources():
    current_node_id = ray.get_runtime_context().get_node_id()
    resource = {}
    for node in ray.nodes():
        if node["NodeID"] == current_node_id:
            resource = node["Resources"]
            break
    return resource.get("GPU", 0), resource.get("CPU", 0)

def launch_job_on_gpu_node(module, main_func):
    # Assume that there is an existing ray cluster when using centralized deployment.
    connect_to_ray_cluster()

    num_gpus, num_cpus = get_current_node_resources()

    if num_gpus > 0:
        # current node has GPU resources
        logger.info("Launch on current node.")
        main_func()
    elif num_cpus == 0 and num_gpus == 0:
        # In some Ray clusters, there may exist a master node that has no CPU resources and no GPU resources (num_cpus=0, num_gpus=0),
        # which is used solely for submitting tasks.
        # Since importing bladellm requires GPU resources, server.py cannot be run on the master node.
        # In this case, use a Ray task with num_cpus=1 to launch bladellm on other worker nodes.
        logger.info("No GPU available on the current node. Launching on another node with GPU resources.")

        # get args
        original_argv = sys.argv[1:]
        ray.get(remote_launch_task.remote(module, original_argv, main_func))
    else:
        logger.info(
            "Current node has CPU resources but no GPU resources, not support now."
        )
