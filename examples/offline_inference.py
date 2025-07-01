from typing import List
import os
import time
import asyncio
import uuid

import ray

from vllm.outputs import RequestOutput
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams

from llumnix import (
    Scaler,
    Manager,
    get_manager_name,
    launch_ray_cluster,
    connect_to_ray_cluster,
    init_scaler,
    ManagerArgs,
    InstanceArgs,
    Llumlet,
    ServerInfo,
    QueueType,
    BackendType,
    LaunchArgs,
    EntrypointsArgs,
    LaunchMode,
    RayQueueServer,
    LlumnixRequestOuputVLLM,
)
from llumnix.entrypoints.vllm.arg_utils import VLLMEngineArgs

from tests.utils import try_convert_to_local_path
from tests.conftest import cleanup_ray_env_func


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Launch ray cluster
os.environ['HEAD_NODE'] = '1'
os.environ['HEAD_NODE_IP'] = '127.0.0.1'
ray_cluster_port=6379

# Note: launch_ray_cluster will stop current ray cluster first, then init a new one.
launch_ray_cluster(port=ray_cluster_port)
connect_to_ray_cluster(port=ray_cluster_port)

# Set manager args and engine args.
manager_args = ManagerArgs()
entrypoints_args = EntrypointsArgs()
instance_args = InstanceArgs()
engine_args = EngineArgs(model=try_convert_to_local_path("facebook/opt-125m"), download_dir="/mnt/model", worker_use_ray=True,
                         trust_remote_code=True, max_model_len=370, enforce_eager=True)
launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=BackendType.VLLM)
vllm_engine_args = VLLMEngineArgs(engine_args=engine_args)

# Create a manager. If the manager is created first, and then the instances are created.
scaler: Scaler = init_scaler(manager_args, instance_args, entrypoints_args, engine_args, launch_args)
ray.get(scaler.is_ready.remote())
manager: Manager = ray.get_actor(get_manager_name(), namespace='llumnix')

# Create instances and register to manager.
instance_ids: List[str] = None
instances: List[Llumlet] = None
node_id = ray.get_runtime_context().get_node_id()
instance_ids, instances = ray.get(scaler.init_instances.remote(
    QueueType("rayqueue"), instance_args, vllm_engine_args, node_id))
num_instance = 0
while num_instance == 0:
    num_instance = ray.get(manager.scale_up.remote([], [], []))
    time.sleep(1.0)

# The requests‘ outputs will be put to the request_output_queue no matter which instance it's running in.
server_id = str(uuid.uuid4().hex)
request_output_queue = RayQueueServer()
server_info = ServerInfo(server_id, QueueType("rayqueue"), request_output_queue, None, None)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
async def background_process_outputs(num_tasks):
    finish_task = 0
    while finish_task != num_tasks:
        request_outputs_engine: List[LlumnixRequestOuputVLLM] = await request_output_queue.get()
        for request_output_engine in request_outputs_engine:
            request_output: RequestOutput = request_output_engine.get_engine_output()
            if request_output.finished:
                finish_task += 1
                prompt = request_output.prompt
                generated_text = request_output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    request_output_queue.cleanup()

async def main():
    output_task = asyncio.create_task(background_process_outputs(len(prompts)))
    asyncio.create_task(request_output_queue.run_server_loop())

    for request in prompts:
        request_id = str(uuid.uuid4().hex)
        await manager.generate.remote(request_id=request_id,
                                      server_info=server_info,
                                      prompt=request,
                                      params=sampling_params)

    await output_task

asyncio.run(main())

cleanup_ray_env_func()

# Shutdown ray cluster.
ray.shutdown()
