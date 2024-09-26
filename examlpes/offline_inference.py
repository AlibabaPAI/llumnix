from typing import List
import os
import asyncio

import ray

from llumnix import launch_ray_cluster, connect_to_ray_cluster, init_manager
from llumnix import (ManagerArgs, InstanceArgs, EngineArgs, Manager,
                     Llumlet, ServerInfo, QueueType, BackendType,
                     SamplingParams)
from llumnix.utils import random_uuid
from llumnix.queue.ray_queue_server import RayQueueServer

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
instance_args = InstanceArgs()
engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True,
                         trust_remote_code=True, max_model_len=370)

# Create a manager. If the manager is created first, and then the instances are created.
manager: Manager = init_manager(manager_args)
ray.get(manager.is_ready.remote())

# Create instances.
instance_ids: List[str] = None
instances: List[Llumlet] = None
instance_ids, instances = ray.get(manager.init_instances.remote(
    QueueType("rayqueue"), BackendType.VLLM, instance_args, engine_args))

# The requests‘ outputs will be put to the request_output_queue no matter which instance it's running in.
server_id = random_uuid()
request_output_queue = RayQueueServer()
server_info = ServerInfo(server_id, QueueType("rayqueue"), request_output_queue, None, None)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
async def background_process_outputs(num_tasks):
    finish_task = 0
    while finish_task != num_tasks:
        request_outputs = await request_output_queue.get()
        for request_output in request_outputs:
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
        request_id = random_uuid()
        await manager.generate.remote(request_id=request_id,
                                      server_info=server_info,
                                      prompt=request,
                                      params=sampling_params,)

    await output_task

asyncio.run(main())

# Kill all actor, as detach actor will not be killed by ray.shutdown.
named_actors = ray.util.list_named_actors(True)
for actor in named_actors:
    try:
        actor_handle = ray.get_actor(actor['name'], namespace=actor['namespace'])
        ray.kill(actor_handle)
    except:
        continue
    
cleanup_ray_env_func()

# Shutdown ray cluster.
ray.shutdown()
