from typing import List
import os
import uuid
import asyncio

import ray
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from llumnix import launch_ray_cluster, connect_to_ray_cluster, init_manager, init_llumlets
from llumnix import (SamplingParams, ServerInfo, EngineManagerArgs, LLMEngineManager, Llumlet,
                     EngineArgs, RequestOutput)
from llumnix.utils import random_uuid
from llumnix.rpc.queue_server import QueueServer
from llumnix.rpc.queue_client import QueueClient
from llumnix.rpc.utils import get_open_zmq_ipc_path
from llumnix.entrypoints.llumnix_utils import get_ip_address


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
ray_cluster_port=37000

# Note: launch_ray_cluster will stop current ray cluster first, then init a new one.
launch_ray_cluster(ray_cluster_port=ray_cluster_port)
connect_to_ray_cluster(port=ray_cluster_port)

# Set manager args and engine args.
manager_args = EngineManagerArgs()
engine_args = EngineArgs(model="facebook/opt-125m", worker_use_ray=True,
                         trust_remote_code=True, max_model_len=370)

# Create llumlets.
llumlet_ids: List[str] = None
llumlets: List[Llumlet] = None
llumlet_ids, llumlets = init_llumlets(manager_args, engine_args,
                                      node_id=ray.get_runtime_context().get_node_id())


# Create a manager. If the manager is created first, and then the llumlets are created, manager.scale_up
# need to be called to add the newly created llumlets to the management of the manager.
manager: LLMEngineManager = init_manager(manager_args)

# The requests‘ outputs will be put to the request_output_queue no matter which instance it's running in.
server_id = random_uuid()
ip = get_ip_address()
port = 1234
server_info = ServerInfo(server_id, ip, port)
rpc_path = get_open_zmq_ipc_path(server_info.request_output_queue_ip, server_info.request_output_queue_port)
request_output_queue = QueueServer(rpc_path)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
async def background_process_outputs(num_tasks):
    finish_task = 0
    while finish_task != num_tasks:
        request_output = await request_output_queue.get()
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
                                      sampling_params=sampling_params,)
    
    await output_task

asyncio.run(main())

# Kill all actor, as detach actor will not be killed by ray.shutdown.
named_actors = ray.util.list_named_actors(True)
for actor in named_actors:
    try:
        actor_handle = ray.get_actor(actor['name'], namespace=actor['namespace'])
    except:
        continue
    try:
        ray.kill(actor_handle)
    except:
        continue

# Shutdown ray cluster.
ray.shutdown()
