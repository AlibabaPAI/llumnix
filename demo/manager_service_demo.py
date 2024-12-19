import asyncio
from queue import Queue
import time
from typing import Dict
from functools import partial
import uvicorn
from fastapi import FastAPI
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.state import list_actors
from ray.util.queue import Queue as RayQueue

from llumnix.utils import random_uuid

WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS = 1.0
AUTO_DEPLOYMENT_INTERVAL_SECONDS = 10.0

app = FastAPI()

def get_server_name(instance_id: str) -> str:
    return f"server_{instance_id}"

def get_instance_name(instance_id: str) -> str:
    return f"instance_{instance_id}"

def initialize_placement_group(lifetime: str = None) -> PlacementGroup:
    placement_group_specs = ([{"CPU": 1}, {"CPU": 1, "GPU": 4}])
    placement_group = ray.util.placement_group(
        placement_group_specs, "STRICT_PACK", lifetime=lifetime)
    return placement_group


class FastAPIServer:
    def __init__(self, instance_id: str, host: str, port: int):
        self.host = host
        self.port = port
        self.server_name = get_server_name(instance_id)
        print("FastAPIServer created")

    def run(self) -> None:
        uvicorn.run(app, host=self.host, port=self.port)

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  host: str,
                  port: int,
                  placement_group: PlacementGroup,
                  lifetime: str = None):
        server_name = get_server_name(instance_id)
        fastapi_server_class = ray.remote(num_cpus=1,
                                          name=server_name,
                                          namespace="llumnix",
                                          lifetime=lifetime)(cls).options(
                                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                                    placement_group=placement_group,
                                                    placement_group_bundle_index=0,
                                                )
                                            )
        fastapi_server = fastapi_server_class.remote(instance_id, host, port)
        return fastapi_server


class Llumlet:
    def __init__(self, instance_id: str):
        self.instance_name = get_instance_name(instance_id)
        print("Llumlet created")

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  placement_group: PlacementGroup,
                  lifetime: str = None):
        instance_name = get_instance_name(instance_id)
        llumlet_class = ray.remote(num_cpus=1,
                                   num_gpus=4,
                                   name=instance_name,
                                   namespace="llumnix",
                                   lifetime=lifetime)(cls).options(
                                       scheduling_strategy=PlacementGroupSchedulingStrategy(
                                                    placement_group=placement_group,
                                                    placement_group_bundle_index=1,
                                                )
                                   )
        llumlet = llumlet_class.remote(instance_id)
        return llumlet
    
    def ready(self) -> bool:
        return True


class LLMEngineManager:
    def __init__(self):
        print("create LLMEngineManager")
        self.host = "localhost"
        self.port = 8000
        self.last_pending_pg: PlacementGroup = None
        self.pgs: Dict[str, PlacementGroup] = {}
        self.servers: Dict[str, FastAPIServer] = {}
        self.instances: Dict[str, Llumlet] = {}
        asyncio.create_task(self._auto_scale_up_loop())
        asyncio.create_task(self._auto_scale_down_loop())
        print("LLMEngineManager created")
    
    async def _auto_scale_down_loop(self) -> None:
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if isinstance(ret, ray.exceptions.RayActorError):
                print(f"instance {instance_id} died, scale down")
                self.scale_down(instance_id)

        for instance_id, instance in self.instances.items():
            task = asyncio.gather(instance.ready.remote(), return_exceptions=True)
            task.add_done_callback(partial(instance_ready_callback, instance_id))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

        await asyncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)

    async def _auto_scale_up_loop(self) -> None:
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, ray.exceptions.RayActorError):
                print(f"instance {instance_id} ready, scale up")
                self.scale_up(instance_id, new_pgs[instance_id], new_servers[instance_id], new_instances[instance_id])
                new_servers[instance_id].run.remote()
            else:
                print(f"instance {instace_id} died, abort")
                self.remove_placement_group(new_pgs[instance_id], instance_id)
                self.kill_server(new_servers[instance_id], instance_id)

        while True:
            # 1. Get new placement group continuously until wait placement group ready timeout.
            new_pg_queue = Queue()
            while True:
                new_pg = initialize_placement_group(lifetime="detached") if not self.last_pending_pg else self.last_pending_pg
                try:
                    await asyncio.wait_for(new_pg.ready(), timeout=WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS)
                    print("initialize new process group")
                    new_pg_queue.put(new_pg)
                except asyncio.TimeoutError:
                    print("new placement group ready timeout")
                    self.last_pending_pg = new_pg
                    break
            print("Get new placement group done")

            # 2. Deploy 1 server and 1 instance to 1 placement group (for all new placement groups).
            new_pgs: Dict[str, PlacementGroup] = {}
            new_servers: Dict[str, FastAPIServer] = {}
            new_instances: Dict[str, Llumlet] = {}
            while not new_pg_queue.empty():
                instance_id = random_uuid()
                new_pg = new_pg_queue.get()
                new_pgs[instance_id] = new_pg
                new_servers[instance_id] = FastAPIServer.from_args(instance_id, self.host, self.port, new_pg, lifetime="detached")
                new_instances[instance_id] = Llumlet.from_args(instance_id, new_pg, lifetime="detached")
                await new_instances[instance_id].ready.remote()
            print("Deploy server and instance done")

            # 3. Wait for all instances ready. (With the assumption that once instance ready, server is ready too.)
            tasks = []
            for instance_id, instance in new_instances.items():
                task = asyncio.gather(instance.ready.remote(), return_exceptions=True)
                task.add_done_callback(partial(instance_ready_callback, instance_id))
                tasks.append(task)
            await asyncio.gather(*tasks, return_exceptions=True)
            print("Wait all instances ready done")

            await asyncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)
    
    def scale_up(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 server: FastAPIServer,
                 instance: Llumlet) -> None:
        print(f"add placement group {instance_id}")
        self.pgs[instance_id] = placement_group
        print(f"add server {instance_id}")
        self.servers[instance_id] = server
        print(f"add instance {instance_id}")
        self.instances[instance_id] = instance
        
    def scale_down(self, instance_id: str) -> None:
        self.kill_server(self.servers[instance_id], instance_id)
        self.remove_placement_group(self.pgs[instance_id], instance_id)
        if instance_id in self.pgs:
            print(f"del placement group {instance_id}")
            del self.pgs[instance_id]
        if instance_id in self.servers:
            print(f"del server {instance_id}")
            del self.servers[instance_id]
        if instance_id in self.instances:
            print(f"del instance {instance_id}")
            del self.instances[instance_id]

    def remove_placement_group(self,
                               placement_group: PlacementGroup,
                               instance_id: str) -> None:
        try:
            ray.util.remove_placement_group(placement_group)
        except Exception as e:
            print(f"try to remove placement group {instance_id}")
    
    def kill_server(self,
                    server: FastAPIServer,
                    instance_id: str) -> None:
        try:
            ray.kill(server)
        # Exception that killing a died actor.
        except Exception as e:
            print(f"try to kill api server {instance_id}")

    @classmethod
    def from_args(cls):
        engine_manager_class = ray.remote(num_cpus=1,
                                          max_restarts=-1,
                                          name="manager",
                                          namespace="llumnix",
                                          lifetime="detached")(cls)
        engine_manager = engine_manager_class.remote()
        return engine_manager


if __name__ == "__main__":
    ray.init()

    # magic actor
    request_output_queue = RayQueue(actor_options={
                                            "namespace": "llumnix",
                                            "name": "magic_queue"
                                        })
    engine_manager = LLMEngineManager.from_args()

    time.sleep(1000)
