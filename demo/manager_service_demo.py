import asyncio
from queue import Queue
from typing import Dict
from functools import partial
import uvicorn
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from llumnix.utils import random_uuid

WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS = 1.0
AUTO_DEPLOYMENT_INTERVAL_SECONDS = 10.0


def get_entrypoints_name(instance_id: str) -> str:
    return f"entrypoints_{instance_id}"

def get_instance_name(instance_id: str) -> str:
    return f"instance_{instance_id}"

def initialize_placement_group(lifetime: str = None) -> PlacementGroup:
    # Any better expression?
    placement_group_specs = ([{"CPU": 1}, {"CPU": 1, "GPU": 4}])
    placement_group = ray.util.placement_group(
        placement_group_specs, "STRICT_PACK", lifetime=lifetime)
    return placement_group


class FastAPIServer:
    def __init__(self, instance_id: str, host: str, port: int):
        self.host = host
        self.port = port
        self.entrypoints_name = get_entrypoints_name(instance_id)

    def run(self) -> None:
        uvicorn.run(app, host=self.host, port=self.port)

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  host: str,
                  port: int,
                  placement_group: PlacementGroup,
                  lifetime: str = None):
        entrypoints_name = get_entrypoints_name(instance_id)
        fastapi_server_class = ray.remote(num_cpus=1,
                                          name=entrypoints_name,
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
        self.host = "localhost"
        self.port = 8000
        self.last_pending_pg = None
        self.pgs: Dict[str, PlacementGroup] = {}
        self.servers: Dict[str, FastAPIServer] = {}
        self.instances: Dict[str, Llumlet] = {}
        asyncio.create_task(self._auto_scale_up_loop())
        asyncio.create_task(self._auto_scale_down_loop())
    
    async def _auto_scale_down_loop(self) -> None:
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if isinstance(ret, ray.exceptions.RayActorError):
                self.scale_down(instance_id)

        for instance_id, instance in self.instances.items():
            task = asyncio.gather(instance.ready().remote(), return_exceptions=True)
            task.add_done_callback(partial(instance_ready_callback, instance_id))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _auto_scale_up_loop(self) -> None:
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, ray.exceptions.RayActorError):
                self.scale_up(instance_id, new_pgs[instance_id], new_servers[instance_id], new_instances[instance_id])
            else:
                self.remove_placement_group(new_pgs[instance_id])
                self.kill_server(new_servers[instance_id])

        while True:
            # 1. Get new placement group continuously until wait placement group ready timeouts.
            new_pg_queue = Queue()
            while True:
                new_pg = initialize_placement_group(lifetime="detached") if not self.last_pending_pg else self.last_pending_pg
                try:
                    await asyncio.wait_for(new_pg.ready(), timeout=WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS)
                    new_pg_queue.put(new_pg)
                except asyncio.TimeoutError:
                    self.last_pending_pg = new_pg
                    break
            # 2. Deploy 1 entrypoints and 1 instance to 1 placement group (for all new placement groups).
            new_pgs: Dict[str, PlacementGroup] = {}
            new_servers: Dict[str, FastAPIServer] = {}
            new_instances: Dict[str, Llumlet] = {}
            while not new_pg_queue.empty():
                instance_id = random_uuid()
                new_pg = new_pg_queue.get()
                new_pgs[instance_id] = new_pg
                new_servers[instance_id] = FastAPIServer.from_args(instance_id, self.host, self.port, new_pg, lifetime="detached")
                new_instances[instance_id] = Llumlet.from_args(instance_id, new_pg, lifetime="detached")
            # 3. Wait for all instances ready. (With the assumption that once instance ready, entrypoints is ready too.)
            tasks = []
            for instance_id, instance in new_instances.items():
                task = asyncio.gather(instance.ready().remote(), return_exceptions=True)
                task.add_done_callback(partial(instance_ready_callback, isntance_id))
                tasks.append(task)
            await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)
    
    def scale_up(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 server: FastAPIServer,
                 instance: Llumlet) -> None:
        self.pgs[instance_id] = placement_group
        self.servers[instance_id] = server
        self.instances[instance_id] = instance
        
    def scale_down(self, instance_id: str) -> None:
        self.kill_server(self.servers[instance_id])
        self.remove_placement_group(self.pgs[instance_id])
        if instance_id in self.pgs:
            del self.pgs[instance_id]
        if instance_id in self.servers:
            del self.servers[instance_id]
        if instance_id in self.instances:
            del self.instances[instance_id]

    def remove_placement_group(self,
                               placement_group: PlacementGroup) -> None:
        try:
            ray.util.remove_placement_group(placement_group)
        except Exception as e:
            print(f"try to remove placement group {instance_id}")
    
    def kill_server(self,
                    server: FastAPIServer) -> None:
        try:
            ray.kill(server[instance_id])
        # Exception that killing a died actor.
        except Exception as e:
            print(f"try to kill api server {instance_id}")

    @classmethod
    def from_args(cls):
        engine_manager_class = ray.remote(num_cpus=1,
                                          max_restarts=-1,
                                          name="manager",
                                          namespace="llumnix",
                                          lifetime="detached")(cls).options()
        engine_manager = engine_manager_class.remote()
        return engine_manager


if __name__ == "__main__":
    engine_manager = LLMEngineManager.from_args()
