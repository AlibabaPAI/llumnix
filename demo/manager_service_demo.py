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
from ray.util.queue import Queue as RayQueue

from llumnix.utils import random_uuid

PLACEMENT_GROUP_NAME_PREFIX = "pg_"
SERVER_NAME_PREFIX = "server_"
INSTANCE_NAME_PREFIX = "instance_"
WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS = 1.0
AUTO_DEPLOYMENT_INTERVAL_SECONDS = 1.0

app = FastAPI()

def get_placement_group_name(instance_id: str) -> str:
    return f"{PLACEMENT_GROUP_NAME_PREFIX}{instance_id}"
def get_server_name(instance_id: str) -> str:
    return f"{SERVER_NAME_PREFIX}{instance_id}"

def get_instance_name(instance_id: str) -> str:
    return f"{INSTANCE_NAME_PREFIX}{instance_id}"

def initialize_placement_group(lifetime: str = None, instance_id: str = None) -> PlacementGroup:
    placement_group_specs = ([{"CPU": 1}, {"CPU": 1, "GPU": 4}])
    placement_group_name = get_placement_group_name(instance_id)
    placement_group = ray.util.placement_group(
        placement_group_specs, "STRICT_PACK", lifetime=lifetime, name=placement_group_name)
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
        asyncio.create_task(self._check_deployment_loop())
        print("LLMEngineManager created")

    async def _auto_scale_down_loop(self) -> None:
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if isinstance(ret, ray.exceptions.RayActorError):
                print(f"instance {instance_id} died, scale down")
                self._scale_down(instance_id)

        while True:
            tasks = []
            for instance_id, instance in self.instances.items():
                task = asyncio.gather(instance.ready.remote(), return_exceptions=True)
                task.add_done_callback(partial(instance_ready_callback, instance_id))
                tasks.append(task)
            await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)

    async def _auto_scale_up_loop_removed(self) -> None:
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, ray.exceptions.RayActorError):
                print(f"instance {instance_id} ready, scale up")
                self._scale_up(instance_id, new_pgs[instance_id], new_servers[instance_id], new_instances[instance_id])
                new_servers[instance_id].run.remote()
            else:
                print(f"instance {instance_id} died, abort")
                self._remove_placement_group(new_pgs[instance_id], instance_id)
                self._kill_server(new_servers[instance_id], instance_id)

        while True:
            # 1. Get new placement group continuously until wait placement group ready timeout.
            new_pg_queue = Queue()
            while True:
                new_pg = initialize_placement_group(lifetime="detached") if not self.last_pending_pg else self.last_pending_pg
                try:
                    await asyncio.wait_for(new_pg.ready(), timeout=WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS)
                    new_pg_queue.put(new_pg)
                except asyncio.TimeoutError:
                    print("Wait new placement group ready timeout")
                    self.last_pending_pg = new_pg
                    break
            print("Get new placement group ready done")

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

    async def _auto_scale_up_loop(self) -> None:
        while True:
            pending_pg_states = ray.util.state.list_placement_groups(filters=[("state", "=", "PENDING")])
            while len(pending_pg_states) > 1:
                self._remove_placement_group(ray.util.get_placement_group(pending_pg_states.pop()["name"]))
            new_pg = initialize_placement_group(lifetime="detached") if len(pending_pg_states) == 0 else pending_pg_states[0]
            try:
                await asyncio.wait_for(new_pg.ready(), timeout=WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS)
                print("Get new placement group ready done")
                instance_id = random_uuid()
                self._initialize_server_and_instance(instance_id, new_pg)
                print("Deploy server and instance to placement group done")
            except asyncio.TimeoutError:
                print("Wait new placement group ready timeout")
                await asyncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)

    async def _check_deployment_loop(self) -> None:
        while True:
            curr_pgs: Dict[str, PlacementGroup] = {}
            curr_servers: Dict[str, PlacementGroup] = {}
            curr_instances: Dict[str, Llumlet] = {}
            created_pg_states = ray.util.state.list_placement_groups(filters=[("state", "=", "CREATED")])
            for created_pg_state in created_pg_states:
                instance_id = created_pg_state["name"].split("_")[-1]
                curr_pgs[instance_id] = ray.util.get_placement_group(created_pg_state["name"])
            alive_actor_states = ray.util.state.list_actors(filters=[("state", "=", "ALIVE")])
            for alive_actor_state in alive_actor_states:
                if alive_actor_state["name"].startswith(SERVER_NAME_PREFIX):
                    instance_id = alive_actor_state["name"].split("_")[-1]
                    curr_servers[instance_id] = ray.get_actor(alive_actor_state["name"])
                elif alive_actor_state["name"].startswith(INSTANCE_NAME_PREFIX):
                    instance_id = alive_actor_state["name"].split("_")[-1]
                    curr_instances[instance_id] = ray.get_actor(alive_actor_state["name"])

            assert len(curr_pgs) > max(len(curr_servers), len(curr_instances))

            for instance_id in curr_pgs:
                if instance_id not in curr_servers or instance_id not in curr_instances:
                    self._scale_down(instance_id)

            await asyncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)

    def _initialize_server_and_instance(self, instance_id: str, placement_group: PlacementGroup):
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, ray.exceptions.RayActorError):
                print(f"instance {instance_id} ready, scale up")
                self._scale_up(instance_id, placement_group, new_server, new_instance)
                new_server.run.remote()
            else:
                print(f"instance {instance_id} died, abort scale up")
                self._remove_placement_group(placement_group, instance_id)
                self._kill_server(new_server, instance_id)

        new_server = FastAPIServer.from_args(instance_id, self.host, self.port, placement_group, lifetime="detached")
        new_instance = Llumlet.from_args(instance_id, placement_group, lifetime="detached")
        instance_ready_task = asyncio.gather(new_instance.ready.remote(), return_exceptions=True)
        instance_ready_task.add_done_callback(partial(instance_ready_callback, instance_id))
        asyncio.create_task(instance_ready_task)

    def _scale_up(self,
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

    def _scale_down(self, instance_id: str) -> None:
        try:
            server = ray.get_actor(get_server_name(instance_id))
            self._kill_server(server, instance_id)
        except ValueError:
            pass
        try:
            instance = ray.get_actor(get_instance_name(instance_id))
            self._kill_instance(instance, instance_id)
        except ValueError:
            pass
        try:
            placement_group = ray.util.get_placement_group(get_placement_group_name(instance_id))
            self._remove_placement_group(placement_group, instance_id)
        except ValueError:
            pass
        if instance_id in self.pgs:
            print(f"del placement group {instance_id}")
            del self.pgs[instance_id]
        if instance_id in self.servers:
            print(f"del server {instance_id}")
            del self.servers[instance_id]
        if instance_id in self.instances:
            print(f"del instance {instance_id}")
            del self.instances[instance_id]

    def _remove_placement_group(self,
                                placement_group: PlacementGroup,
                                instance_id: str = None) -> None:
        try:
            ray.util.remove_placement_group(placement_group)
        # pylint: disable=broad-except
        except Exception:
            print(f"try to remove placement group {instance_id}")

    def _kill_server(self,
                     server: FastAPIServer,
                     instance_id: str = None) -> None:
        try:
            ray.kill(server)
        # Exception that killing a died actor.
        # pylint: disable=broad-except
        except Exception:
            print(f"try to kill api server {instance_id}")

    def _kill_instance(self,
                       instance: Llumlet,
                       instance_id: str = None) -> None:
        try:
            ray.kill(instance)
        # Exception that killing a died actor.
        # pylint: disable=broad-except
        except Exception:
            print(f"try to kill instance {instance_id}")

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
    manager = LLMEngineManager.from_args()

    time.sleep(1000)
