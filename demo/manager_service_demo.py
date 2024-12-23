import asyncio
import time
import traceback
from typing import Dict
from functools import partial
import uvicorn
from fastapi import FastAPI
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.queue import Queue as RayQueue
from ray.util import placement_group_table

from llumnix.utils import random_uuid
from llumnix.logger import init_logger

logger = init_logger(__name__)

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

def initialize_placement_group(instance_id: str = None, lifetime: str = None) -> PlacementGroup:
    placement_group_specs = ([{"CPU": 1}, {"CPU": 1, "GPU": 4}])
    if instance_id is None:
        instance_id = random_uuid()
    placement_group_name = get_placement_group_name(instance_id)
    placement_group = ray.util.placement_group(
        placement_group_specs, "STRICT_PACK", lifetime=lifetime, name=placement_group_name)
    return placement_group

def remove_placement_group(instance_id: str = None) -> bool:
    placement_group = ray.util.get_placement_group(get_placement_group_name(instance_id))
    if not placement_group:
        return False
    try:
        # asynchronous
        ray.util.remove_placement_group(placement_group)
        print(f"remove placement group {instance_id}")
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def kill_server(instance_id: str = None) -> bool:
    try:
        server = ray.get_actor(get_server_name(instance_id))
    except ValueError:
        return False
    try:
        ray.kill(server)
        print(f"kill server {instance_id}")
    # pylint: disable=broad-except
    except Exception:
        return False
    return True

def kill_instance(instance_id: str = None) -> bool:
    try:
        instance = ray.get_actor(get_instance_name(instance_id))
    except ValueError:
        return False
    try:
        ray.kill(instance)
        print(f"kill instance {instance_id}")
        return True
    # pylint: disable=broad-except
    except Exception:
        return False

def actor_exists(actor_name: str) -> bool:
    try:
        ray.get_actor(actor_name)
        return True
    except ValueError:
        return False


class FastAPIServer:
    def __init__(self, instance_id: str, host: str, port: int):
        self.host = host
        self.port = port
        self.server_name = get_server_name(instance_id)
        print("FastAPIServer created")
        self.run_loop_thread = threading.Thread(
            target=self._run_loop, args=(), daemon=True, name="run_loop"
        )

    def _run_loop(self):
        uvicorn.run(app, host=self.host, port=self.port)
    
    def run(self):
        self.run_loop_thread.start()

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

    def ready(self) -> bool:
        return True


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
        asyncio.create_task(self._check_deployment_correctness_loop())
        print("LLMEngineManager created")

    async def _auto_scale_down_loop(self) -> None:
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if isinstance(ret, ray.exceptions.RayActorError):
                print(f"server/instance {instance_id} died, scale down")
                self._scale_down(instance_id)

        while True:
            try:
                tasks = []
                for instance_id, instance in self.instances.items():
                    task = asyncio.gather(instance.ready.remote(), return_exceptions=True)
                    task.add_done_callback(partial(instance_ready_callback, instance_id))
                    tasks.append(task)
                await asyncio.gather(*tasks, return_exceptions=True)

                tasks = []
                for instance_id, server in self.servers.items():
                    task = asyncio.gather(server.ready.remote(), return_exceptions=True)
                    task.add_done_callback(partial(instance_ready_callback, instance_id))
                    tasks.append(task)
                await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("unexpected exception occurs: {}".format(e))
                logger.error("exception traceback: {}".format(traceback.format_exc()))

    async def _auto_scale_up_loop(self) -> None:
        while True:
            try:
                pending_pg_states = ray.util.state.list_placement_groups(filters=[("state", "=", "PENDING")])
                for pending_pg_state in pending_pg_states:
                    instance_id = pending_pg_state["name"].split("_")[-1]
                    self._scale_down(instance_id)
                instance_id = random_uuid()
                new_pg = initialize_placement_group(instance_id, lifetime="detached")
                try:
                    await asyncio.wait_for(new_pg.ready(), WAIT_PLACEMENT_GROUP_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    print("Get new placement group ready timeout")
                    ayncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)
                    continue
                print("Get new placement group ready done")
                self._initialize_server_and_instance(instance_id, new_pg)
                print("Deploy server and instance to new placement group done")
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("unexpected exception occurs: {}".format(e))
                logger.error("exception traceback: {}".format(traceback.format_exc()))

    async def _check_deployment_correctness_loop(self) -> None:
        while True:
            try:
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
                        if instance_id in curr_pgs:
                            curr_pgs.pop(instance_id)
                        if instance_id in curr_servers:
                            curr_servers.pop(instance_id)
                        if instance_id in curr_instances:
                            curr_instances.pop(instance_id)

                self.pgs = curr_pgs
                self.servers = curr_servers
                self.instance = curr_instances

                await asyncio.sleep(AUTO_DEPLOYMENT_INTERVAL_SECONDS)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("unexpected exception occurs: {}".format(e))
                logger.error("exception traceback: {}".format(traceback.format_exc()))

    def _initialize_server_and_instance(self, instance_id: str, placement_group: PlacementGroup):
        def instance_ready_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, ray.exceptions.RayActorError):
                print(f"instance {instance_id} ready, scale up")
                new_server.run.remote()
                self._scale_up(instance_id, placement_group, new_server, new_instance)
            else:
                print(f"instance {instance_id} died, abort scale up")
                self._scale_down(instance_id)

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
        kill_server(instance_id)
        kill_instance(instance_id)
        remove_placement_group(instance_id)
        if instance_id in self.pgs:
            print(f"del placement group {instance_id}")
            del self.pgs[instance_id]
        if instance_id in self.servers:
            print(f"del server {instance_id}")
            del self.servers[instance_id]
        if instance_id in self.instances:
            print(f"del instance {instance_id}")
            del self.instances[instance_id]

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
