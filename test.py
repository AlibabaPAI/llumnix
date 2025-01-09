import ray

from llumnix.utils import get_actor_data_from_ray_internal_kv, put_actor_data_to_ray_internal_kv


@ray.remote(num_cpus=1)
class Manager:
    def __init__(self):
        value = get_actor_data_from_ray_internal_kv("manager", "port_offset")
        self.port_offset = 0 if value is None else int(value)
        print("[__init__] self.port_offset: {}".format(self.port_offset))
    
    def put(self):
        self.port_offset = 2
        put_actor_data_to_ray_internal_kv("manager", "port_offset", self.port_offset)
        print("[put] self.port_offset: {}".format(self.port_offset))

manager = Manager.remote()
ray.get(manager.put.remote())
