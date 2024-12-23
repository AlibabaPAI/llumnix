import ray

from manager_service_demo import (initialize_placement_group,
                                  Llumlet,
                                  get_instance_name)

from llumnix.utils import random_uuid


def test_get_died_actor():
    placement_group = initialize_placement_group()
    instance_id = random_uuid()
    llumlet = Llumlet.from_args(instance_id, placement_group)
    ray.get(llumlet.ready.remote())
    ray.get_actor(get_instance_name(instance_id), namespace="llumnix")
    ray.kill(llumlet)
    try:
        ray.get_actor(get_instance_name(instance_id))
        print("Get died actor successfully")
    except ValueError:
        print("Get died actor failed")

if __name__ == "__main__":
    test_get_died_actor()
