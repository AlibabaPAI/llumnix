import time
import asyncio
import ray
from ray.util import placement_group_table
from ray.util.state import (list_actors,
                            list_placement_groups)

from manager_service_demo import (initialize_placement_group,
                                  Llumlet)


def test_actor_if_pg_died(life_time_pg, lifetime_llumlet):
    print(f"### placement group lifetime: {life_time_pg}, llumlet lifetime: {lifetime_llumlet}")
    print("### create placement group and llumlet")
    placement_group = initialize_placement_group(lifetime=life_time_pg)
    _ = Llumlet.from_args("0", placement_group, lifetime=lifetime_llumlet)
    print(f"placement group state: {placement_group_table(placement_group)}")
    print(f"llumlet state: {list_actors()}")
    print("### sleep 1s")
    time.sleep(5)
    print(f"llumlet state: {list_actors()}")
    print("### remove placement group")
    ray.util.remove_placement_group(placement_group)
    print(f"placement group state: {placement_group_table(placement_group)}")
    print(f"llumlet state: {list_actors()}")

def test_pg_if_actor_died(life_time_pg, lifetime_llumlet):
    print(f"### placement group lifetime: {life_time_pg}, llumlet lifetime: {lifetime_llumlet}")
    print("### create placement group and llumlet")
    placement_group = initialize_placement_group(lifetime=life_time_pg)
    llumlet = Llumlet.from_args("0", placement_group, lifetime=lifetime_llumlet)
    print(f"placement group state: {placement_group_table(placement_group)}")
    print(f"llumlet state: {list_actors()}")
    print("### sleep 5s")
    time.sleep(5)
    print(f"llumlet state: {list_actors()}")
    print("### kill llumlet")
    ray.kill(llumlet)
    print(f"placement group state: {placement_group_table(placement_group)}")
    print(f"llumlet state: {list_actors()}")
    print("### remove placement group")
    ray.util.remove_placement_group(placement_group)

def test_pending(life_time_pg, lifetime_llumlet):
    print(f"### placement group lifetime: {life_time_pg}, llumlet lifetime: {lifetime_llumlet}")
    print("### create placement group and llumlet")
    placement_group1 = initialize_placement_group(lifetime=life_time_pg)
    llumlet1 = Llumlet.from_args("0", placement_group1, lifetime=lifetime_llumlet)
    time.sleep(5)
    print(f"placement group 1 state: {placement_group_table(placement_group1)}")
    print(f"llumlet 1 state: {list_actors()}")
    print("### create placement group and llumlet")
    placement_group2 = initialize_placement_group(lifetime=life_time_pg)
    llumlet2 = Llumlet.from_args("1", placement_group2, lifetime=lifetime_llumlet)
    time.sleep(5)
    print(f"placement group 2 state: {placement_group_table(placement_group2)}")
    print(f"llumlet 2 state: {list_actors()}")
    print("### kill llumlet")
    ray.kill(llumlet1)
    time.sleep(5)
    print(f"placement group 2 state: {placement_group_table(placement_group2)}")
    print(f"llumlet 2 state: {list_actors()}")
    print("### remove placement group")
    ray.util.remove_placement_group(placement_group1)
    time.sleep(5)
    print(f"placement group 2 state: {placement_group_table(placement_group2)}")
    print(f"llumlet 2 state: {list_actors()}")
    ray.util.remove_placement_group(placement_group2)
    ray.kill(llumlet2)

async def test_pg_ready():
    placement_group1 = initialize_placement_group()
    try:
        await asyncio.wait_for(placement_group1.ready(), timeout=5.0)
        print("placement group 1 ready")
    except asyncio.TimeoutError:
        print("wait placement group 1 timeout")
    placement_group2 = initialize_placement_group()
    try:
        await asyncio.wait_for(placement_group2.ready(), timeout=5.0)
        print("placement group 2 ready")
    except asyncio.TimeoutError:
        print("wait placement group 2 timeout")
    ray.util.remove_placement_group(placement_group1)
    try:
        await asyncio.wait_for(placement_group2.ready(), timeout=5.0)
        print("placement group 2 ready")
    except asyncio.TimeoutError:
        print("wait placement group 2 timeout")
    placement_group3 = initialize_placement_group()
    ray.util.remove_placement_group(placement_group3)
    await placement_group3.ready()

def test_pg_api():
    placement_group1 = initialize_placement_group()
    placement_group2 = initialize_placement_group()
    time.sleep(3)
    all_pgs = list_placement_groups()
    print(f"all placement groups: {all_pgs}")
    all_pgs_detail = list_placement_groups(detail=True)
    print(f"all placement groups (detail): {all_pgs_detail}")
    pending_pgs = list_placement_groups(filters=[("state", "=", "PENDING")])
    print(f"pending placement groups: {pending_pgs}")
    created_pgs = list_placement_groups(filters=[("state", "=", "CREATED")])
    print(f"created placement groups: {created_pgs}")

    print(f"placement group 1 state: {placement_group_table(placement_group1)}")
    print(f"placement group 2 state: {placement_group_table(placement_group2)}")

if __name__ == "__main__":
    # test_actor_if_pg_died(life_time_pg=None, lifetime_llumlet=None)
    # test_actor_if_pg_died(life_time_pg=None, lifetime_llumlet="detached")
    # test_actor_if_pg_died(life_time_pg="detached", lifetime_llumlet=None)
    # test_actor_if_pg_died(life_time_pg=None, lifetime_llumlet="detached")

    # test_pg_if_actor_died(life_time_pg=None, lifetime_llumlet=None)
    # test_pg_if_actor_died(life_time_pg=None, lifetime_llumlet="detached")
    # test_pg_if_actor_died(life_time_pg="detached", lifetime_llumlet=None)
    # test_pg_if_actor_died(life_time_pg=None, lifetime_llumlet="detached")

    # test_pending(life_time_pg=None, lifetime_llumlet=None)

    asyncio.run(test_pg_ready())

    # test_pg_api()
