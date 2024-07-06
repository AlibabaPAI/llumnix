# Fault Tolerance

In traditional multi-instance deployment, the multiple serving instances are independent from each other and also naturally fault-isolated. However, in a Llumnix cluster, fault-isolation becomes non-trivial since instances have coordination between each other. Therefore, failures of certain instance, if not handled properly, could impact the others and even the overall service availability.

Llumnix is designed to be fault-tolerant and provide high service availability. In a nutshell, Llumnix can tolerate failures of any components / instances, possibly with degraded serving throughput or scheduling quality during failover, but with no service downtimes. This also facilitates elasticity scaling, i.e., instances can be arbitrarily started or terminated, without impacting the service availability.

We describe the fault-tolerance mechanism and behavior for each component of Llumnix as follows.

# Ray Actors

We first introduce fault tolerance for Llumnix internal components, which are launched as Ray actors. Upon failures, these actors will be restarted by Ray. We describe the behavior during the failover periods below.

## API server

When an API server fails, all requests associated with it will be aborted. In particular, these requests can possibly run other instances not co-located with this server, and such requests will be automatically aborted, too.

## Scheduler

Currently, the scheduler is on the critical path of request serving in normal cases. If the scheduler fails, Llumnix falls back to a scheduler-bypassing mode with simple scheduling logic inside API servers.

## Instance (llumlet / backend)

When an instance fails, e.g., the llumlet or the backend engine, all requests running on this instance will be aborted. In particular, these requests can possibly be dispatched from other API servers not co-located with this instance, and such requests will be automatically aborted, too.

# Ray Head

Llumnix currently has limited support for Ray head fault-tolerance. That is, upon Ray head failures, Llumnix actors that are alive will keep running, thus service availability will not be impacted. However, Llumnix will be not able to create / restart new actors after a Ray head failure.

We will incorporate Ray's high-availability mechanism to enable Ray failover in the upcoming versions.