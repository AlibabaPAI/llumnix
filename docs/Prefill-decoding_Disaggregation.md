# Prefill-decoding Disaggregation
Existing large language model (LLM) serving systems that support prefill-decoding disaggregation typically compute the prefill and decoding phases on separate instances. For each request, following the prefill phase, the system migrates the generated key-value (KV) cache to the decoding instance and continues the computation. 


Llumnix can manage the scheduling of runtime requests across instances. It can perceive the execution process of requests and the state of the KV cache. Llumnix provides a series of scheduling semantics to support the initial dispatch and runtime rescheduling (request migration across instances). 


Therefore, Llumnix is inherently well-suited for the "special" cross-instance scheduling requirements in prefill-decoding disaggregation. Specifically, it limits the initial **dispatch only to the prefill instances** and the **migration between prefill and decoding instances**. 

## Supported Features
1. Requests can be **automatically migrated** from prefill instance to decoding instances.

2. Users can specify the number of prefill instances.

3. Llumnix supports both **one-to-many and many-to-one migrations** from prefill to decoding instances.

4. Decoding instances can migrate requests among themselves based on different scheduling strategies (e.g. load-balance).


## How to use
Llumnix only uses two arguments to enable prefill-decoding disaggregation for simplicity. 
- `--enable-pd-disagg True` is used to enable prefill-decoding disaggregation.
- `--num-available-dispatch-instances` is used to configure the number of prefill instances. 

Note that `num-available-dispatch-instances` < `initial_instance-num` especially when `--enable-scaling` is not set, as it determines the number of decoding instances.