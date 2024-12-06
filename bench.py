#!/usr/bin/env python

import asyncio
import json
import time
from typing import List

from websockets.sync.client import connect

from blade_llm.protocol import (
    GenerateRequest,
    GenerateStreamResponse,
    SamplingParams,
    StoppingCriteria,
)

port = 8081

finish = 0

async def hello(max_new_tokens, ignore_eos):
    headers = {
        # "Authorization": "<You may need this header for EAS."
    }
    url = f"ws://22.3.131.34:{port}/generate_stream"
    with connect(url, additional_headers=headers) as websocket:
        import random
        prompts = [f"what's {random.randint(a=0, b=1000000)} plus {random.randint(a=0, b=1000000)}?"]
        for p in prompts:
            print(f"Prompt : {p}")
            req = GenerateRequest(
                prompt=p,
                sampling_params=SamplingParams(
                    temperature=-0.9,
                    top_p=0.9,
                    top_k=0,
                ),
                stopping_criterial=StoppingCriteria(max_new_tokens=max_new_tokens, ignore_eos=ignore_eos),
            )
            websocket.send(req.model_dump_json())
            texts = []
            idx = 0
            global finish
            while True:
                await asyncio.sleep(0)
                msg = websocket.recv()
                resp = GenerateStreamResponse(**json.loads(msg))
                texts.extend([t.text for t in resp.tokens])
                idx += 1
                for t in resp.tokens:
                    print(t.text, end="")
                if resp.is_finished:
                    finish += 1
                    break
            print(len(texts), idx)
            print(f"{finish}, Generated text: {''.join(texts)}")
            print("-" * 40)


async def get_range(n):
    for i in range(n):
        yield i

async def main():
    tasks: List[asyncio.Task] = []
    num_requests = 500
    max_new_tokens = 20
    ignore_eos = False
    start = time.time()
    async for i in get_range(num_requests):
        await asyncio.sleep(0.001)
        task = asyncio.create_task(hello(max_new_tokens, ignore_eos))
        tasks.append(task)
    await asyncio.gather(*tasks)
    elapsed = time.time() - start
    output_tps = max_new_tokens * num_requests / elapsed
    print(f"Generate {output_tps} tokens/s")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="The port number to use")
args = parser.parse_args()

port = args.port

asyncio.run(main())
