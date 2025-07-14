# Python script
import argparse
import json
import multiprocessing as mp
import random
import sys
from typing import List

import requests

questions = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def _client_process_handle(num_requests: int, pid: int, host: str, port: int):
    random.seed(pid)
    for _ in range(num_requests):
        url = f"http://{host}:{port}/v1/chat/completions"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            # {'role': 'user', 'content': questions[random.randint(0, len(questions) - 1)]},
            {'role': 'user', 'content': questions[1]*100},
        ]
        req = {
            "messages": messages,
            "temperature": 0.0,
            "top_k": 1,
            "stream": "true",
            "ignore_eos": "false",
            "presence_penalty": 1.1,
            "repetition_penalty": 1.1,
        }
        response = requests.post(
            url,
            json=req,
            headers={"Content-Type": "application/json"},
            stream=True,
        )
        last_resp = ""
        tokens = ""
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False):
            msg = chunk.decode("utf-8")
            try:
                if msg.startswith('data'):
                    info = msg[6:]
                    if info == '[DONE]':
                        break
                    else:
                        last_resp = json.loads(info)
                        tokens += last_resp['choices'][0]['delta']['content']
                        print(last_resp['choices'][0]['delta']['content'], end='', flush=True)
            except Exception:
                print(msg)
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=1, help='number of clients')
    parser.add_argument('--num_requests', type=int, default=1, help='total number of requests for each client')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='host ip for prefill instance')
    parser.add_argument('--port', type=int, default=23333, help='web port for prefill instance service')
    args = parser.parse_args()

    procs: List[mp.Process] = []
    for pid in range(args.num_clients):
        proc = mp.Process(
            target=_client_process_handle, args=(args.num_requests, pid, args.host, args.port), daemon=True
        )
        proc.start()
        procs.append(proc)

    for _proc in procs:
        _proc.join()
        if _proc.exitcode != 0:
            sys.exit(_proc.exitcode)

    # print("done")