import argparse
import time
import ray
from fastapi import FastAPI
import uvicorn

# @@@
# from llumnix.queue.ray_queue_server import RayQueueServer
from ray.util.queue import Queue as RayQueue

app = FastAPI()
# @@@
# request_output_queue = RayQueueServer()
request_output_queue = RayQueue()


class FastAPIServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def run(self):
        uvicorn.run(app, host=self.host, port=self.port)

    @classmethod
    def from_args(cls, host: str, port: int):
        fastapi_server_class = ray.remote(num_cpus=1, name="entrypoints")(cls)
        fastapi_server = fastapi_server_class.remote(host, port)

        return fastapi_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    ray.init(namespace="llumnix")
    
    fastapi_server = FastAPIServer.from_args(args.host, args.port)

    time.sleep(5)
