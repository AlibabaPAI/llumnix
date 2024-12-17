import asyncio
import time
import uvicorn
from uvicorn import Config, Server
import argparse
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from contextlib import asynccontextmanager
import ray
from ray import serve

from llumnix.queue.zmq_server import ZmqServer
from llumnix.queue.zmq_client import ZmqClient
from llumnix.queue.utils import get_open_zmq_ipc_path
from llumnix.utils import random_uuid
from llumnix.server_info import ServerInfo

from llumnix.queue.ray_queue_server import RayQueueServer


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # @@@
    # loop = asyncio.get_event_loop()
    # loop.create_task(request_output_queue_server.run_server_loop())
    asyncio.create_task(request_output_queue_server.run_server_loop())
    yield
    # @@@
    request_output_queue_server.cleanup()

app = FastAPI(lifespan=lifespan)
# @@@
request_output_queue = RayQueueServer()
request_output_queue_server = None


@app.get("/is_ready")
async def is_ready() -> bool:
    return True

@app.post("/generate")
async def generate(request: Request) -> Response:
    ret = {"text": ""}
    return JSONResponse(ret)

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/generate_stream")
async def generate_stream(request: Request) -> StreamingResponse:
    async def number_generator():
        for i in range(10):
            t = time.time()
            yield f"Number: {i}, Time: {t}; "
            await asyncio.sleep(0.5)
    return StreamingResponse(number_generator(), media_type="text/plain")


class FastAPIServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        rpc_path = get_open_zmq_ipc_path(self.host, 8002)
        global request_output_queue_server
        request_output_queue_server = ZmqServer(rpc_path)
        # loop = asyncio.get_event_loop()
        # loop.create_task(request_output_queue_server.run_server_loop())

    def run(self):
        uvicorn.run(app, host=self.host, port=self.port)
        # rpc_path = get_open_zmq_ipc_path(self.host, 8002)
        # request_output_queue_server = ZmqServer(rpc_path)
        # loop = asyncio.get_event_loop()
        # loop.create_task(request_output_queue_server.run_server_loop())
        # config = Config(app=app, loop=loop, host=self.host, port=self.port)
        # server = Server(config)
        # loop.run_until_complete(server.serve())

    @classmethod
    def from_args(cls, host: str, port: int):
        fastapi_server_class = ray.remote(num_cpus=1, name="entrypoints")(cls)
        fastapi_server = fastapi_server_class.remote(host, port)

        return fastapi_server

async def wait_request_output_queue_server_ready(request_output_queue_client: ZmqClient,
                                                 server_info: ServerInfo):
    time.sleep(5)
    await request_output_queue_client.wait_for_server_rpc(server_info)
    # request_output_queue_server.cleanup()
    print("Request output queue server is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='172.23.75.202')
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    ray.init(namespace="llumnix")

    # rpc_path = get_open_zmq_ipc_path(args.host, 8002)
    # request_output_queue_server = ZmqServer(rpc_path)
    request_output_queue_client = ZmqClient()
    server_id = random_uuid()
    server_info = ServerInfo(server_id, 'zmq', None, args.host, 8002)

    fastapi_server = FastAPIServer.from_args(args.host, args.port)
    fastapi_server.run.remote()
    
    time.sleep(5)

    ip_address = f"{args.host}:{args.port}"
    api_list = [
        "is_ready",
        "generate",
        "generate_stream",
        "health",
    ]
    for api in api_list:
        try:
            url = f"http://{ip_address}/{api}"
            if api in ["is_ready", "health"]:
                response = requests.get(url)
            else:
                response = requests.post(url)
            response.raise_for_status()
            print(f"api: {api}, response: {response}, response.text: {response.text}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")

    asyncio.run(wait_request_output_queue_server_ready(request_output_queue_client, server_info))
