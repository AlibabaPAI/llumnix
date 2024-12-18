import asyncio
import time
import uvicorn
import argparse
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from contextlib import asynccontextmanager
import ray

from llumnix.queue.ray_queue_server import RayQueueServer


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.create_task(request_output_queue.run_server_loop())
    yield
    request_output_queue.cleanup()

app = FastAPI(lifespan=lifespan)
request_output_queue = RayQueueServer()

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
    
    def run(self):
        uvicorn.run(app, host=self.host, port=self.port)

    @classmethod
    def from_args(cls, host: str, port: int):
        fastapi_server_class = ray.remote(num_cpus=1, name="entrypoints", namespace="llumnix", lifetime="detached")(cls).options()
        fastapi_server = fastapi_server_class.remote(host, port)

        return fastapi_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

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