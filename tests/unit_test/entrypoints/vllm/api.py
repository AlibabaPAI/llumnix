from fastapi.responses import JSONResponse, Response
import ray

import llumnix.entrypoints.vllm.api_server

llumnix_client = llumnix.entrypoints.vllm.api_server.llumnix_client
manager = None
instance = None
app = llumnix.entrypoints.vllm.api_server.app


@app.get("/stats")
def stats() -> Response:
    """Get the statistics of the engine."""
    return JSONResponse(ray.get(instance.testing_stats.remote()))
