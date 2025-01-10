from fastapi.responses import JSONResponse, Response
import ray

import llumnix.entrypoints.vllm.api_server

manager = None
llumnix_client = llumnix.entrypoints.vllm.api_server.llumnix_client
app = llumnix.entrypoints.vllm.api_server.app


@app.get("/stats")
def stats() -> Response:
    """Get the statistics of the engine."""
    return JSONResponse(ray.get(manager.testing_stats.remote()))
