from ray.util.queue import Queue as RayQueue


class ServerInfo:
    def __init__(self,
                 server_id: str,
                 request_output_queue: RayQueue) -> None:
        self.server_id = server_id
        self.request_output_queue = request_output_queue
