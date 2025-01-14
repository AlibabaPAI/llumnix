import logging
import os
import ray


class NodeFileHandler(logging.Handler):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    def emit(self, record):
        node_id = ray.get_runtime_context().get_node_id()
        filename = os.path.join(self.base_path, f"{node_id}.log")
        with open(filename, 'w') as f:
            f.write(self.format(record) + '\n')
