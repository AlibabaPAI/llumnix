import logging
import os
import ray


class NodeFileHandler(logging.Handler):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        self.ensure_base_path_exists()

    def ensure_base_path_exists(self):
        if not os.path.exists(self.base_path):
            try:
                os.makedirs(self.base_path)
                print(f"Created log node path: {self.base_path}")
            except OSError as e:
                print(f"Error creating log node path {self.base_path}: {e}")

    def emit(self, record):
        node_id = ray.get_runtime_context().get_node_id()
        filename = os.path.join(self.base_path, f"{node_id}.log")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.format(record) + '\n')
