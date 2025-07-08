# from third_party.MetaService.src.python.metaservice_client import MetaServiceClient
import json


class QueryClient:
    def query_cache_locality(self, hash_key: str) -> float:
        raise NotImplementedError
    

# class MetaServiceQueryClient(QueryClient):
#     def __init__(self, config):
#         self.config = config
#         self.client = MetaServiceClient()
#         self.client.initialize(config)

#     def query_cache_locality(self, hash_key: str) -> list[str]:
#         results = self.client.zreadrange(hash_key, 0)
#         return [item[0] for item in results]
    

class MockQueryClient(QueryClient):
    def __init__(self):
        # Simple mock data, key is hash_key, value is instance_id list
        self.mock_data = {
            "hash_key_1": ["instance_1", "instance_2", "instance_3"],
            "hash_key_2": ["instance_2", "instance_4"],
            "hash_key_3": ["instance_1", "instance_5"],
        }

    def query_cache_locality(self, hash_key: str) -> list[str]:
        # Return corresponding instance_id list, return empty list if not exists
        return self.mock_data.get(hash_key, [])



def build_meta_client_from_config(config_path) -> QueryClient:
    with open(config_path, 'r') as f:
        config = json.load(f)
    backend_type = config.get("metadata_backend", "mock").lower()

    if backend_type == "metaservice":
        return MetaServiceQueryClient(config_path)
    elif backend_type == "mock":
        return MockQueryClient()

    #  Future: support more backends
    raise ValueError(f"Unsupported metadata backend: {backend_type}")

