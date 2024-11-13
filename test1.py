from pydantic import BaseModel
from typing import Any, Tuple, Dict

class req(BaseModel):
    id: int = -1

class ServerRequest(BaseModel):
    id: int = -1  # internal use only
    external_id: str = ''  # external request id
    input_text: req = None



class ServerRequestLlumnix(ServerRequest):
    llumnix_request_args: Tuple[Any, ...]


    @classmethod
    def from_server_request(cls, server_request: ServerRequest, *args):
        server_request_dict = server_request.model_dump()
        # 确保 input_text 被正确地保留为 req 实例
        server_request_dict['input_text'] = req.model_validate(server_request_dict['input_text'])
        return cls.model_construct(**server_request_dict, llumnix_request_args=args)

# 示例用法
server_request_instance = ServerRequest(id=123, external_id="example",input_text=req(id=2))
server_request_instance.test = 1
# llumnix_request_instance = ServerRequestLlumnix.from_server_request(server_request_instance, "extra_arg1", "extra_arg2")
# print(llumnix_request_instance.input_text.id)

# print(llumnix_request_instance)
# print(llumnix_request_instance.id)  # 输出: 123
# print(llumnix_request_instance.external_id)  # 输出: example
# print(llumnix_request_instance.llumnix_request_args)  # 输出: ('extra_arg1', 'extra_arg2')

# exclude_dict = {
# "llumnix_request_args": True
# }
# req_dict = llumnix_request_instance.model_dump(exclude=exclude_dict)
# req_dict = server_request_instance.model_dump()

# from dataclasses import dataclass
# @dataclass
# class GenerationGroupState:
#     """Information of generation that belong to the same prompt."""

#     request_group_id: int = -1
#     # the max length of generated + prompt tokens
#     length: int = 0


# class LlumnixRequest:
#     def __init__(self, request_id: int, expected_steps: int) -> None:
#         self.request_id = request_id

#         # strict pre-migration args
#         self.expected_steps = expected_steps

#         # migration args


# class GenerationGroupStateLlumnix(GenerationGroupState, LlumnixRequest):
#     def __init__(self, gen_group: GenerationGroupState, llumnix_request_args) -> None:
#         GenerationGroupState.__init__(self, **gen_group.__dict__)
#         LlumnixRequest.__init__(self, *llumnix_request_args)
#         #TODO[xinyi]: pagedreqstate prefill
#         self.is_prefill = True
#         self.is_finished = False
#         # The total chunk size (number of tokens) to process for next iteration.
#         # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
#         # chunked, it can be smaller than that.
#         self.token_chunk_size = 0
#         self._num_computed_tokens = 0

# gen_group = GenerationGroupState(request_group_id=123, length=50)

# # 创建一个 LlumnixRequest 参数
# llumnix_request_args = (456, 100)

# # 初始化 GenerationGroupStateLlumnix 实例
# instance = GenerationGroupStateLlumnix(gen_group, llumnix_request_args)

# print(instance.request_id)