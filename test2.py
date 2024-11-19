import ray
from pydantic import BaseModel

# 定义 Pydantic 模型
class User(BaseModel):
    name: str
    age: int

# 定义 Ray 远程函数
@ray.remote
def process_user(user: User) -> str:
    return f"Processed user: {user.name}, age: {user.age}"

if __name__ == "__main__":
    # 初始化 Ray
    ray.init()

    # 创建 Pydantic 对象
    user = User(name="Alice", age=30)

    # 调用远程函数
    result = ray.get(process_user.remote(user))
    print(result)

    # 关闭 Ray
    ray.shutdown()
