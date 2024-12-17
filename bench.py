import asyncio
import os
import aiohttp
import json

# 定义请求的URL
URL = "http://172.23.75.204:8082/v1/chat/completions"

# 定义请求的JSON负载
PAYLOAD = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "write a quick sort in python"
        }
    ],
    "stream": "false",
    "presence_penalty": 1.1,
    "max_tokens": 100
}

def max_tokens():
    import random
    return random.randint(50, 200)

def gen_content():
    import random
    return f"what is the sum of {random.randint(1, 1000000)} and {random.randint(1, 1000000)}?"
# 定义每次请求的间隔（秒）
INTERVAL = 0.1

from loguru import logger
async def fetch(url, payload):
    try:
        payload["messages"][1]["content"] = gen_content()
        payload["max_tokens"] = max_tokens()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                resp_json = await response.json()
                logger.info(resp_json)
    except aiohttp.ClientError as e:
        print(f"请求失败: {e}")
        os._exit(1)
    except asyncio.TimeoutError:
        print("请求超时")
        os._exit(1)

async def main():
    while True:
        asyncio.create_task(fetch(URL, PAYLOAD))
        await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("脚本已手动停止")