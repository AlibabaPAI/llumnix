import asyncio

async def main():
    print("Hello, World!")

loop = asyncio.get_running_loop()
loop.create_task(main())