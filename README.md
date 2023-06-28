### Test 150 async requests:

```
import aiohttp
import asyncio


async def make_request(session):
    url = "http://ec2-44-202-120-160.compute-1.amazonaws.com/v0/classify/image"
    payload = {
        "input": 'https://i0.wp.com/dictionaryblog.cambridge.org/wp-content/uploads/2023/05/rich-mom-energy-e1684141908206.jpg',
    }
    async with session.post(url, json=payload) as response:
        print(response)
        return await response.json()


async with aiohttp.ClientSession() as session:
    tasks = []

    for i in range(0, 150):
        tasks.append(asyncio.create_task(make_request(session)))

    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)
```
