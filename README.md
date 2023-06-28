# Installation:

```
sudo su -
git clone https://github.com/kopyl/crossdot-yolo-simple.git && \
mv crossdot-yolo-simple/* . && \
rm -r crossdot-yolo-simple && \
wget https://bootstrap.pypa.io/get-pip.py && \
python3 get-pip.py && \
python3 -m pip install --ignore-installed flask==2.3.2 && \
python3 -m pip install psycopg2-binary==2.9.6 && \
python3 -m pip install onnxruntime==1.13.1 && \
python3 -m pip install opencv-python-headless==4.7.0.72 && \
python3 -m pip install socketify==0.0.20
```

# Test 150 async requests:

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

# Run app:

### Flask:

`sudo python3 flask-server.py`

### Sockerify single-process:

`sudo python3 socketify-single-process.py`

### Sockerify multi-process:

`sudo python3 socketify-multi-process.py`

### gunicorn:

`sudo gunicorn --workers=10 -b :80 flask-server:app`
