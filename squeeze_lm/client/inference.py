import asyncio
import aiohttp
import collections
import httpx
import time
from typing import Dict
from squeeze_lm.logger import init_logger

logger = init_logger()

class Inference:
    def __init__(self, base_url: str, api_key: str, rate_limit: int = 10, time_window: float = 1.0, retries: int = 5, wait_time_base: float = 3):
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.retries = retries
        self.request_times = collections.deque([], self.rate_limit+1)
        self.wait_time_base = wait_time_base
        self.header = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
    

    async def ainference(self, method: str, url: str, body: dict, session: aiohttp.ClientSession) -> Dict:
        for retry in range(self.retries):
            try:
                await self.await_for_rate_limit()
                # print(f"{self.base_url}{url}")
                req = session.request(method, f"{self.base_url}{url}", json=body, headers=self.header)
                response = await req
                if response.status == 200:
                    return await response.json()
                elif response.status in {429, 500, 502, 503, 504}:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=await response.text(),
                        headers=response.headers
                    )
                else:
                    response.raise_for_status()
                
            except aiohttp.ClientResponseError as e:
                if retry < self.retries - 1 and e.status in {429, 500, 502, 503, 504}:
                    wait_time = 2 ** (retry + self.wait_time_base)
                    logger.warning(f"Retryable HTTP error {e.status}, retry {retry+1}, sleeping {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Non-retryable HTTP error {e.status}: {e.message}")
                    raise

            except aiohttp.ClientConnectionError as e:
                if retry < self.retries - 1:
                    wait_time = 2 ** (retry + self.wait_time_base)
                    logger.warning(f"Connection failed on retry {retry+1}, sleeping {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Connection failed permanently.")
                    raise

            except aiohttp.ClientError as e:
                if retry < self.retries - 1:
                    wait_time = 2 ** (retry + self.wait_time_base)
                    logger.warning(f"Client error on retry {retry+1}, sleeping {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Client error, giving up.")
                    raise
                

    def inference(self, method: str, url: str, body: dict, client: httpx.Client) -> Dict:
        for retry in range(self.retries):
            try:
                response = client.request(method, f"{self.base_url}{url}", json=body, headers=self.header)
                if response.status_code != 200:
                    raise Exception(f"Error: {response.status_code}")
                return response.json()
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except Exception as e:
                if retry < self.retries - 1:
                    wait_time = 2 ** (retry+self.wait_time_base)
                    logger.warning(f"Exception on retry {retry+1}, waiting {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(e)
                    return str(e)
                

    async def await_for_rate_limit(self):
        while True:
            now = time.time()
            while len(self.request_times) > 0 and self.request_times[0] < now - self.time_window:
                self.request_times.popleft()

            if len(self.request_times) < self.rate_limit:
                self.request_times.append(now)
                return
            
            wait = self.time_window - (now - self.request_times[0])
            if wait > 0:
                await asyncio.sleep(wait)