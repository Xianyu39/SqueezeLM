import asyncio
import aiohttp
import json
from typing import Dict, Callable, List, Any, AsyncGenerator
from squeeze_lm.logger import init_logger


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)
    else:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)


logger = init_logger()

class InvalidResponseContent(Exception):
    """
    Raised when a response is successfully received (e.g. HTTP 200)
    but the content does not meet expected format or constraints.
    """
    def __init__(self, message: str, response: dict = None):
        self.message = message
        self.response = response
        super().__init__(message)

    def __str__(self):
        return f"InvalidResponseContent: {self.message}\nResponse: {self.response}"
    


class Inference:
    def __init__(self, base_url: str, api_key: str, concurrency_limit: int=40, retries: int = 5, wait_time_base: float = 3):
        self.base_url = base_url
        self.api_key = api_key
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.retries = retries
        self.wait_time_base = wait_time_base
        self.retryable_statuses = {429, 500, 502, 503, 504}
    
    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }
    

    async def _request_with_retries(
        self,
        method: str,
        endpoint: str,
        session: aiohttp.ClientSession,
        check_response: Callable = None,
        stream: bool = False,
        **kwargs
    ) -> Dict | aiohttp.ClientResponse:

        url = f"{self.base_url}{endpoint}"
        last_exception = None
        if kwargs.get("headers") is None:
            kwargs["headers"] = self._headers
            
        for attempt in range(self.retries):
            try:
                async with self.semaphore:
                    # 流式则直接返回response给调用者处理
                    if stream:
                        response = await session.request(method, url, **kwargs)
                        if response.status == 200:
                            return response
                        else:
                            text = await response.text()

                    async with session.request(method, url, **kwargs) as response:
                        if response.status == 200:
                            # 非流式则解码并且检查内容
                            response_json = await response.json()
                            if check_response is None or check_response(response_json):
                                return response_json
                            else:
                                raise InvalidResponseContent(
                                    message="Invalid response content",
                                    response=response_json
                                )
                        else:
                            text = await response.text()

                    # Retryable HTTP errors
                    if response.status in self.retryable_statuses:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=text,
                            headers=response.headers
                        )

                    # Non-retryable HTTP errors
                    response.raise_for_status()

            # 网络层的错误
            except aiohttp.ClientConnectorError as e:
                logger.warning(f"Connection error: {e}")
                last_exception = e
            except aiohttp.ClientOSError as e:
                logger.warning(f"Client OS error: {e}")
                last_exception = e
            except aiohttp.ServerTimeoutError as e:
                logger.warning(f"Server timeout: {e}")
                last_exception = e
            except asyncio.TimeoutError as e:
                logger.warning(f"Request timeout: {e}")
                last_exception = e

            # 应用层错误
            except aiohttp.ClientResponseError as e:
                if e.status in self.retryable_statuses:
                    logger.warning(f"Retryable HTTP error {e.status}: {e.message}")
                    last_exception = e
                else:
                    raise
            
            # 业务层错误
            except InvalidResponseContent as e:
                logger.warning(f"Invalid response content: {e}")
                last_exception = e

            # 兜底
            except aiohttp.ClientError as e:
                logger.warning(f"Unexpected aiohttp error: {e}")
                last_exception = e

            # 重试等待
            if attempt < self.retries:
                wait_time = 2 ** (attempt + 1 + self.wait_time_base)
                await asyncio.sleep(wait_time)
            else:
                logger.error("Max retries reached.")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Request failed without exception (unexpected).")


    async def ainference_stream(self, url, body, session:aiohttp.ClientSession, timeout, prefix: str = 'data: ', chunk_size:int=3):
        headers = self._headers.copy()
        headers["Accept"] = "text/event-stream"
        body["stream"] = True

        response = await self._request_with_retries(
            method='POST',
            endpoint=url,
            session=session,
            timeout=timeout,
            json=body,
            headers=headers,
            check_response=None,
            stream=True
        )
        
        try:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Stream API Error {response.status}: {text}")

            # 使用 aiohttp 这一层已经处理过一部分编码问题的 content
            text_buffer = ""
            async for line_bytes in response.content:
                # aiohttp 的 content iterator 默认按行分割 (readline)
                # 但为了安全起见，我们手动 decode 并忽略错误或使用 replace，或者确认源是标准 UTF-8
                line = line_bytes.decode('utf-8', errors='replace').strip()
                
                if not line or line.startswith(":"): # keep-alive comments
                    continue
                
                if line.startswith(prefix):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        # 假设这里也是 OpenAI 格式
                        content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if content:
                            text_buffer += content
                        for i in range(0, len(text_buffer), chunk_size):
                            if i + chunk_size <= len(text_buffer):
                                yield text_buffer[i:i+chunk_size]
                        text_buffer = text_buffer[(len(text_buffer)//chunk_size)*chunk_size:]

                    except json.JSONDecodeError:
                        continue
            yield text_buffer  # yield any remaining text

        finally:
            response.release()


    async def _process_stream(self, response: aiohttp.ClientResponse, prefix: str = "data: ", encoding:str='utf-8') -> AsyncGenerator[Dict, None]:
        buffer = b""
        async for chunk in response.content.iter_any():
            # print(chunk.decode(encoding))
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode(encoding).strip()
                if line.startswith(prefix):
                    line = line.removeprefix(prefix)

                if line == "[DONE]":
                    return
                try:
                    line = json.loads(line)
                    yield line
                except json.JSONDecodeError:
                    pass


    async def ainference(
            self, 
            url: str, 
            body: dict, 
            session: aiohttp.ClientSession,
            check_response: Callable = None,
            timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=180), 
        ) -> Dict:
        response = await self._request_with_retries(
            method="POST", 
            endpoint=url, 
            json=body, 
            session=session, 
            check_response=check_response,
            timeout=timeout
        )
        return response


    def inference(
        self, url: str, body: dict, timeout: float = 180
    ) -> Dict:
        async def _ainference_wrapper():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
                return await self.ainference(
                    url=url,
                    body=body,
                    session=session,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                )
        return run_async(_ainference_wrapper())