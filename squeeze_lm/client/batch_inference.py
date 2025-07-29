from typing import List
from math import ceil
from tqdm.asyncio import trange
from squeeze_lm.client.inference import Inference
from typing import Dict
import aiohttp, aiofiles, json, asyncio


def batch_inference(
    lines: List,
    output_file: str,
    batch_size: int,
    base_url: str,
    api_key: str,
    verbose=False,
    write_mode='w',
    timeout: float = 180,
    rate_limit: int = 10,
    time_window: float = 5.0,
    wait_time_base: float = 1,
    retries: int = 5,
    check_response=lambda x: True
) -> None:

    async def run_all_tasks():
        async with aiofiles.open(output_file, write_mode, encoding='utf-8') as f:
            inf = Inference(base_url=base_url, api_key=api_key, rate_limit=rate_limit, time_window=time_window, retries=retries, wait_time_base=wait_time_base, check_response=check_response)
            
            if verbose:
                print(f"Handle {len(lines)} request(s), split into {ceil(len(lines)/batch_size)} batch(es) of size {batch_size}.")
            
            async with aiohttp.ClientSession() as session:
                iterator = (
                    trange(0, len(lines), batch_size, desc="Processing batch(es)", total=ceil(len(lines) / batch_size))
                    if verbose else range(0, len(lines), batch_size)
                )
                client_timeout = aiohttp.ClientTimeout(total=timeout)

                for i in iterator:
                    batch = lines[i:i + batch_size]
                    batch = [json.loads(line) if isinstance(line, str) else line for line in batch]
                    
                    tasks = [
                        inf.ainference(
                            line['method'], url=line['url'], body=line['body'], session=session, custom_id=line['custom_id'], timeout=client_timeout
                        ) for line in batch
                    ]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)

                    for response, line in zip(responses, batch):
                        nl = {
                            'custom_id': line['custom_id'],
                            'response': None
                        }
                        if isinstance(response, Exception):
                            nl['response'] = {
                                'error': str(response),
                                'type': type(response).__name__
                            }
                        else:
                            nl['response'] = response

                        await f.write(json.dumps(nl, ensure_ascii=False)+'\n')
                        await f.flush()

    asyncio.run(run_all_tasks())
        
            
def generate_prompt_line(custom_id: str, prompt: str | List[Dict[str, str]], method="POST", url="/v1/chat/completions", model: str=None, temperature: float=None,  max_tokens: int=4096, n_predict: int=2048) -> Dict:
    if isinstance(prompt, str):
        return {
            'custom_id': custom_id,
            "method":method,
            "url":url,
            "body":{
                "model":LLM_MODEL_NAME if model is None else model,
                "messages":[
                    {"role":"user", "content":prompt}
                ],
                "stream":False,
                "temperature":LLM_TEMPERATURE if temperature is None else temperature,
                "max_tokens":max_tokens,
                "n_predict":n_predict
            }
        }
    elif isinstance(prompt, list):
        return {
            'custom_id': custom_id,
            "method":method,
            "url":url,
            "body":{
                "model":LLM_MODEL_NAME if model is None else model,
                "messages":prompt,
                "stream":False,
                "temperature":LLM_TEMPERATURE if temperature is None else temperature,
            }
        }
    else:
        raise ValueError("Prompt must be a string or a list of dictionaries.")


def generate_batch_prompt_lines(prompts: List[str | List[Dict[str, str]]], custom_ids: List[str] = None, method="POST", url="/v1/chat/completions", model: str=None, temperature: float=None, max_tokens: int=4096, n_predict: int=2048) -> List[Dict]:
    """
    Generate a list of prompt lines for batch processing.
    Args:
        prompts (List[str]): A list of prompts to generate lines for.
        custom_ids (List[str]): A list of custom IDs corresponding to each prompt.
        method (str): The HTTP method to use for the request.
        url (str): The URL endpoint for the request.
        model (str): The model name to use for the request.
        temperature (float): The temperature setting for the model.
    Returns:
        List[Dict]: A list of dictionaries representing the prompt lines.
    """
    if custom_ids is None:
        custom_ids = [f"{i}" for i in range(len(prompts))]
    return [generate_prompt_line(custom_id, prompt, method, url, model, temperature, max_tokens=max_tokens, n_predict=n_predict) for custom_id, prompt in zip(custom_ids, prompts)]