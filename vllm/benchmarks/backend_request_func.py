# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import sys
import math
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

import aiohttp
import asyncio
import huggingface_hub.constants
import requests
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.model_executor.model_loader.weight_utils import get_lock

from vllm.core.learn_conversation import combine_user_requests

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

chat_template = "{%- for message in messages %}" \
        "{%- if message['role'] == 'user' %}" \
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}" \
        "{%- elif message['role'] == 'assistant' %}" \
        "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}" \
        "{%- endif %}{%- endfor %}" \
        "{% if add_generation_prompt %}" \
        "{{ '<|im_start|>assistant\n' }}" \
        "{% endif %}"

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    tokenizer: Optional[AutoTokenizer] = None
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    user_latency: float = 0 # deprecated
    timestamp: float = 0
    next_timestamp: float = 0
    interval: float = 0
    time_limit: float = 10000
    id: int = 0
    conversation_id: int = -1
    turn_id: int = -1
    predicted_latency: float = 0
    exp_scale: float = 1
    checkpoint: str = ''
    use_oracle: float = 0
    use_token_id: int = 0
    use_lru: int = 0


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    server: str = "localhost"
    done_time: float = 0.0


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        params = {
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
            "truncate": request_func_input.prompt_len,
            # TGI does not accept ignore_eos flag.
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        # NOTE: Sometimes TGI returns a ping response without
                        # any data, we should skip it.
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = chunk_bytes.removeprefix("data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        if request_func_input.ignore_eos:
            payload["min_length"] = request_func_input.output_len
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    output.generated_text = parsed_resp["text"][0]
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "min_tokens": request_func_input.output_len+1, 
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


conversation_history = defaultdict(list)
conversation_last_time = {}
n_follow_up = 0
n_completed_req = 0
n_running_req = 0
start_time = time.time()

async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    global n_running_req
    global n_completed_req
    global n_follow_up
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    receive_time = time.time()
    if receive_time - start_time > request_func_input.time_limit:
        output = RequestFuncOutput()
        output.error = "timeout"
        return output
    # wait until all previous turns are finished
    cur_turn_id = len(conversation_history[request_func_input.conversation_id]) // 2
    if cur_turn_id > 0:
        while cur_turn_id != request_func_input.turn_id \
                and request_func_input.turn_id >= 0:
            await asyncio.sleep(3)
            if time.time() - start_time > request_func_input.time_limit:
                output = RequestFuncOutput()
                output.error = "timeout"
                return output
        scheduled_time = conversation_last_time[request_func_input.conversation_id] \
            + request_func_input.interval
        if scheduled_time - start_time + request_func_input.output_len / 30 > request_func_input.time_limit:
            output = RequestFuncOutput()
            output.error = "timeout"
            return output
        wait_for = scheduled_time - time.time()
        await asyncio.sleep(wait_for)
        request_func_input.timestamp += time.time() - receive_time
        request_func_input.next_timestamp += time.time() - receive_time \
            + request_func_input.output_len / 30 # assume token throughput is 30

    if request_func_input.turn_id >= 1:
        n_follow_up += 1
        if n_follow_up % 100 == 0:
            print("number of follow up requests: ", n_follow_up)

    def print_conversation_history(tokenizer, conversation_id):
        for message in conversation_history[conversation_id]:
            decoded_text = tokenizer.decode(message['token_ids'], skip_special_tokens=True)
            print(f"role: {message['role']}\n{message['content']}\n{decoded_text}\n")
    
    def get_messages(request_func_input):
        conversation_id = request_func_input.conversation_id
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
        user_message = {
                "role": "user",
                "content": request_func_input.prompt,
            }
        # append system prompt to the first user message
        if len(conversation_history[conversation_id]) == 0:
            token_ids = request_func_input.tokenizer.apply_chat_template([user_message], tokenize=True, 
                add_generation_prompt=True)
        else: # do not append system prompt to the other user messages
            token_ids = request_func_input.tokenizer.apply_chat_template([user_message], tokenize=True,
                chat_template=chat_template, add_generation_prompt=True)
        user_message["token_ids"] = token_ids
        conversation_history[conversation_id].append(user_message)
        # print_conversation_history(request_func_input.tokenizer, conversation_id)
        return conversation_history[conversation_id]

    def get_prompt_tokens(request_func_input):
        conversation_id = request_func_input.conversation_id
        prompt_tokens = []
        for message in conversation_history[conversation_id]:
            prompt_tokens += message['token_ids']
        # print('client token_ids: ', prompt_tokens)
        # decoded_text = request_func_input.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        # print('client prompt: ', decoded_text)
        return prompt_tokens
    
    def get_cache_hint(request_func_input):
        conversation_id = request_func_input.conversation_id
        turns = request_func_input.turn_id
        true_label = (request_func_input.next_timestamp < 1e8)
        prob_has_next = -1
        if request_func_input.use_lru:
            prob_has_next = 1
        # oracle
        if request_func_input.use_oracle > 0:
            prob_has_next = true_label
            if request_func_input.use_oracle == 3:
                prob_has_next *= random.random() * 0.9
                # math.exp(-request_func_input.next_timestamp / request_func_input.exp_scale)
            if request_func_input.use_oracle < 1:
                # with probablity 1 - request_func_input.use_oracle, flip
                error_rate = (1-request_func_input.use_oracle)
                uncertainty = (1-request_func_input.use_oracle)
                if random.random() < error_rate:
                    prob_has_next = (1 - prob_has_next) * (1- uncertainty * 2) + uncertainty
                else:
                    prob_has_next = prob_has_next * (1- uncertainty * 2) + uncertainty

        hint = {"turns": turns,
                "conversation_input": combine_user_requests(conversation_history[conversation_id]),
                "exp_scale": request_func_input.exp_scale,
                "true_tta": request_func_input.next_timestamp - request_func_input.timestamp,
                "id": conversation_id,
                "checkpoint": request_func_input.checkpoint,
        }
        if prob_has_next != -1:
            hint["prob_has_next"] = prob_has_next
        if request_func_input.use_oracle == 2:
            hint["next_timestamp"] = request_func_input.next_timestamp
        if request_func_input.use_lru:
            hint['use_lru'] = 1
        return hint
    
    def update_conversation(conversation_id, generated_text, generated_tokens):
        conversation_history[conversation_id].append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": generated_text}],
                "token_ids": generated_tokens
            }
        )
        conversation_last_time[conversation_id] = time.time()
    
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "messages": get_messages(request_func_input),
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "min_tokens": request_func_input.output_len, 
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            "cache_hint": get_cache_hint(request_func_input),
        }
        if request_func_input.use_token_id:
            payload["prompt_tokens"] = get_prompt_tokens(request_func_input)
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        if "prompt_tokens" in payload:
            output.prompt_len = len(payload["prompt_tokens"])

        generated_text = ""
        generated_tokens = []
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        n_running_req += 1
        #print(round(time.time()-start_time,2), request_func_input.timestamp, 
        #      request_func_input.conversation_id, n_completed_req)
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                if "token_ids" in choices[0]:
                                    token_id = choices[0]["token_ids"]
                                    generated_tokens += token_id
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    end_tokens = request_func_input.tokenizer.encode('<|im_end|>\n')
                    # print("generated: ", generated_text)
                    if len(generated_tokens) <= 1 or generated_tokens[-2] != end_tokens[-2]:
                        generated_tokens += end_tokens
                    update_conversation(request_func_input.conversation_id, generated_text, generated_tokens)
                    output.success = True
                    output.latency = most_recent_timestamp - st
                    #print(request_func_input.conversation_id, request_func_input.turn_id,
                    #    len(output.itl), request_func_input.output_len, output.latency)
                else:
                    output.error = response.reason or ""
                    output.success = False
                    error_text = await response.text()
                    update_conversation(request_func_input.conversation_id, "Error", [])
                    print("Response status:", response.status)
                    print("Response headers:", response.headers)
                    print("Response body:", error_text)
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            print(output.error)
        n_running_req -= 1
        n_completed_req += 1
        if n_completed_req % 100 == 0:
            metrics_url = f"{request_func_input.api_url.replace('v1/chat/completions', '')}metrics"
            response = requests.get(metrics_url)
            for line in response.text.split("\n"):
                if "gpu_prefix_cache_hit_rate{" in line:
                    print(line)

    if pbar:
        pbar.update(1)
    return output


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv('VLLM_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(pretrained_model_name_or_path):
            model_path = snapshot_download(
                model_id=pretrained_model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])

            return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
            pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(
            pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    if tokenizer_mode == "mistral":
        try:
            from vllm.transformers_utils.tokenizer import MistralTokenizer
        except ImportError as e:
            raise ImportError("MistralTokenizer requires vllm package.\n"
                              "Please install it with `pip install vllm` "
                              "to use mistral tokenizer mode.") from e
        return MistralTokenizer.from_pretrained(
            str(pretrained_model_name_or_path))
    else:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
}
