# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import base64
import csv
import gc
import io
import json
import os
import csv
import math
import numpy
import random
import requests
import time
import warnings
from asyncio import Lock
from collections.abc import AsyncGenerator, Collection
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict
from sortedcontainers import SortedDict
from typing import Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from datasets import load_dataset
from PIL.Image import Image
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

MILLISECONDS_TO_SECONDS_CONVERSION = 1000

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    num_conversations: int = 0,
    fixed_output_len: Optional[int] = None,
) -> list[tuple[str, int, int, None]]:
    """
    Load ShareGPT dataset and return tokenized conversations with all turns.
    """
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Filter out conversations with less than 2 turns.
    # dataset = [data for data in dataset if len(data["conversations"]) >= 4]
    
    # Only use the first num_conversations of the dataset.
    #if num_conversations == 0:
    #    num_conversations = int(num_requests * 0.5)
    #dataset = dataset[:num_conversations]
    
    # Flatten all turns from all conversations.
    flattened_dataset = []
    for data in dataset:
        conversations = data["conversations"]
        for i in range(len(conversations) - 1):  # Ensure pairs (Q/A) are preserved
            prompt = conversations[0]["value"]
            completion = conversations[i + 1]["value"]
            flattened_dataset.append((prompt, completion))
    
    # Shuffle dataset.
    random.shuffle(flattened_dataset)
    
    # Filter out sequences that are too long or too short.
    filtered_dataset: list[tuple[str, int, int, None]] = []
    for i in range(len(flattened_dataset)):
        #if len(filtered_dataset) == num_requests:
        #    break
        
        prompt, completion = flattened_dataset[i]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        
        filtered_dataset.append([prompt, prompt_len, output_len, None])
    
    return filtered_dataset


def sample_burstgpt_requests(
    dataset_path: str,  # Path to the CSV file
    prefix_len: int,
    input_len_range: Optional[list[int]],
    output_len: int,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    random_seed: int,
) -> list[tuple[str, int, int]]:
    """
    Generate prompts based on input lengths read from a CSV file.
    """
    if not args.time_scale:
        raise(ValueError, "csv requires args.time_scale")
    if args.time_scale != 0:
        print('time_scale will override request_rate')

    # Read input lengths from the CSV file
    input_lens = []
    output_lens = []
    input_timestamps = []
    with open(dataset_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        i = 0
        print('request len range: ', input_len_range)
        for row in csv_reader:
            if int(row['Request tokens']) > input_len_range[0] and int(row['Request tokens']) < input_len_range[1]:
                input_lens.append(int(row['Request tokens']))
                output_lens.append(int(row['Response tokens']))
                input_timestamps.append(float(row['Timestamp']) / args.time_scale)
                i = i + 1
                if i >= num_requests:
                    break

    # Generate prefix tokens
    prefix_token_ids = np.random.randint(0, tokenizer.vocab_size, size=prefix_len).tolist()

    input_requests = []
    for i in range(len(input_lens)):
        input_len_current = input_lens[i]

        prompt = tokenizer.decode(prefix_token_ids +\
         [np.random.randint(0, tokenizer.vocab_size) % tokenizer.vocab_size for j in range(input_len_current)])

        cur_output_len = output_len
        if output_len == -1:
            cur_output_len = output_lens[i]
        input_requests.append([prompt, int(prefix_len + input_len_current),
                               int(cur_output_len), None])

    return input_requests, input_timestamps


'''
def sample_burstgpt_requests(
    dataset_path: str,
    num_requests: int,
    random_seed: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[str, int, int, None]]:
    df = pd.read_csv(dataset_path)
    gpt4_df = df[df["Model"] == "GPT-4"]
    # Remove the failed requests (i.e., response length is 0)
    gpt4_df = gpt4_df[gpt4_df["Response tokens"] > 0]
    # Randomly sample num_requests from the dataset
    if num_requests <= len(gpt4_df):
        gpt4_df = gpt4_df.sample(n=num_requests, random_state=random_seed)
    else:
        gpt4_df = gpt4_df.sample(n=num_requests,
                                 random_state=random_seed,
                                 replace=True)
    # Convert the dataframe to a list of tuples
    dataset = gpt4_df.values.tolist()
    input_requests = []
    for i in range(num_requests):
        input_len = int(dataset[i][2])
        output_len = int(dataset[i][3])
        prompt = tokenizer.decode([(i + j) % tokenizer.vocab_size
                                   for j in range(input_len)])
        input_requests.append((prompt, input_len, output_len, None))
    return input_requests
'''

def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[str, str, int, int, None]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: list[tuple[str, int, int]] = []
    for _ in range(num_requests):
        num_lines_needed = num_input_lines - num_prefix_lines
        sampled_lines = "".join(prefix_lines +
                                random.choices(poem_lines, k=num_lines_needed))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len, None))

    return sampled_requests


def sample_vision_arena_requests(
    dataset,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> list[tuple[str, str, int, Optional[dict[str, Collection[str]]]]]:
    sampled_requests: list[tuple[str, int, int, dict[str,
                                                     Collection[str]]]] = []
    for data in dataset:
        if len(sampled_requests) == num_requests:
            break

        prompt = data["turns"][0][0]['content']

        prompt_token_ids = tokenizer(prompt).input_ids
        if fixed_output_len is None:
            # Default max output len is set to 128
            print("--hf-output-len is not provided. Using default value 128.")
            fixed_output_len = 128

        prompt_len = len(prompt_token_ids)
        output_len = fixed_output_len

        assert isinstance(
            data["images"][0],
            Image), ("Input image format must be `PIL.Image.Image`, "
                     f"given {type(data['image'])}.")
        image: Image = data["images"][0]
        image = image.convert("RGB")
        image_data = io.BytesIO()
        image.save(image_data, format='JPEG')
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        mm_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

        sampled_requests.append((prompt, prompt_len, output_len, mm_content))

    return sampled_requests


def sample_hf_requests(
    dataset_path: str,
    dataset_subset: Optional[str],
    dataset_split: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    random_seed: int,
    fixed_output_len: Optional[int] = None,
) -> list[tuple[str, str, int, Optional[dict[str, Collection[str]]]]]:

    # Special case for vision_arena dataset
    if dataset_path == 'lmarena-ai/vision-arena-bench-v0.1' \
        and dataset_subset is None:
        assert dataset_split == "train"
        dataset = load_dataset(dataset_path,
                               name=dataset_subset,
                               split=dataset_split,
                               streaming=True)
        dataset = dataset.shuffle(seed=random_seed)
        return sample_vision_arena_requests(dataset, num_requests, tokenizer,
                                            fixed_output_len)

    dataset = load_dataset(dataset_path,
                           name=dataset_subset,
                           split=dataset_split,
                           streaming=True)
    assert "conversations" in dataset.features, (
        "HF Dataset must have 'conversations' column.")
    filter_func = lambda x: len(x["conversations"]) >= 2
    filtered_dataset = dataset.shuffle(seed=random_seed).filter(filter_func)
    sampled_requests: list[tuple[str, int, int, dict[str,
                                                     Collection[str]]]] = []
    for data in filtered_dataset:
        if len(sampled_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = data["conversations"][0]["value"]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = data["conversations"][1]["value"]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if fixed_output_len is None and (prompt_len < 4 or output_len < 4):
            # Prune too short sequences.
            continue
        if fixed_output_len is None and \
            (prompt_len > 1024 or prompt_len + output_len > 2048):
            # Prune too long sequences.
            continue

        if "image" in data and isinstance(data["image"], Image):
            image: Image = data["image"]
            image = image.convert("RGB")
            image_data = io.BytesIO()
            image.save(image_data, format='JPEG')
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
            mm_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            }
        else:
            mm_content = None

        sampled_requests.append([prompt, prompt_len, output_len, mm_content])

    return sampled_requests


def sample_random_requests(
    prefix_len: int,
    input_len: int,  # Single integer input length
    output_len: int,
    num_requests: int,
    range_ratio: float,
    input_lens: Optional[list[int]],  # Optional list of input lengths to overwrite input_len
    input_ratios: list[float],  # Ratios for selecting input lengths if input_lens is provided
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[str, int, int]]:
    # If input_lens is provided, use it to override input_len behavior
    if input_lens:
        assert len(input_lens) == len(input_ratios), "input_lens and input_ratios must have the same length"
        assert abs(sum(input_ratios) - 1.0) < 1e-6, "input_ratios must sum to 1.0"
        selected_input_lens = np.random.choice(input_lens, size=num_requests, p=input_ratios)
    else:
        # Use input_len if input_lens is not provided
        selected_input_lens = [input_len] * num_requests

    # Generate prefix tokens
    prefix_token_ids = np.random.randint(0, tokenizer.vocab_size, size=prefix_len).tolist()

    input_requests = []
    for i in range(num_requests):
        input_len_current = selected_input_lens[i]
        output_length_variation = np.random.randint(
            int(output_len * range_ratio),
            output_len + 1,
        )

        prompt = ' '.join(['cat' for j in range(input_len_current)])
        # print(len(tokenizer.encode(prompt)), input_len_current)

        input_requests.append([prompt, int(prefix_len + input_len_current),
                               int(output_length_variation), None])

    return input_requests


async def get_request(
    input_requests: list[tuple[str, int, int]],
    request_rate: float,
    input_timestamps: Optional[list[float]],
    burstiness: float = 1.0,
) -> AsyncGenerator[tuple[str, int, int], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a tuple.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = iter(input_requests)
    for i, request in enumerate(input_requests):
        # Sample the request interval from the exponential distribution.
        if input_timestamps and i+1 < len(input_timestamps):
            yield request, input_timestamps[i]
            interval = input_timestamps[i+1] - input_timestamps[i]
        else:
            yield request, i
            # interval = np.random.exponential(1.0 / request_rate)
            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue
            interval = 1.0 / request_rate

        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[tuple[str, int, int]],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if output_len is None:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(outputs[i].generated_text,
                              add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.median(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.mean(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


class RequestStats:
    def __init__(self):
        self.training_data = OrderedDict()
        training_dir = os.path.join(args.result_dir, "training_data")
        os.makedirs(training_dir, exist_ok=True)
        self.training_file = os.path.join(training_dir, f'{int(time.time())}.csv')
        with open(self.training_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['tp', 'pp', 'in_flight_tokens', 'progress', 'queue_tokens',
                            'in_flight_prompt_cnt', 'prompt_len', 'ttft'])
    
    def add(self, id, pp, tp, in_flight_tokens, queue_tokens,
            in_flight_prompt_lens, prompt_dispatch_timestamps):
        progress = 0
        for l, t in zip(in_flight_prompt_lens, prompt_dispatch_timestamps):
            progress += (time.time() - t) * l
        if in_flight_tokens > 0:
            progress /= in_flight_tokens
        self.training_data[id] = [
            tp, pp, in_flight_tokens, progress, 
            queue_tokens, len(in_flight_prompt_lens)
        ]

    def label(self, sid, request_func_input, ttft, log_file):
        """Save training data for later model training."""
        self.training_data[request_func_input.id].extend([request_func_input.prompt_len, ttft])
        print(f"Server {sid} prompt_len={request_func_input.prompt_len} "
              f"ttft={ttft} ttft_={request_func_input.predicted_latency}", file=log_file)
        if request_func_input.id % 128 == 0:
            self.dump()
    
    def dump(self):
        # Save training_data to CSV.
        if 'test' not in args.dataset_path:
            with open(self.training_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                id_to_pop = []
                for id, row in self.training_data.items():
                    if len(row) >= 8: # already labelled
                        writer.writerow(row)
                        id_to_pop.append(id)
                for id in id_to_pop:
                    self.training_data.pop(id)
            # print(f"Training data saved to {self.training_file}")

class ServerStat:
    def __init__(self, i, url, max_concurrency, overhead, tp, pp):
        self.id = i
        self.url = url
        self.max_concurrency = max_concurrency
        self.overhead = overhead
        self.tp = tp            # Fixed: assign tp here.
        self.pp = pp
        self.in_flight_tokens = 0    # Tokens currently in process.
        self.count = 0               # Number of in-progress requests.
        self.ttft = 0
        self.prompt_lens = []        # List of prompt lengths processed.
        self.in_flight_prompt_lens = []
        self.prompt_dispatch_timestamps = []
        self.accumulated_lens = []   # History of accumulated token counts.
        self.queue_latencys = []     # Latency values for tasks waiting in queue.

# --------------------------
# Main benchmark function
# --------------------------
async def benchmark(
    backend: str,
    model_id: str,
    model_name: str,
    tokenizer,  # assumed PreTrainedTokenizerBase or similar.
    input_requests: list,
    input_timestamps,  # Optional list of floats.
    logprobs: int,
    request_rate: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list,
    selected_percentiles: list,
    ignore_eos: bool,
    goodput_config_dict: dict,
    max_concurrencys: list,
    overheads: list,
    tp_degrees: list,
    pp_degrees: list,
    lora_modules: list = None,
):
    # Open log file.
    if args.result_dir:
        file_name = os.path.join(args.result_dir, args.result_filename)
        log_name = file_name + '.log'
        log_file = open(log_name, 'w')
        # Initialize an empty dictionary for the models.
        models = {}
        if args.len_based_dispatching >= 2:
            for tp in [1, 2, 4]:
                for pp in [1, 2, 4]:
                    model_file = f"{args.result_dir}/training_data/lgb_model_tp{tp}_pp{pp}.txt"
                    if os.path.exists(model_file):
                        models[(tp, pp)] = lgb.Booster(model_file=model_file)
                        print(f"Loaded model for tp={tp}, pp={pp} from {model_file}")
    else:
        log_file = open("benchmark.log", 'w')

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    server_ids = range(len(args.hosts))
    server_stats_condition = asyncio.Condition()
    server_stats = []

    request_queues = [SortedDict() for _ in range(len(args.hosts))]
    req_stats = RequestStats()

    def estimate_request_latency(stats, queue_tokens, prompt_len):
        """
        Estimate the latency for a request using a LightGBM model.
        """
        current_time = time.time()
        progress = 0.0
        for l, t in zip(stats.in_flight_prompt_lens, stats.prompt_dispatch_timestamps):
            progress += (current_time - t) * l
        if stats.in_flight_tokens > 0:
            progress /= stats.in_flight_tokens
        else:
            progress = 0.0

        in_flight_prompt_cnt = len(stats.in_flight_prompt_lens)
        
        # Construct the feature vector as a 2D numpy array.
        feature_array = np.array([[stats.in_flight_tokens, progress, queue_tokens,
                                    in_flight_prompt_cnt, prompt_len]])
        
        # Use a dictionary to select the appropriate model based on stats.pp.
        model = models.get((stats.tp, stats.pp))
        if model is None:
            raise ValueError("cannot find lightgbm model", stats.tp, stats.pp)
        
        # Predict and return the latency.
        predicted_latency = model.predict(feature_array)[0]
        return predicted_latency

    def estimate_server_max_latency(stats, queue: SortedDict):
        max_latency = float('-inf')
        queue_tokens = 0
        for req, _ in queue.values():
            req_latency = estimate_request_latency(stats, queue_tokens, req.prompt_len)
            req_latency += time.time() - req.timestamp
            queue_tokens += req.prompt_len
            if req_latency > max_latency:
                max_latency = req_latency
        return max_latency

    def get_base_url(server_id):
        return f"http://{args.hosts[server_id]}:{args.ports[server_id]}"
    
    def get_api_url(request_func_input) -> int:
        prompt_len = request_func_input.prompt_len
        best_server = None
        if args.len_based_dispatching <= 0:
            best_load = float('inf')
            for i in range(len(args.hosts)):
                queue_load = sum(req.prompt_len for req, _ in request_queues[i].values())
                load = server_stats[i].in_flight_tokens + queue_load
                if load < best_load:
                    best_load = load
                    request_func_input.predicted_latency = load
                    best_server = i
        else:
            best_load = float('inf')
            for i in range(len(args.hosts)):
                queue_load = sum(req.prompt_len for req, _ in request_queues[i].values())
                load = server_stats[i].in_flight_tokens / server_stats[i].pp + queue_load
                load = (load + prompt_len * server_stats[i].pp) * server_stats[i].overhead
                if args.len_based_dispatching >= 2:
                    load = estimate_request_latency(server_stats[i], queue_load, prompt_len)
                if load < best_load:
                    best_load = load
                    request_func_input.predicted_latency = load
                    best_server = i
        return best_server

    def rebalance_queues():
        """
        Globally rebalance the request queues across all servers.
        """
        
        while True:
            latencies = []
            for i in range(len(args.hosts)):
                lat = estimate_server_max_latency(server_stats[i], request_queues[i])
                latencies.append(lat)
            H = max(range(len(latencies)), key=lambda i: latencies[i])
            L = min(range(len(latencies)), key=lambda i: latencies[i])
            
            if H == L or not request_queues[H]:
                break

            old_latency_H = latencies[H]
            new_queue_H = request_queues[H].copy()
            candidate_timestamp, candidate_value = new_queue_H.popitem(0)
            new_latency_H = estimate_server_max_latency(server_stats[H], new_queue_H)
            
            new_queue_L = request_queues[L].copy()
            new_queue_L[candidate_timestamp] = candidate_value
            new_latency_L = estimate_server_max_latency(server_stats[L], new_queue_L)
            
            simulated_max = max(new_latency_H, new_latency_L)
            if simulated_max < old_latency_H:
                if not request_queues[H]:
                    raise ValueError('queue should not be empty')
                actual_timestamp, _ = request_queues[H].popitem(0)
                if actual_timestamp != candidate_timestamp:
                    raise ValueError('queue content mismatch')
                request_queues[L][candidate_timestamp] = candidate_value
                print("=== Rebalance attempt at time:", time.time(), "===", file=log_file)
                for idx in range(len(args.hosts)):
                    stats = server_stats[idx]
                    q = request_queues[idx]
                    print(f"Server {idx}: ttft={stats.ttft} in_flight_tokens={stats.in_flight_tokens}", file=log_file)
                    for key, (req, _) in q.items():
                        print(f"        Request: time = {req.timestamp}, len = {req.prompt_len}", file=log_file)
                print(f"Moved prompt len {candidate_value[0].prompt_len} from {H} to {L}", file=log_file)
            else:
                break

            new_latency_H_actual = estimate_server_max_latency(server_stats[H], request_queues[H])
            new_latency_L_actual = estimate_server_max_latency(server_stats[L], request_queues[L])
            if new_latency_L_actual < new_latency_H_actual:
                continue
            else:
                break

    async def request_handler(server_id: int):
        """Process requests for a specific server."""
        this_server = server_stats[server_id]
        while True:
            if not request_queues[server_id]:
                await asyncio.sleep(0.001)
                continue

            if this_server.in_flight_tokens > this_server.max_concurrency and \
                    len(this_server.in_flight_prompt_lens) > this_server.pp:
                await asyncio.sleep(0.001)
                continue

            if args.len_based_dispatching >= 3:

                rebalance_queues()

                if not request_queues[server_id]:
                    await asyncio.sleep(0.001)
                    continue

            _, (request_func_input, future) = request_queues[server_id].popitem(0)

            # Set the API URL and record the dispatch time.
            request_func_input.api_url = this_server.url
            request_func_input.dispatch_timestamp = round(time.time(), 3)
            prompt_len = request_func_input.prompt_len

            req_stats.add(
                id=request_func_input.id,
                pp=this_server.pp,
                tp=this_server.tp,
                in_flight_tokens=this_server.in_flight_tokens,
                queue_tokens=sum(req.prompt_len for req, _ in request_queues[server_id].values()),
                in_flight_prompt_lens=this_server.in_flight_prompt_lens,
                prompt_dispatch_timestamps=this_server.prompt_dispatch_timestamps
            )

            async with server_stats_condition:
                this_server.count += 1
                this_server.in_flight_prompt_lens.append(prompt_len)
                this_server.prompt_dispatch_timestamps.append(request_func_input.dispatch_timestamp)
                this_server.in_flight_tokens += prompt_len
                this_server.prompt_lens.append(prompt_len)
                this_server.accumulated_lens.append(this_server.in_flight_tokens)
                this_server.queue_latencys.append(time.time() - request_func_input.timestamp)

            # Process the request.
            result = await request_func(request_func_input=request_func_input, pbar=pbar)
            result.server = this_server.url
            result.done_timestamp = time.time()
            result.ttft = result.done_timestamp - request_func_input.timestamp

            async with server_stats_condition:
                this_server.count -= 1
                this_server.in_flight_prompt_lens.remove(prompt_len)
                this_server.prompt_dispatch_timestamps.remove(request_func_input.dispatch_timestamp)
                this_server.in_flight_tokens -= prompt_len
                this_server.ttft = result.ttft
                
                req_stats.label(this_server.id, request_func_input, result.ttft, log_file)
                server_stats_condition.notify_all()

            future.set_result(result)

    async def enqueue_request(request_func_input):
        """Adds a request to the appropriate queue and awaits its result."""
        async with server_stats_condition:
            chosen_server = get_api_url(request_func_input)
        future = asyncio.Future()
        request_queues[chosen_server][request_func_input.timestamp] = (request_func_input, future)
        return await future

    # Initialize each server's stats.
    for i in server_ids:
        server_stats.append(ServerStat(
            i,
            url=f"{get_base_url(i)}{args.endpoint}",
            max_concurrency=max_concurrencys[i],
            overhead=overheads[i],
            tp=tp_degrees[i],
            pp=pp_degrees[i],
        ))

    consumer_tasks = []
    for i in server_ids:
        for _ in range(64):  # Number of concurrent consumer tasks per server.
            consumer_tasks.append(asyncio.create_task(request_handler(i)))

    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Adjust prompt lengths in input_requests.
    for i in range(len(input_requests)):
        input_requests[i][1] = int(input_requests[i][1] * args.prompt_len_scale)

    # Reset prefix cache for each server.
    for i in server_ids:
        response = requests.post(get_base_url(i) + "/reset_prefix_cache")
        if response.status_code == 200:
            print(f"Server {i}: Prefix cache reset successfully.")
        else:
            print(f"Server {i}: Failed to reset prefix cache. Status code: {response.status_code}")

    # If using LoRA modules, create an iterator for them.
    if lora_modules:
        lora_modules_iter = iter([random.choice(lora_modules) for _ in range(len(input_requests))])
    else:
        lora_modules_iter = None

    if profile:
        print("Starting profiler...")
        for i in server_ids:
            profile_input = RequestFuncInput(
                model=model_id,
                model_name=model_name,
                prompt=input_requests[0][0],
                api_url=get_base_url(i) + "/start_profile",
                prompt_len=input_requests[0][1],
                output_len=input_requests[0][2],
                logprobs=logprobs,
                multi_modal_content=None,
                ignore_eos=ignore_eos,
                timestamp=time.time()
            )
            profile_output = await request_func(request_func_input=profile_input, pbar=pbar)
            if profile_output.success:
                print(f"Profiler started on server {i}")

    print(f"Traffic request rate: {request_rate}")
    if args.len_based_dispatching >= 3:
        print(f"Max # of tokens on each instance: {max_concurrencys}")

    benchmark_start_time = time.perf_counter()

    # Producer: enqueue requests.
    producer_tasks = []
    async for request, timestamp in get_request(input_requests, request_rate, input_timestamps, args.burstiness):
        prompt, prompt_len, output_len, mm_content = request
        req_model_id, req_model_name = model_id, model_name
        if lora_modules_iter:
            req_lora_module = next(lora_modules_iter)
            req_model_id = req_lora_module
            req_model_name = req_lora_module

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url="",  # Will be set later in the handler.
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            timestamp=time.time(),
            id=len(producer_tasks)
        )
        producer_tasks.append(asyncio.create_task(enqueue_request(request_func_input)))

    print('Done enqueueing requests.')

    def all_requests_processed():
        return all(server_stats[i].count == 0 for i in server_ids)

    async with server_stats_condition:
        await server_stats_condition.wait_for(all_requests_processed)
    print('Done processing requests.')
    for s in server_stats:    
        print('number of prompts and tokens: ', len(s.prompt_lens), sum(s.prompt_lens))

    outputs = await asyncio.gather(*producer_tasks)
    for task in consumer_tasks:
        task.cancel()

    if profile:
        print("Stopping profiler...")
        for i in server_ids:
            profile_input = RequestFuncInput(
                model=model_id,
                prompt=input_requests[0][0],
                api_url=get_base_url(i) + "/stop_profile",
                prompt_len=input_requests[0][1],
                output_len=input_requests[0][2],
                logprobs=logprobs,
                timestamp=time.time()
            )
            profile_output = await request_func(request_func_input=profile_input, pbar=pbar)
            if profile_output.success:
                print(f"Profiler stopped on server {i}")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    print(f"Benchmark duration: {benchmark_duration} seconds")

    # Retrieve metrics from each server
    for i in server_ids:
        metrics_url = f"{get_base_url(i)}/metrics"
        response = requests.get(metrics_url)
        for line in response.text.split("\n"):
            if "prefix_cache_hit_rate{" in line:
                print(line)

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "output_lens": actual_output_lens,
        "input_lens": [server_stats[i].prompt_lens for i in server_ids],
        "accumulated_lens": [server_stats[i].accumulated_lens for i in server_ids],
        "queue_latencys": [[round(num, 2) for num in server_stats[i].queue_latencys] for i in server_ids],
        "ttfts": [round(output.ttft, 2) for output in outputs],
        "ttfts_host1": [round(output.ttft, 2) for output in outputs if args.hosts[0] in output.server],
        "ttfts_host2": [round(output.ttft, 2) for output in outputs if args.hosts[0] not in output.server],
        #"itls": [round(output.itl, 2) for output in outputs],
        #"itls_host1": [round(output.itl, 2) for output in outputs if args.hosts[0] in output.server],
        #"itls_host2": [round(output.itl, 2) for output in outputs if args.hosts[0] not in output.server],
        # "generated_texts": [output.generated_text for output in outputs],
        # "errors": [output.error for output in outputs],
    }
    print("{:<40} {:<10.2f}".format("P90 latency (all):", np.percentile(result['ttfts'], 90)))
    print("{:<40} {:<10.2f}".format("P90 latency (host1):", np.percentile(result['ttfts_host1'], 90)))
    print("{:<40} {:<10.2f}".format("P90 latency (host2):", np.percentile(result['ttfts_host2'], 90)))
    print("{:<40} {:<10.2f}".format("P99 latency (all):", np.percentile(result['ttfts'], 99)))
    print("{:<40} {:<10.2f}".format("P99 latency (host1):", np.percentile(result['ttfts_host1'], 99)))
    print("{:<40} {:<10.2f}".format("P99 latency (host2):", np.percentile(result['ttfts_host2'], 99)))

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds.") from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any],
                                     file_name: str) -> None:
    metrics = [
        "median_ttft_ms", "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "median_itl_ms", "mean_itl_ms", "std_itl_ms", "p99_itl_ms"
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]]
                 for k in metrics},
        extra_info={
            k: results[k]
            for k in results if k not in metrics and k not in ignored_metrics
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    # print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    tokenizer = get_tokenizer(tokenizer_id,
                              tokenizer_mode=tokenizer_mode,
                              trust_remote_code=args.trust_remote_code)
    input_timestamps = None

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required.")

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.random_output_len,
        )

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt, prompt_len, output_len, None)
                              for prompt, prompt_formatted, prompt_len,
                              output_len, _ in input_requests]
        else:
            assert (
                tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt_formatted, prompt_len, output_len, None)
                              for prompt, prompt_formatted, prompt_len,
                              output_len, _ in input_requests]

    elif args.dataset_name == "hf":
        input_requests = sample_hf_requests(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            random_seed=args.seed,
            fixed_output_len=args.hf_output_len,
        )

    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            input_lens=args.input_lens,
            input_ratios=args.input_ratios,
            tokenizer=tokenizer,
        )
    elif args.dataset_name == "csv" or args.dataset_name == "burstgpt":
        input_requests, input_timestamps = sample_burstgpt_requests(
            dataset_path=args.dataset_path,
            prefix_len=args.random_prefix_len,
            input_len_range=args.input_lens,
            output_len=args.random_output_len,
            num_requests=args.num_prompts,
            random_seed=args.seed,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    goodput_config_dict = check_goodput_args(args)

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            input_timestamps=input_timestamps,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrencys=args.max_concurrencys,
            overheads=args.overheads,
            tp_degrees=args.tp_degrees,
            pp_degrees=args.pp_degrees,
            lora_modules=args.lora_modules,
        ))

    # Save config and results to json
    if args.save_result:
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (args.request_rate if args.request_rate
                                       < float("inf") else "inf")
        result_json["burstiness"] = args.burstiness

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w", encoding='utf-8') as outfile:
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--hosts", 
                        nargs='+',
                        type=str, 
                        default="localhost,localhost", 
                        help="address of another vllm instance")
    parser.add_argument("--ports",
                        nargs='+',
                        type=str, 
                        default="8000,8000")

    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "csv"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--max-concurrencys",
        nargs="+",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")
    parser.add_argument(
        "--overheads",
        nargs="+",
        type=float,
        default=None,
        help="server_load = number_of_tokens * overhead")
    parser.add_argument(
        "--prompt-len-ranges",
        nargs="+",
        type=int,
        default=None,
        help="server will mainly run the prompt len in the range")
    parser.add_argument(
        "--tp-degrees",
        nargs="+",
        type=int,
        default=None,
        help="server's tensor parallelism degree")
    parser.add_argument(
        "--pp-degrees",
        nargs="+",
        type=int,
        default=None,
        help="server's pipeline parallelism degree")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0xCADE)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives for goodput as \"KEY:VALUE\" "
        "pairs, where the key is a metric name, and the value is in "
        "milliseconds. Multiple \"KEY:VALUE\" pairs can be provided, "
        "separated by spaces. Allowed request level metric names are "
        "\"ttft\", \"tpot\", \"e2el\". For more context on the definition of "
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )
    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")
    
    random_group.add_argument(
        "--input-lens",
        type=int,
        nargs='+',
        default=[],  # Default list of possible input lengths
        help="List of possible input lengths to use for generating random requests. "
            "Specify multiple values to define varying input lengths."
    )

    random_group.add_argument(
        "--input-ratios",
        type=float,
        nargs='+',
        default=[0.94, 0.06],  # Default ratios corresponding to input lengths
        help="List of ratios for selecting input lengths. "
            "The length of this list should match the length of --input-len, "
            "and the ratios must sum to 1.0."
    )

    random_group.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help="speed up requests by time_scale.")
    
    random_group.add_argument(
        "--burstiness",
        type=float,
        default=None,
        help=".")
    
    random_group.add_argument(
        "--prompt-len-scale",
        type=float,
        default=1,
        help=".")
    
    random_group.add_argument(
        "--len-based-dispatching",
        type=int,
        default=0,
        help="prompt len based dispatching to vllm instances.")

    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral', 'custom'],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        'always use the slow tokenizer. \n* '
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer.')

    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. "
                        "If not specified, the model name will be the "
                        "same as the ``--model`` argument. ")

    parser.add_argument("--lora-modules",
                        nargs='+',
                        default=None,
                        help="A subset of LoRA module names passed in when "
                        "launching the server. For each request, the "
                        "script chooses a LoRA module at random.")

    args = parser.parse_args()
    if args.len_based_dispatching >= 3:
        print("rebalance enabled; no fairness guarantee")
    main(args)
