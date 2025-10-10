import json
import subprocess
import time
import asyncio
from utils import kill_server
import os
import re
import copy
from constants_nips import *

async def main(sizes, scales, alg, _dataset, tag):
    server_configs = []
    i = 0
    for size in sizes:
        for scale in scales:
            if 'tensor-parallel' not in VLLM_SERVER_CMD_TEMPLATE:
                cuda_device_id = i
            else:
                cuda_device_id = ','.join(range(i,i+4))
            server_configs.append({'host': 'localhost', 
                'cuda_devices': f'CUDA_VISIBLE_DEVICES={cuda_device_id}',
                'eviction_algorithm': alg,
                'port': 8000+i,
                'size': size,
                'scale': scale,
                'args': f' --port {8000+i} '
                f' --eviction_algorithm {alg} --max-num-batched-tokens 2048 '
            })
            i += 1

    
    ''' {
            'num_prompts': 30000,
            'use_oracle': 3,
            'use_token_id': 1,
            'algorithm': 'ml-true'
        },
        {
            'num_prompts': 30000,
            'use_oracle': 2,
            'use_token_id': 1,
            'algorithm': 'ml-belady'
        }
        '''
    client_config_template = [
        {
            'num_prompts': 30000,
            'use_oracle': 0,
            'use_token_id': 1,
            'algorithm': 'ml'
        },
        {
            'num_prompts': 30000,
            'use_oracle': 0,
            'use_token_id': 0,
            'algorithm': 'lru'
        }
    ]

    client_configs = []
    for conf in client_config_template:
        # c['max_active_conversations'] = conv_cnt
        c = copy.deepcopy(conf)
        c['session_rate'] = 10
        if _dataset.startswith('tay'):
            c['checkpoint'] = f'{HOME}/vllm/benchmarks/checkpoints_tay_20/tay_epoch17_metric_0_6332.pt'
            c['dataset_file'] = f'{DATA_HOME}/tay.json'
            c['request_rate'] = 1
            c['session_rate'] = 10
            c['max_active_conversations'] = 200
            c['time_limit'] = 300
        elif _dataset.startswith('chatbot'):
            c['request_rate'] = 0.01
            c['max_active_conversations'] = 200
            c['checkpoint'] = f'{HOME}/vllm/benchmarks/checkpoints_chatbot_arena_20/chatbot_arena_epoch16_metric_0_4005.pt'
            c['dataset_file'] = '"lmsys/chatbot_arena_conversations"'
            c['time_limit'] = 1200
        elif _dataset.startswith('lmsys'):
            c['request_rate'] = 0.01
            c['max_active_conversations'] = 200
            c['checkpoint'] = f'{HOME}/vllm/benchmarks/checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_epoch11_metric_0_5797.pt'
            c['dataset_file'] = '"lmsys/lmsys-chat-1m"'
            c['time_limit'] = 1200
        elif _dataset.startswith('sharegpt'):
            c['request_rate'] = 0.01
            c['max_active_conversations'] = 200
            c['checkpoint'] = f'{HOME}/vllm/benchmarks/checkpoints_sharegpt_20/sharegpt_epoch19_metric_0_5427.pt'
            c['dataset_file'] = f'{HOME}/vllm_cache_bench/ShareGPT_V3_unfiltered_cleaned_split.json'
            c['time_limit'] = 1200
        # c['max_active_conversations'] = 200
        c['dataset_name'] = _dataset
        client_configs.append(c)

    def run_server(server_config):
        """Start the server with specified parallel sizes."""
        os.makedirs(f'{DIR}/{_dataset}-{tag}', exist_ok=True)
        os.makedirs(f'{DIR}/{_dataset}-{tag}/client_logs', exist_ok=True)
        log_file_name = f"{DIR}/{_dataset}-{tag}/server_{server_config['port']}_{server_config['client_algorithm']}.log"
        server_config['args'] += f"--num-gpu-blocks-override {server_config['size']} "
        server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(server_config['args'])

        ssh_command = (
            #f'ssh {server_config["host"]} '
            # f"source /opt/conda/etc/profile.d/conda.sh && "  # Ensure Conda is sourced
            # f"conda activate pytorch && "  # Activate the environment
            f"{SERVER_COMMAND_PREFIX} {server_config['cuda_devices']} {server_cmd} {SERVER_COMMAND_SUFFIX}"
        )
        print('\n', ssh_command, '\n')
        with open(log_file_name, "w") as log_file:
            process = subprocess.Popen(ssh_command, shell=True, stdout=log_file, stderr=log_file,
    executable="/bin/bash")

        return log_file_name


    def wait_for_server_ready(log_file_name, timeout=600):
        """Wait until the server is ready or a timeout occurs."""
        for _ in range(timeout):
            if os.path.exists(log_file_name):
                with open(log_file_name, "r") as log_file:
                    log_data = log_file.read()
                    if re.search(SERVER_READY_PATTERN, log_data):
                        print("Server is ready.")
                        return True
                    if re.search(CUDA_OOM_PATTERN, log_data) or re.search(ERROR_PATTERN, log_data) or re.search(RAISE_PATTERN, log_data):
                        print("Server encountered an error.")
                        return False
            time.sleep(1)
        print("Server startup timed out.")
        return False

    def get_file_name(server_config):
        # Use a simple timestamp for a unique result filename.
        return str(int(time.time())) + str()

    async def run_client(client_config, server_config):
        num_prompts = client_config['num_prompts']
        request_rate = client_config['request_rate']
        session_rate = client_config['session_rate']
        max_active_conversations = client_config['max_active_conversations']
        checkpoint = client_config['checkpoint']
        if 'scale' in server_config:
            client_config['scale'] = server_config['scale']
        request_rate *= client_config['scale'] if 'scale' in client_config else 1
        client_config['request_rate'] = request_rate
        print(client_config['scale'], client_config['request_rate'])
        time_limit = client_config['time_limit'] if 'time_limit' in client_config else 10000
        use_oracle = client_config['use_oracle'] if 'use_oracle' in client_config else 0
        use_token_id = client_config['use_token_id'] if 'use_token_id' in client_config else 0
        use_lru = 1 if 'lru' in client_config['algorithm'] else 0
        prefix = f"{server_config['port']}_{client_config['algorithm']}"
        result_filename = f"{_dataset}-{tag}/client_logs/{prefix}.json"
        
        # Extract host directly from the dictionary.
        host = server_config["host"]
        
        # Extract the port from the 'args' string.
        port_match = re.search(r'--port\s+(\d+)', server_config["args"])
        if not port_match:
            raise ValueError("Port not found in server configuration 'args'.")
        port = port_match.group(1)
        
        client_cmd = CLIENT_CMD_TEMPLATE.format(
            client_config['dataset_file'], 'conversation', host, 
            port, result_filename, num_prompts, request_rate, session_rate,
            checkpoint, use_oracle, use_token_id, use_lru, max_active_conversations,
            time_limit
        )
        client_cmd = f'{server_config["cuda_devices"]} {client_cmd}'
        
        while True:
            print("Running client command:", client_cmd, flush=True)
            process = await asyncio.create_subprocess_shell(
                client_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            log_file_name = f"{DIR}/{_dataset}-{tag}/client_{server_config['port']}_{client_config['algorithm']}.log"
            log_file = open(log_file_name, "w")
            print("Client stdout:", stdout.decode(), file=log_file)
            print("Client error:", stderr.decode(), file=log_file)
            if len(str(stdout.decode())) > 1000:
                break
        print('Done client: ' + client_cmd + '\n', flush=True)
        hit_ratios = []
        result = {}
        for line in stdout.decode().split("\n"):
            if 'gpu_prefix_cache_hit_rate' in line:
                hit_ratios.append(line.split()[-1])
            if 'Mean TTFT (ms):' in line:
                result['mean_ttft'] = line.split()[-1]
            if 'Median TTFT (ms):' in line:
                result['median_ttft'] = line.split()[-1]
            if 'P99 TTFT (ms):' in line:
                result['p99_ttft'] = line.split()[-1]
            if 'Total input tokens:' in line:
                result['input_tokens'] = line.split()[-1]
            if 'Total generated tokens:' in line:
                result['output_tokens'] = line.split()[-1]
            if 'Successful requests:' in line:
                result['num_requests'] = line.split()[-1]
        
        with open(f'{DIR}/{result_filename}.config', 'w') as fp:
            json.dump([client_config, server_config], fp)
        result['hit_ratios'] = hit_ratios
        for k in ['args', 'size']:
            result[k] = server_config[k]
        for k in client_config.keys():
            result[k] = client_config[k]
        result["result_file"] = result_filename
        return result

    async def start_server(server_config):
        # Launch the server and wait for it to be ready.
        log_file_name = await asyncio.to_thread(run_server, server_config)
        print("wait_for_server_ready:", log_file_name)
        is_ready = await asyncio.to_thread(wait_for_server_ready, log_file_name)
        return is_ready

    async def start_exp(server_conf, client_conf):
        client_config = copy.deepcopy(client_conf)
        server_config = copy.deepcopy(server_conf)
        server_config['client_algorithm'] = client_config['algorithm']
        if 'ml' in server_config['client_algorithm']:
            server_config['size'] -= 250 # 2GB
        print("Starting server configuration:", server_config)
        is_ready = await start_server(server_config)
        if not is_ready:
            print("Server startup failed. Skipping this experiment.")
            return
        result = await run_client(client_config, server_config)
        dataset_name = client_config['dataset_name']

        # Save results to `exp.json` in append mode
        exp_file = f"{DIR}/exp_{dataset_name}.json"
        
        # Load existing data if the file exists
        if os.path.exists(exp_file):
            with open(exp_file, "r") as fp:
                try:
                    existing_data = json.load(fp)
                    if not isinstance(existing_data, list):
                        existing_data = []  # Reset if data is corrupted
                except json.JSONDecodeError:
                    existing_data = []  # Reset if file is empty or corrupted
        else:
            existing_data = []

        # Append new results
        existing_data.append(result)

        # Write back to the file
        with open(exp_file, "w") as fp:
            json.dump(existing_data, fp, indent=4)

        print(f"Saved results to {exp_file}")

    for i, client_config in enumerate(client_configs):
        kill_server(server_configs[0]['host'])
        tasks = [start_exp(server_config, client_config) for server_config in server_configs]
        await asyncio.gather(*tasks)

'''# varying chat interval
if __name__ == "__main__":
    if '32B' in MODEL:
        for alg in ['ml']:
            for dataset in ['lmsys', 'chatbot', 'sharegpt', 'tay']:
                for sizes in [[10000]]:
                    for scales in [[0.25, 0.5, 1, 2]]:
                        asyncio.run(main(sizes, scales, alg, dataset, 'reqrate++'))

'''

# varying cache size
if __name__ == "__main__":
    for alg in ['ml']:
        for dataset in ['sharegpt', 'lmsys', 'chatbot']:
            for sizes in [[4000], [6000], [8000], [10000]]:
                for scales in [[1]]:
                    asyncio.run(main(sizes, scales, alg, dataset, 'size++'))

