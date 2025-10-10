import os

HOME = '/home/ubuntu'
MODEL = "Qwen/Qwen3-32B-FP8"
DIR = f"results/{MODEL.split('/')[-1]}"
if not os.path.exists(DIR):
    os.makedirs(DIR)

SERVER_COMMAND_SUFFIX = ""
SERVER_COMMAND_PREFIX = ""

VLLM_SERVER_CMD_TEMPLATE = (
    # "/usr/local/bin/nsys profile -o /tmp/0.nsys-rep -w true -t cuda,nvtx,osrt,cudnn,cublas 
    # -s cpu -f true -x false --duration=120 --cuda-graph-trace node " TRANSFORMERS_OFFLINE=1 
    f"VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO"
    f" vllm serve {MODEL} --gpu_memory_utilization 0.95 --disable-log-requests "
    "--max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce "
    f"--enable-chunked-prefill --enable-prefix-caching --pipeline-parallel-size 1 " # --no-enable-prefix-caching
    "{} "
)
# VLLM_SERVER_CMD_TEMPLATE += "" if "0.5B" in MODEL else " --tensor-parallel-size 4 "

CLIENT_CMD_TEMPLATE = (
    f"python ~/vllm/benchmarks/benchmark_serving.py --result-dir {DIR} "
    f"--save-result --model {MODEL} --endpoint /v1/chat/completions "
    "--dataset-path {} --dataset-name {}  --host {} --port {} "
    "--result-filename {} --num-prompts {} --request-rate {} --session-rate {} "
    "--checkpoint {} --use-oracle {} --use-token-id {} --use-lru {} --max-active-conversations {} "
    "--time-limit {} "
)

SERVER_READY_PATTERN = r"startup complete"
CUDA_OOM_PATTERN = r"CUDA out of memory"
ERROR_PATTERN = r"Traceback (most recent call last):"
RAISE_PATTERN = r"raise"
