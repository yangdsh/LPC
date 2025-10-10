# LPC Setup
The below setup is tested on AWS EC2 image: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)

```
conda env create -f vllm_cache_bench/environment.yml
conda activate vllm-cuda121
```

## Install vllm
```
cd vllm
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/8d/cf/9b775a1a1f5fe2f6c2d321396ad41b9849de2c76fa46d78e6294ea13be91/vllm-0.7.3-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

## Download dataset
```
cd ../vllm_cache_bench
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```
## Run experiments
change HOME in constants_nips.py and configurations are on line 45, 265, 276
```
python run_nips.py
```

## Plot results
the raw results files in the paper are already inlcuded in results/ and fig/
```
python plot_size.py
python plot_reqrate.py
python plot_throughput.py
python plot_ttft.py
python plot_true_line.py
python get_predictor_accuracy.py
```

# Cite
https://neurips.cc/virtual/2025/poster/117662
