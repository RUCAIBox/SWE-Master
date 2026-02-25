#!/bin/bash

# N_HOURS=4 N_GPUS=4 modal run --detach swesmith/train/serve_sglang.py \
#     --model-path /weights/outputs/qwen2p5-coder-32b-lora-lr1e-4-warmup5___difficulty/qwen2p5-coder-32b-lora-lr1e-4-warmup5___difficulty/merged \
#     --served-model-name gpt-4o \
#     --tokenizer-path /weights/Qwen/Qwen2.5-Coder-32B-Instruct

N_HOURS=3 N_GPUS=8 modal run --detach swesmith/train/serve_sglang.py \
    --model-path /weights/final/qwen2p5-coder-32b-full-lr5e-5-warmup5___ft_xml_all_250413_run3/epoch_2 \
    --served-model-name gpt-4o \
    --tokenizer-path /weights/Qwen/Qwen2.5-Coder-32B-Instruct
