#!/bin/bash

#SBATCH --job-name=llm
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gpus-per-node 4
#SBATCH --time=04:00:00
#SBATCH --output=bx_log/llama3_8b_instruct_gpqa1.out
#SBATCH --error=bx_log/llama3_8b_instruct_gpqa1.err

cd [Path to install] /lm-evaluation-harness
source [Your env path] e.g., env/bin/activate
export HF_TOKEN=[Your HF token, can be found in huggingface account settings]
export HF_DATASETS_CACHE=[where to save HF datasets such as GSM8K]
export HF_HOME=[Model cache]
export EB_KV=4E-3

# lm_eval --model hf \
#     --model_args pretrained='EleutherAI/gpt-j-6B' \
#     --tasks hellaswag \
#     --device cuda:0 \
#     --batch_size 8

# lm_eval --model hf \
#     --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks gsm8k \
#     --device cuda:0 \
#     --num_fewshot 0 \
#     --batch_size 2

# lm_eval --model hf \
#     --model_args pretrained=facebook/opt-30b,parallelize=True,use_cache=True \
#     --tasks copa \
#     --batch_size 4 \
#     --show_config \
#     --trust_remote_code