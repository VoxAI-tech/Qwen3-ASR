#!/usr/bin/env bash
set -e

# GPU 0 has ~30GB used by other processes, so single-GPU on GPU 1.
# To use both GPUs, kill other processes on GPU 0 and switch to the
# dual-GPU section below.

# --- Single GPU (GPU 1) ---
CUDA_VISIBLE_DEVICES=1 uv run python finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file /data/razhan/qwen_data/train.jsonl \
  --eval_file /data/razhan/qwen_data/eval.jsonl \
  --output_dir /data/razhan/qwen_data/qwen3-asr-finetuning-out \
  --batch_size 16 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 3 \
  --log_steps 10 \
  --save_steps 500 \
  --save_total_limit 3 \
  --num_workers 2

# --- Dual GPU (uncomment below, comment above) ---
# export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nproc_per_node=2 finetuning/qwen3_asr_sft.py \
#   --model_path Qwen/Qwen3-ASR-1.7B \
#   --train_file /data/razhan/qwen_data/train.jsonl \
#   --eval_file /data/razhan/qwen_data/eval.jsonl \
#   --output_dir /data/razhan/qwen_data/qwen3-asr-finetuning-out \
#   --batch_size 16 \
#   --grad_acc 4 \
#   --lr 2e-5 \
#   --epochs 3 \
#   --log_steps 10 \
#   --save_steps 500 \
#   --save_total_limit 3 \
#   --num_workers 2
