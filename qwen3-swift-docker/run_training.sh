#!/bin/bash

# Qwen3-Embedding SWIFT Training Script
# Based on: https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md

echo "🚀 Starting Qwen3-Embedding SWIFT Training"

# Set number of GPUs (adjust based on your system)
nproc_per_node=1  # Change to 8 if you have 8 GPUs

# Run the SWIFT training command as per documentation
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset data \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 20 \
    --num_train_epochs 5 \
    --save_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3

echo "✅ Training completed!" 