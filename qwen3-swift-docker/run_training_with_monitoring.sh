#!/bin/bash

# Qwen3-Embedding SWIFT Training Script with GPU Monitoring and Optimization
# Based on: https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md

echo "🚀 Starting Qwen3-Embedding SWIFT Training with Maximum GPU Utilization"
echo "======================================================================"

# First, verify GPU is available
echo "🔍 Verifying GPU availability..."
./verify_gpu.sh

if [ $? -ne 0 ]; then
    echo "❌ GPU verification failed. Please check your Docker GPU setup."
    exit 1
fi

echo ""
echo "🔧 Optimizing GPU settings for maximum utilization..."

# Function to get GPU memory in GB
get_gpu_memory() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1
}

# Function to calculate optimal batch size for 0.6B model
calculate_optimal_batch_size() {
    local gpu_memory_gb=$1
    
    # For 0.6B model, calculate optimal batch size
    if [ $gpu_memory_gb -ge 24 ]; then
        echo 14
    elif [ $gpu_memory_gb -ge 16 ]; then
        echo 10
    elif [ $gpu_memory_gb -ge 12 ]; then
        echo 7
    elif [ $gpu_memory_gb -ge 8 ]; then
        echo 5
    else
        echo 4
    fi
}

# Function to calculate optimal gradient accumulation steps
calculate_gradient_accumulation() {
    local per_device_batch_size=$1
    
    # Target effective batch size of 32 for good training
    local effective_batch_size=32
    local grad_acc_steps=$((effective_batch_size / per_device_batch_size))
    
    # Ensure minimum of 1
    if [ $grad_acc_steps -lt 1 ]; then
        echo 1
    else
        echo $grad_acc_steps
    fi
}

# Get GPU information and calculate optimal settings
echo "📊 Analyzing GPU for optimal settings..."
gpu_memory_mb=$(get_gpu_memory)
gpu_memory_gb=$((gpu_memory_mb / 1024))
gpu_count=$(nvidia-smi --list-gpus | wc -l)

echo "GPU Memory: ${gpu_memory_gb}GB"
echo "Number of GPUs: ${gpu_count}"

# Calculate optimal settings
per_device_batch_size=$(calculate_optimal_batch_size $gpu_memory_gb)
gradient_accumulation_steps=$(calculate_gradient_accumulation $per_device_batch_size)
effective_batch_size=$((per_device_batch_size * gradient_accumulation_steps))

echo "✅ Optimized Settings for Maximum GPU Utilization:"
echo "  Per Device Batch Size: ${per_device_batch_size}"
echo "  Gradient Accumulation Steps: ${gradient_accumulation_steps}"
echo "  Effective Batch Size: ${effective_batch_size}"

# Memory utilization estimate
model_memory_gb=1.2  # 0.6B model
estimated_memory_usage=$((model_memory_gb + per_device_batch_size * 2))
memory_utilization=$((estimated_memory_usage * 100 / gpu_memory_gb))

echo "  Estimated Memory Usage: ~${estimated_memory_usage}GB"
echo "  Memory Utilization: ~${memory_utilization}%"

if [ $memory_utilization -gt 90 ]; then
    echo "⚠️  High memory utilization - monitor for OOM errors"
elif [ $memory_utilization -gt 70 ]; then
    echo "✅ Good memory utilization"
else
    echo "💡 Can potentially increase batch size for better utilization"
fi

# Set number of GPUs (adjust based on your system)
nproc_per_node=1  # Change to 8 if you have 8 GPUs

# Start GPU monitoring in background
echo ""
echo "📊 Starting GPU monitoring in background..."
./monitor_gpu.sh &
MONITOR_PID=$!

# Function to cleanup monitoring on exit
cleanup() {
    echo ""
    echo "🛑 Stopping GPU monitoring..."
    kill $MONITOR_PID 2>/dev/null
    echo "✅ Training completed with maximum GPU utilization!"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

echo ""
echo "🎯 Starting SWIFT training with optimized settings..."

# Run the SWIFT training command with optimized settings
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
    --per_device_train_batch_size $per_device_batch_size \
    --per_device_eval_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3

# The cleanup function will be called automatically on exit 