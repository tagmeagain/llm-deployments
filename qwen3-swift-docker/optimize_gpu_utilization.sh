#!/bin/bash

echo "🔧 GPU Utilization Optimization Script"
echo "====================================="

# Function to get GPU memory in GB
get_gpu_memory() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1
}

# Function to calculate optimal batch size
calculate_optimal_batch_size() {
    local gpu_memory_gb=$1
    local model_size=$2
    
    # Base batch size calculation based on GPU memory and model size
    case $model_size in
        "0.5B")
            # For 0.5B model, we can use larger batches
            if [ $gpu_memory_gb -ge 24 ]; then
                echo 16
            elif [ $gpu_memory_gb -ge 16 ]; then
                echo 12
            elif [ $gpu_memory_gb -ge 12 ]; then
                echo 8
            elif [ $gpu_memory_gb -ge 8 ]; then
                echo 6
            else
                echo 4
            fi
            ;;
        "0.6B")
            # For 0.6B model
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
            ;;
        "1B")
            # For 1B model
            if [ $gpu_memory_gb -ge 24 ]; then
                echo 8
            elif [ $gpu_memory_gb -ge 16 ]; then
                echo 6
            elif [ $gpu_memory_gb -ge 12 ]; then
                echo 4
            else
                echo 2
            fi
            ;;
        "2B")
            # For 2B model
            if [ $gpu_memory_gb -ge 24 ]; then
                echo 4
            elif [ $gpu_memory_gb -ge 16 ]; then
                echo 3
            else
                echo 2
            fi
            ;;
        *)
            echo 4
            ;;
    esac
}

# Function to calculate optimal gradient accumulation steps
calculate_gradient_accumulation() {
    local gpu_memory_gb=$1
    local model_size=$2
    
    # Target effective batch size of 32 for good training
    local effective_batch_size=32
    local per_device_batch_size=$(calculate_optimal_batch_size $gpu_memory_gb $model_size)
    
    # Calculate gradient accumulation steps
    local grad_acc_steps=$((effective_batch_size / per_device_batch_size))
    
    # Ensure minimum of 1
    if [ $grad_acc_steps -lt 1 ]; then
        echo 1
    else
        echo $grad_acc_steps
    fi
}

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Cannot optimize GPU settings."
    exit 1
fi

# Get GPU information
echo "📊 Analyzing GPU configuration..."
gpu_memory_mb=$(get_gpu_memory)
gpu_memory_gb=$((gpu_memory_mb / 1024))
gpu_count=$(nvidia-smi --list-gpus | wc -l)

echo "GPU Memory: ${gpu_memory_gb}GB"
echo "Number of GPUs: ${gpu_count}"

# Test different model sizes
models=("0.5B" "0.6B" "1B" "2B")

echo ""
echo "🎯 Optimal Configuration for Maximum GPU Utilization:"
echo "====================================================="

for model_size in "${models[@]}"; do
    echo ""
    echo "Model: Qwen3-Embedding-${model_size}"
    echo "----------------------------------------"
    
    # Calculate optimal settings
    optimal_batch_size=$(calculate_optimal_batch_size $gpu_memory_gb $model_size)
    optimal_grad_acc=$(calculate_gradient_accumulation $gpu_memory_gb $model_size)
    effective_batch_size=$((optimal_batch_size * optimal_grad_acc))
    
    echo "Per Device Batch Size: ${optimal_batch_size}"
    echo "Gradient Accumulation Steps: ${optimal_grad_acc}"
    echo "Effective Batch Size: ${effective_batch_size}"
    
    # Memory utilization estimate
    case $model_size in
        "0.5B") model_memory_gb=1 ;;
        "0.6B") model_memory_gb=1.2 ;;
        "1B") model_memory_gb=2 ;;
        "2B") model_memory_gb=4 ;;
    esac
    
    estimated_memory_usage=$((model_memory_gb + optimal_batch_size * 2))
    memory_utilization=$((estimated_memory_usage * 100 / gpu_memory_gb))
    
    echo "Estimated Memory Usage: ~${estimated_memory_usage}GB"
    echo "Memory Utilization: ~${memory_utilization}%"
    
    if [ $memory_utilization -gt 90 ]; then
        echo "⚠️  High memory utilization - monitor for OOM errors"
    elif [ $memory_utilization -gt 70 ]; then
        echo "✅ Good memory utilization"
    else
        echo "💡 Can potentially increase batch size for better utilization"
    fi
done

echo ""
echo "🚀 Recommended Training Commands:"
echo "================================="

# Generate optimized training commands
for model_size in "${models[@]}"; do
    optimal_batch_size=$(calculate_optimal_batch_size $gpu_memory_gb $model_size)
    optimal_grad_acc=$(calculate_gradient_accumulation $gpu_memory_gb $model_size)
    
    echo ""
    echo "For Qwen3-Embedding-${model_size}:"
    echo "swift sft \\"
    echo "    --model Qwen/Qwen3-Embedding-${model_size} \\"
    echo "    --task_type embedding \\"
    echo "    --model_type qwen3_emb \\"
    echo "    --train_type full \\"
    echo "    --dataset data \\"
    echo "    --split_dataset_ratio 0.05 \\"
    echo "    --eval_strategy steps \\"
    echo "    --output_dir output \\"
    echo "    --eval_steps 20 \\"
    echo "    --num_train_epochs 5 \\"
    echo "    --save_steps 20 \\"
    echo "    --per_device_train_batch_size ${optimal_batch_size} \\"
    echo "    --per_device_eval_batch_size ${optimal_batch_size} \\"
    echo "    --gradient_accumulation_steps ${optimal_grad_acc} \\"
    echo "    --learning_rate 6e-6 \\"
    echo "    --loss_type infonce \\"
    echo "    --label_names labels \\"
    echo "    --dataloader_drop_last true \\"
    echo "    --deepspeed zero3"
done

echo ""
echo "💡 Tips for Maximum GPU Utilization:"
echo "===================================="
echo "1. Use the largest batch size that fits in GPU memory"
echo "2. Enable gradient accumulation to maintain effective batch size"
echo "3. Use DeepSpeed ZeRO-3 for memory optimization"
echo "4. Monitor GPU utilization with: ./monitor_gpu.sh"
echo "5. Adjust batch size if you see OOM errors"
echo "6. Consider using larger models if GPU memory allows" 