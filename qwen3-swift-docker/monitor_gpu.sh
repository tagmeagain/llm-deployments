#!/bin/bash

echo "📊 GPU Monitoring Script for Qwen3-Embedding Training"
echo "==================================================="

# Function to display GPU stats
show_gpu_stats() {
    echo "🕐 $(date '+%Y-%m-%d %H:%M:%S')"
    echo "GPU Usage:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r index name gpu_util mem_util mem_used mem_total temp; do
        echo "  GPU $index ($name):"
        echo "    GPU Utilization: ${gpu_util}%"
        echo "    Memory Utilization: ${mem_util}%"
        echo "    Memory Used: ${mem_used}MB / ${mem_total}MB"
        echo "    Temperature: ${temp}°C"
    done
    echo "----------------------------------------"
}

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Cannot monitor GPU."
    exit 1
fi

echo "Starting GPU monitoring..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# Monitor GPU every 5 seconds
while true; do
    show_gpu_stats
    sleep 5
done 