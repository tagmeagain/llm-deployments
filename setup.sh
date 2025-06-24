#!/bin/bash

# Qwen3-Embedding SWIFT Training Setup Script
# Based on: https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md

set -e

echo "🚀 Setting up Qwen3-Embedding SWIFT Training Environment"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  NVIDIA Docker runtime not detected. Make sure you have nvidia-docker installed."
    echo "   Install with: curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
    echo "   And: distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p output data

# Make training script executable
chmod +x run_training.sh

# Build and start the container
echo "🔨 Building Docker container..."
docker-compose build

echo "🚀 Starting container..."
docker-compose up -d

echo "✅ Container is running!"
echo ""
echo "📋 Next steps:"
echo "1. Access the container: docker-compose exec qwen3-swift-training bash"
echo "2. Navigate to workspace: cd /workspace"
echo "3. Run training: ./run_training.sh"
echo ""
echo "📊 Monitor training:"
echo "- Container logs: docker-compose logs -f qwen3-swift-training"
echo "- Check output directory: ls -la output/"
echo ""
echo "🛑 To stop: docker-compose down" 