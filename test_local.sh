#!/bin/bash

# Local testing script for Qwen3-Embedding SWIFT training

echo "🔨 Building Docker image..."
cd qwen3-swift-docker
docker build -t qwen3-swift-docker:latest .

echo "📁 Creating output directory..."
mkdir -p ../output

echo "🚀 Running container with data mounted..."
docker run --gpus all -it \
    -v $(pwd)/../output:/workspace/output \
    -v $(pwd)/../data:/workspace/data \
    qwen3-swift-docker:latest

echo "✅ Container stopped!" 