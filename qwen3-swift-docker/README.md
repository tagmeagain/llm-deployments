# Qwen3-Embedding SWIFT Training with Docker

This repository provides a Docker-based solution for fine-tuning Qwen3-Embedding models using SWIFT (SwiftLLM), solving the GLIBC version compatibility issue.

## Problem Solved

The original error `Import Error Version GLIBC_2.32 not found` occurs because your Ubuntu version has an older GLIBC than required by flash-attention. This Docker solution uses Ubuntu 24.04 with the correct GLIBC version.

## Prerequisites

1. **Docker** installed on your system
2. **Docker Compose** installed
3. **NVIDIA Docker runtime** installed
4. **NVIDIA GPU** with CUDA support

### Install NVIDIA Docker Runtime

```bash
# Add NVIDIA Docker repository
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🚀 GPU Utilization Guide

### 1. Verify GPU Setup Before Training

Before starting training, ensure your GPU is properly accessible:

```bash
# Build and start the container
docker-compose build
docker-compose up -d

# Access the container
docker-compose exec qwen3-swift-training bash

# Run GPU verification script
./verify_gpu.sh
```

The verification script will check:
- ✅ NVIDIA drivers and nvidia-smi availability
- ✅ CUDA environment variables
- ✅ PyTorch CUDA support
- ✅ GPU memory allocation
- ✅ SWIFT installation and GPU compatibility

### 2. Optimize GPU Settings for Maximum Utilization

Use the GPU optimization script to get the best settings for your hardware:

```bash
# Analyze your GPU and get optimal settings
./optimize_gpu_utilization.sh
```

This script will:
- 📊 Analyze your GPU memory and capabilities
- 🎯 Calculate optimal batch sizes for different model sizes
- 📈 Estimate memory utilization
- 🚀 Generate optimized training commands

### 3. Run Training with Maximum GPU Utilization

Use the optimized training script that automatically calculates the best settings:

```bash
# Run training with automatic GPU optimization
./run_training_optimized.sh
```

Or use the monitoring version for real-time GPU tracking:

```bash
# Run training with optimization and monitoring
./run_training_with_monitoring.sh
```

### 4. Monitor GPU During Training

Use the enhanced training script with built-in GPU monitoring:

```bash
# Run training with GPU monitoring
./run_training_with_monitoring.sh
```

Or monitor GPU manually in a separate terminal:

```bash
# In another terminal, access the container
docker-compose exec qwen3-swift-training bash

# Start GPU monitoring
./monitor_gpu.sh
```

### 5. Verify GPU is Being Fully Utilized

During training, you should see:
- **GPU Utilization**: Should be >90% during training
- **Memory Usage**: Should be >80% of available GPU memory
- **Temperature**: Should rise during intensive training
- **Effective Batch Size**: Should be 32 or higher for good training

### 6. Troubleshoot GPU Issues

If GPU is not being fully utilized:

1. **Check Docker GPU runtime**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

2. **Verify container GPU access**:
   ```bash
   docker-compose exec qwen3-swift-training nvidia-smi
   ```

3. **Check PyTorch CUDA**:
   ```bash
   docker-compose exec qwen3-swift-training python3 -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Test GPU memory allocation**:
   ```bash
   docker-compose exec qwen3-swift-training python3 -c "import torch; x = torch.randn(1000, 1000).cuda(); print('GPU memory:', torch.cuda.memory_allocated() / 1024**2, 'MB')"
   ```

5. **Run optimization analysis**:
   ```bash
   ./optimize_gpu_utilization.sh
   ```

## 🎯 Maximum GPU Utilization Techniques

### Automatic Optimization
The optimization scripts automatically:
- **Calculate optimal batch sizes** based on GPU memory
- **Adjust gradient accumulation** to maintain effective batch size of 32
- **Estimate memory usage** to prevent OOM errors
- **Recommend settings** for different model sizes

### Manual Optimization Tips
1. **Increase batch size** until you approach GPU memory limits
2. **Use gradient accumulation** to maintain effective batch size
3. **Enable DeepSpeed ZeRO-3** for memory optimization
4. **Monitor GPU utilization** with real-time monitoring
5. **Use larger models** if GPU memory allows
6. **Adjust learning rate** if batch size changes significantly

### GPU Memory Optimization
- **Shared memory**: Increased to 16GB in docker-compose.yml
- **CUDA memory allocation**: Optimized with `PYTORCH_CUDA_ALLOC_CONF`
- **DeepSpeed ZeRO-3**: Reduces memory footprint
- **Gradient accumulation**: Allows larger effective batch sizes

## Quick Start

1. **Clone and setup:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Access the container:**
   ```bash
   docker-compose exec qwen3-swift-training bash
   ```

3. **Optimize and run training:**
   ```bash
   cd /workspace
   ./verify_gpu.sh                           # Verify GPU setup
   ./optimize_gpu_utilization.sh             # Get optimal settings
   ./run_training_with_monitoring.sh         # Run with max GPU utilization
   ```

## Manual Setup

If you prefer manual setup:

1. **Build the container:**
   ```bash
   docker-compose build
   ```

2. **Start the container:**
   ```bash
   docker-compose up -d
   ```

3. **Access the container:**
   ```bash
   docker-compose exec qwen3-swift-training bash
   ```

4. **Optimize and run training:**
   ```bash
   cd /workspace
   ./verify_gpu.sh                           # Verify GPU setup
   ./optimize_gpu_utilization.sh             # Get optimal settings
   ./run_training_with_monitoring.sh         # Run with max GPU utilization
   ```

## Training Configuration

The training script uses the exact configuration from the [SWIFT documentation](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md):

- **Model**: Qwen/Qwen3-Embedding-0.6B
- **Dataset**: sentence-transformers/stsb:positive
- **Training**: Full fine-tuning with DeepSpeed ZeRO-3
- **Loss**: InfoNCE loss
- **Batch size**: Automatically optimized based on GPU memory
- **Learning rate**: 6e-6
- **Epochs**: 5

### Customizing Training

Edit `run_training.sh` to modify:

- **Number of GPUs**: Change `nproc_per_node=1` to your GPU count
- **Model size**: Change `--model Qwen/Qwen3-Embedding-0.6B` to other sizes (0.5B, 1B, 2B)
- **Dataset**: Change `--dataset sentence-transformers/stsb:positive` to your dataset
- **Hyperparameters**: Modify learning rate, batch size, epochs, etc.

## Available Models

- `Qwen/Qwen3-Embedding-0.5B`
- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-Embedding-1B`
- `Qwen/Qwen3-Embedding-2B`

## Monitoring Training

1. **Container logs:**
   ```bash
   docker-compose logs -f qwen3-swift-training
   ```

2. **Check output directory:**
   ```bash
   ls -la output/
   ```

3. **Monitor GPU usage:**
   ```bash
   docker-compose exec qwen3-swift-training nvidia-smi
   ```

4. **Real-time GPU monitoring:**
   ```bash
   docker-compose exec qwen3-swift-training ./monitor_gpu.sh
   ```

5. **GPU optimization analysis:**
   ```bash
   docker-compose exec qwen3-swift-training ./optimize_gpu_utilization.sh
   ```

## Troubleshooting

### GLIBC Version Issues
This Docker solution specifically addresses GLIBC version issues by using Ubuntu 24.04 with the correct GLIBC version.

### CUDA Issues
- Ensure NVIDIA Docker runtime is properly installed
- Check GPU availability: `nvidia-smi`
- Verify CUDA in container: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- Use `./optimize_gpu_utilization.sh` to get optimal batch sizes
- Reduce batch size if you see OOM errors
- Use smaller model size
- Increase `shm_size` in `docker-compose.yml`

### Multi-GPU Training
To use multiple GPUs:
1. Change `nproc_per_node=8` in `run_training.sh`
2. Ensure all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`

### GPU Not Being Fully Utilized
1. Run `./verify_gpu.sh` to check GPU setup
2. Run `./optimize_gpu_utilization.sh` to get optimal settings
3. Use `./run_training_optimized.sh` for automatic optimization
4. Monitor GPU utilization with `./monitor_gpu.sh`
5. Check training logs for CUDA errors
6. Ensure batch size is optimized for your GPU memory

## Files Structure

```
.
├── Dockerfile                          # Ubuntu 24.04 + CUDA + SWIFT environment
├── docker-compose.yml                  # Container orchestration with GPU optimization
├── run_training.sh                     # Basic training script
├── run_training_optimized.sh           # Training with automatic GPU optimization
├── run_training_with_monitoring.sh     # Enhanced training with GPU monitoring
├── optimize_gpu_utilization.sh         # GPU optimization analysis script
├── verify_gpu.sh                       # GPU verification script
├── monitor_gpu.sh                      # Real-time GPU monitoring
├── setup.sh                           # Automated setup script
├── README.md                          # This file
├── output/                            # Training outputs (created by setup)
└── data/                              # Dataset directory (created by setup)
```

## Cleanup

To stop and remove the container:
```bash
docker-compose down
```

To remove all data:
```bash
docker-compose down -v
rm -rf output data
```

## References

- [SWIFT Documentation](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md)
- [Qwen3-Embedding Repository](https://github.com/QwenLM/Qwen3-Embedding)
- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 