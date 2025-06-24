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

3. **Run training:**
   ```bash
   cd /workspace
   ./run_training.sh
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

4. **Run training commands:**
   ```bash
   cd /workspace
   ./run_training.sh
   ```

## Training Configuration

The training script uses the exact configuration from the [SWIFT documentation](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md):

- **Model**: Qwen/Qwen3-Embedding-0.6B
- **Dataset**: sentence-transformers/stsb:positive
- **Training**: Full fine-tuning with DeepSpeed ZeRO-3
- **Loss**: InfoNCE loss
- **Batch size**: 4 (per device)
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

## Troubleshooting

### GLIBC Version Issues
This Docker solution specifically addresses GLIBC version issues by using Ubuntu 24.04 with the correct GLIBC version.

### CUDA Issues
- Ensure NVIDIA Docker runtime is properly installed
- Check GPU availability: `nvidia-smi`
- Verify CUDA in container: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- Reduce batch size in `run_training.sh`
- Use smaller model size
- Increase `shm_size` in `docker-compose.yml`

### Multi-GPU Training
To use multiple GPUs:
1. Change `nproc_per_node=8` in `run_training.sh`
2. Ensure all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`

## Files Structure

```
.
├── Dockerfile              # Ubuntu 24.04 + CUDA + SWIFT environment
├── docker-compose.yml      # Container orchestration
├── run_training.sh         # Training script with exact SWIFT commands
├── setup.sh               # Automated setup script
├── README.md              # This file
├── output/                # Training outputs (created by setup)
└── data/                  # Dataset directory (created by setup)
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