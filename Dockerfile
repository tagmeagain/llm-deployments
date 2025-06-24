# Use NVIDIA CUDA base image with Ubuntu 24.04
FROM nvidia/cuda:12.4-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install SWIFT and dependencies as per the documentation
RUN pip install ms-swift -U
RUN pip install git+https://github.com/modelscope/ms-swift.git
RUN pip install transformers -U

# Install optional packages for multi-GPU training and memory optimization
RUN pip install deepspeed
RUN pip install liger-kernel
RUN pip install flash-attn --no-build-isolation

# Set working directory
WORKDIR /workspace

# Set environment variable for CUDA
ENV CUDA_VISIBLE_DEVICES=0

# Create a script to run training commands
RUN echo '#!/bin/bash\n\
echo "Starting SWIFT training environment..."\n\
echo "CUDA devices available: $CUDA_VISIBLE_DEVICES"\n\
echo "Python version: $(python --version)"\n\
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")\n\
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")\n\
echo "CUDA version: $(python -c "import torch; print(torch.version.cuda)")\n\
echo "SWIFT version: $(python -c "import swift; print(swift.__version__)")\n\
echo "Ready to run SWIFT training commands!"\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"] 