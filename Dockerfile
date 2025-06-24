FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone FlagEmbedding repository
RUN git clone https://github.com/FlagOpen/FlagEmbedding.git
WORKDIR /workspace/FlagEmbedding

# Install FlagEmbedding with fine-tuning dependencies
RUN pip install -e .[finetune]

# Install additional evaluation and acceleration dependencies
RUN pip install pytrec_eval faiss-gpu accelerate

# Set environment variables
ENV PYTHONPATH=/workspace/FlagEmbedding:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

CMD ["/bin/bash"]
