docker run --gpus all \
    --shm-size=4g \
    -it -d \
    --name bge-m3-training \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/output:/workspace/output \
    bge-m3-finetune:latest
