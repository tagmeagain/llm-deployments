# LLM Model Deployment

This repository contains code for deploying large language models (LLMs) on AWS SageMaker using G5 instances.

## Models Supported
- Qwen-3 4B (4-bit quantized)
- Custom fine-tuned 8B models (4-bit quantized)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:
```bash
aws configure
```

## Deployment

### For Qwen-3 4B Model
```bash
python deploy_qwen.py
```

### For Custom Fine-tuned 8B Model
```bash
python deploy_qwen.py
# Enter the path to your local model when prompted
```

## Requirements
- Python 3.10+
- AWS Account with SageMaker access
- Sufficient AWS credits for G5 instance usage

## Files
- `deploy_qwen.py`: Main deployment script
- `requirements.txt`: Python dependencies
- `model.py`: Model inference code 