import os
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import boto3

def deploy_to_sagemaker(model_dir):
    """
    Deploy a local fine-tuned model to SageMaker G5 instance
    Args:
        model_dir: Path to your local fine-tuned model directory
    """
    # Initialize SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Create HuggingFace model
    huggingface_model = HuggingFaceModel(
        transformers_version='4.36.0',
        pytorch_version='2.1.0',
        py_version='py310',
        env={
            'HF_MODEL_ID': model_dir,
            'HF_TASK': 'text-generation',
            'HF_MODEL_QUANTIZE': '4bit'
        },
        role=role,
        image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.36.0-gpu-py310-cu118-ubuntu20.04'
    )
    
    # Deploy model to G5 instance
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type='ml.g5.xlarge',
        endpoint_name='fine-tuned-8b-endpoint'
    )
    
    return predictor

if __name__ == "__main__":
    # Specify the path to your local fine-tuned model
    model_dir = input("Enter the path to your local fine-tuned model directory: ")
    
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist!")
    
    # Deploy to SageMaker
    print("Deploying to SageMaker...")
    predictor = deploy_to_sagemaker(model_dir)
    
    print("Model deployed successfully!")
    print(f"Endpoint name: fine-tuned-8b-endpoint") 