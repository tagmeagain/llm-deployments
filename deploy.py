import sagemaker
from sagemaker.model import Model
from sagemaker.pytorch import PyTorchModel
import boto3
import os

def deploy_model(model_path, role_arn, region='us-east-1'):
    """
    Deploy the model to SageMaker endpoint
    """
    # Initialize SageMaker session
    session = sagemaker.Session()
    
    # Create model
    model = PyTorchModel(
        model_data=model_path,
        role=role_arn,
        framework_version='2.0.0',
        py_version='py3',
        entry_point='model.py',
        source_dir='.',
        env={
            'SAGEMAKER_MODEL_SERVER_TIMEOUT': '600',
            'SAGEMAKER_MODEL_SERVER_WORKERS': '1',
        }
    )
    
    # Deploy model to endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.inf2.xlarge',  # Inferentia2 instance
        endpoint_name='qwen-3-4b-endpoint'
    )
    
    return predictor

if __name__ == '__main__':
    # Replace these with your actual values
    MODEL_PATH = 's3://your-bucket/model.tar.gz'  # Path to your model artifacts
    ROLE_ARN = 'arn:aws:iam::your-account:role/your-role'  # Your SageMaker execution role
    
    predictor = deploy_model(MODEL_PATH, ROLE_ARN)
    print(f"Model deployed successfully to endpoint: {predictor.endpoint_name}") 