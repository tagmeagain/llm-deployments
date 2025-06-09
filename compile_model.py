from optimum.neuron import NeuronModelForCausalLM
from transformers import AutoTokenizer
import torch
import sagemaker
import os
import tarfile

def compile_model(model_path):
    """
    Compile the model for Inferentia2 using Optimum-Neuron
    Args:
        model_path: Local path or Hugging Face model ID
    Returns:
        tuple: (compiled_model, tokenizer)
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Loading and compiling model...")
    model = NeuronModelForCausalLM.from_pretrained(
        model_path,
        export=True,  # This enables model compilation
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer

def save_model(model, tokenizer, output_path):
    """
    Save the compiled model and tokenizer to disk
    Args:
        model: Compiled model
        tokenizer: Model tokenizer
        output_path: Directory to save the model
    Returns:
        str: Path to the saved model directory
    """
    print("Saving compiled model...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")
    return output_path

def create_tar_archive(model_dir, output_tar_path):
    """
    Create a tar.gz archive of the model directory
    Args:
        model_dir: Path to the model directory
        output_tar_path: Path where the tar.gz file should be saved
    Returns:
        str: Path to the created tar.gz file
    """
    print(f"Creating tar.gz archive at {output_tar_path}")
    with tarfile.open(output_tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname=os.path.basename(model_dir))
    return output_tar_path

def upload_to_s3(tar_path, bucket, key_prefix="models"):
    """
    Upload the model tar.gz file to S3
    Args:
        tar_path: Path to the tar.gz file
        bucket: S3 bucket name
        key_prefix: S3 key prefix
    Returns:
        str: S3 path of the uploaded file
    """
    session = sagemaker.Session()
    s3_path = f"s3://{bucket}/{key_prefix}/{os.path.basename(tar_path)}"
    print(f"Uploading to {s3_path}")
    session.upload_data(tar_path, bucket=bucket, key_prefix=key_prefix)
    return s3_path

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "Qwen/Qwen-3-4B"  # Can be local path or Hugging Face model ID
    OUTPUT_DIR = "compiled_model"
    TAR_PATH = f"{OUTPUT_DIR}.tar.gz"
    S3_BUCKET = "your-bucket-name"  # Your S3 bucket name
    
    # Step 1: Compile the model
    model, tokenizer = compile_model(MODEL_PATH)
    
    # Step 2: Save the compiled model
    saved_model_path = save_model(model, tokenizer, OUTPUT_DIR)
    
    # Step 3: Create tar archive
    tar_path = create_tar_archive(saved_model_path, TAR_PATH)
    
    # Step 4: Upload to S3 (optional)
    s3_path = upload_to_s3(tar_path, S3_BUCKET)
    print(f"Compilation and upload complete. Model available at: {s3_path}") 