import os
import json
import torch
from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM

def model_fn(model_dir):
    """
    Load the compiled model and tokenizer from the model directory
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load compiled model
    model = NeuronModelForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_tokenizer):
    """
    Apply model to the input data
    """
    model, tokenizer = model_tokenizer
    
    # Get input text
    input_text = input_data.get('text', '')
    max_length = input_data.get('max_length', 100)
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    if response_content_type == 'application/json':
        return json.dumps({'generated_text': prediction})
    raise ValueError(f"Unsupported content type: {response_content_type}") 