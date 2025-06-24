#!/bin/bash

echo "🔍 GPU Verification Script for Qwen3-Embedding Training"
echo "======================================================"

# Check if nvidia-smi is available
echo "1. Checking nvidia-smi availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi is available"
    echo "GPU Information:"
    nvidia-smi
else
    echo "❌ nvidia-smi not found. GPU may not be accessible."
    exit 1
fi

echo ""
echo "2. Checking CUDA environment variables..."
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'not set'}"
echo "NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-'not set'}"

echo ""
echo "3. Checking PyTorch CUDA support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'Current device: {torch.cuda.current_device()}')
else:
    print('❌ CUDA is not available in PyTorch')
"

echo ""
echo "4. Testing GPU memory allocation..."
python3 -c "
import torch
if torch.cuda.is_available():
    try:
        # Allocate a small tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        print(f'✅ Successfully allocated tensor on GPU')
        print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
        print(f'GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB')
        del x
        torch.cuda.empty_cache()
        print('✅ GPU memory test passed')
    except Exception as e:
        print(f'❌ GPU memory test failed: {e}')
else:
    print('❌ Cannot test GPU memory - CUDA not available')
"

echo ""
echo "5. Checking SWIFT installation..."
python3 -c "
try:
    import swift
    print(f'✅ SWIFT version: {swift.__version__}')
except ImportError:
    print('❌ SWIFT not installed')
"

echo ""
echo "6. Testing SWIFT GPU compatibility..."
python3 -c "
import torch
import swift
if torch.cuda.is_available():
    print('✅ SWIFT should work with GPU')
    print('Ready to run training with GPU acceleration!')
else:
    print('❌ SWIFT will run on CPU only')
"

echo ""
echo "🎯 GPU Verification Complete!"
echo "If all checks pass, your GPU is ready for training." 