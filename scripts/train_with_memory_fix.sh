#!/bin/bash
# Training script with memory optimization

# Set PyTorch memory allocation config to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
cd /home/tnzl/GPT-scratch
source .venv/bin/activate
python scripts/train.py --config "$@"
