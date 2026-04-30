#!/bin/bash
# Download Qwen2.5-VL-7B-Instruct weights to Ocean storage.
# Run on the login node — no GPU needed, just network.

set -e

HF_HOME="/ocean/projects/cis260099p/zliu51/hf_cache"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

echo "HF_HOME = $HF_HOME"
mkdir -p "$HF_HOME"

export HF_HOME="$HF_HOME"

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Downloads will be slow (throttled)."
    echo "Set it with: export HF_TOKEN=hf_xxxx"
fi

echo "Downloading $MODEL (~15 GB) ..."
python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print('Downloading processor...')
AutoProcessor.from_pretrained('$MODEL')
print('Downloading model weights...')
Qwen2_5_VLForConditionalGeneration.from_pretrained('$MODEL', torch_dtype='auto')
print('Download complete.')
"
