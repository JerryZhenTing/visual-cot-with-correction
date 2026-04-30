#!/bin/bash
# Run ONCE on the Bridges2 login node to create the conda environment.
#
# Usage:
#   bash scripts/setup_env.sh

set -e

module load anaconda3

conda create -n vcot python=3.10 -y
conda activate vcot

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.48.0 accelerate datasets pillow qwen-vl-utils huggingface_hub

echo ""
echo "Environment 'vcot' is ready."
echo "Activate with: module load anaconda3 && conda activate vcot"
