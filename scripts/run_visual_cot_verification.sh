#!/bin/bash
#SBATCH --job-name=vcot_verify2
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -A cis260099p
#SBATCH -o logs/visual_cot_verification_%j.out
#SBATCH -e logs/visual_cot_verification_%j.err

PROJECT_DIR="/jet/home/zliu51/project"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results"

module load anaconda3

export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib:$LD_LIBRARY_PATH
export HF_HOME="/ocean/projects/cis260099p/zliu51/hf_cache"
export TRANSFORMERS_OFFLINE=1

python "$PROJECT_DIR/src/run_visual_cot_verification.py"

echo "Job finished at $(date)"
