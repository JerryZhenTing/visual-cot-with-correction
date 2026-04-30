#!/bin/bash
#SBATCH --job-name=corrupted
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -A cis260099p
#SBATCH -o logs/corrupted_%j.out
#SBATCH -e logs/corrupted_%j.err

METHOD=$1
CORRUPTION=$2

if [ -z "$METHOD" ] || [ -z "$CORRUPTION" ]; then
    echo "Usage: sbatch scripts/run_corrupted.sh <method> <corruption>"
    echo "  method     : textual | visual | verification"
    echo "  corruption : blur-1 | blur-3 | blur-5 | noise-001 | noise-005 | noise-010 | rot-15 | rot-45 | rot-90"
    exit 1
fi

PROJECT_DIR="/jet/home/zliu51/project"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Method: $METHOD   Corruption: $CORRUPTION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results/corrupted/$METHOD"

module load anaconda3

export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib:$LD_LIBRARY_PATH
export HF_HOME="/ocean/projects/cis260099p/zliu51/hf_cache"
export TRANSFORMERS_OFFLINE=1

python "$PROJECT_DIR/src/run_corrupted.py" --method "$METHOD" --corruption "$CORRUPTION"

echo "Job finished at $(date)"
