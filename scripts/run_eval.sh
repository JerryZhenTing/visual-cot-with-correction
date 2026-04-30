#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -A cis260099p
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err

# Usage:
#   sbatch scripts/run_eval.sh <method> <condition>
#
# Examples:
#   sbatch scripts/run_eval.sh textual clean
#   sbatch scripts/run_eval.sh visual blur-1
#   sbatch scripts/run_eval.sh verification rot-45
#   sbatch scripts/run_eval.sh multistage noise-005
#
# Run all 40 cells at once:
#   for method in textual visual verification multistage; do
#     for cond in clean blur-1 blur-3 blur-5 noise-001 noise-005 noise-010 rot-15 rot-45 rot-90; do
#       sbatch scripts/run_eval.sh $method $cond
#     done
#   done

METHOD=$1
CONDITION=$2

if [ -z "$METHOD" ] || [ -z "$CONDITION" ]; then
    echo "Usage: sbatch scripts/run_eval.sh <method> <condition>"
    echo "  method    : textual | visual | verification | multistage"
    echo "  condition : clean | blur-1 | blur-3 | blur-5 | noise-001 | noise-005 | noise-010 | rot-15 | rot-45 | rot-90"
    exit 1
fi

PROJECT_DIR="/jet/home/zliu51/project"
SUBSET="data/subsets/vsr_n200_seq.json"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Method: $METHOD   Condition: $CONDITION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results/raw"

module load anaconda3

export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib:$LD_LIBRARY_PATH
export HF_HOME="/ocean/projects/cis260099p/zliu51/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python "$PROJECT_DIR/src/eval_runner.py" \
    --method "$METHOD" \
    --condition "$CONDITION" \
    --subset "$SUBSET" \
    --out-dir results/raw

echo "Job finished at $(date)"
