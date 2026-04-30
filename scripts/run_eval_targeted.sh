#!/bin/bash
#SBATCH --job-name=eval_tgt
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -A cis260099p
#SBATCH -o logs/eval_tgt_%j.out
#SBATCH -e logs/eval_tgt_%j.err

# Usage:
#   sbatch scripts/run_eval_targeted.sh <method> <perturbation>
#
# Examples:
#   sbatch scripts/run_eval_targeted.sh visual occlusion-obj1-low
#   sbatch scripts/run_eval_targeted.sh multistage mask-union-high
#
# Run all 60 cells (4 methods × 15 perturbation specs):
#   SPECS="occlusion-obj1-low occlusion-obj1-medium occlusion-obj1-high \
#          occlusion-obj2-low occlusion-obj2-medium occlusion-obj2-high \
#          mask-union-low mask-union-medium mask-union-high \
#          distractor-nearobj1-low distractor-nearobj1-medium distractor-nearobj1-high \
#          distractor-nearobj2-low distractor-nearobj2-medium distractor-nearobj2-high"
#   for method in textual visual verification multistage; do
#     for spec in $SPECS; do
#       sbatch scripts/run_eval_targeted.sh $method $spec
#     done
#   done

METHOD=$1
PERTURBATION=$2

if [ -z "$METHOD" ] || [ -z "$PERTURBATION" ]; then
    echo "Usage: sbatch scripts/run_eval_targeted.sh <method> <perturbation>"
    echo "  method      : textual | visual | verification | multistage"
    echo "  perturbation: occlusion-obj1-{low,medium,high} | occlusion-obj2-{low,medium,high}"
    echo "                mask-union-{low,medium,high}"
    echo "                distractor-nearobj1-{low,medium,high} | distractor-nearobj2-{low,medium,high}"
    exit 1
fi

PROJECT_DIR="/jet/home/zliu51/project"
SUBSET="data/subsets/vsr_n200_seq.json"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Method: $METHOD   Perturbation: $PERTURBATION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results/targeted_raw"

module load anaconda3

export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib:$LD_LIBRARY_PATH
export HF_HOME="/ocean/projects/cis260099p/zliu51/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python "$PROJECT_DIR/src/eval_targeted.py" \
    --method "$METHOD" \
    --perturbation "$PERTURBATION" \
    --targeted-dir "$PROJECT_DIR/data/targeted" \
    --subset "$SUBSET" \
    --out-dir results/targeted_raw

echo "Job finished at $(date)"
