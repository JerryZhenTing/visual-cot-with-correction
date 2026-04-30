#!/bin/bash
#SBATCH --job-name=guidance
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -A cis260099p
#SBATCH -o logs/guidance_%j.out
#SBATCH -e logs/guidance_%j.err

# Usage:
#   sbatch scripts/run_guidance.sh [sft|rl_grounding|rl_combined|eval]
#
# Stages (run in order):
#   sft          — build dataset cache + supervised training (~30 min)
#   rl_grounding — RL fine-tuning, grounding reward only  (~30 min)
#   rl_combined  — RL fine-tuning, answer-aware reward    (~3-4h, VLM calls)
#   eval         — evaluate on test split, no VLM         (~10 min)
#   all          — run sft + rl_grounding + eval in sequence (fits in 5h)
#
# Default stage: all

STAGE=${1:-all}

PROJECT_DIR="/jet/home/zliu51/project"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Stage: $STAGE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_DIR/logs" \
         "$(dirname $PROJECT_DIR)/checkpoints/guidance_sft" \
         "$(dirname $PROJECT_DIR)/checkpoints/guidance_rl" \
         "$PROJECT_DIR/results/guidance_raw" \
         "$PROJECT_DIR/results/guidance_aggregated"

module load anaconda3

export LD_LIBRARY_PATH=/opt/packages/anaconda3-2024.10-1/lib:$LD_LIBRARY_PATH
export HF_HOME="/ocean/projects/cis260099p/zliu51/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ---------------------------------------------------------------------------
# Stage: build dataset cache from full VSR validation set (~1222 examples)
# ---------------------------------------------------------------------------
run_sft() {
    echo "=== [1/2] Building guidance dataset cache ==="
    python "$PROJECT_DIR/src/guidance_dataset.py" \
        --cache "$PROJECT_DIR/data/vsr_guidance_full.json" \
        --n 1222

    echo "=== [2/2] SFT training (30 epochs, batch=64) ==="
    python "$PROJECT_DIR/src/train_guidance_sft.py" \
        --data "$PROJECT_DIR/data/vsr_guidance_full.json" \
        --epochs 30 \
        --batch-size 64 \
        --lr 1e-4 \
        --seed 42 \
        --output-dir "$(dirname $PROJECT_DIR)/checkpoints/guidance_sft"
}

# ---------------------------------------------------------------------------
# Stage: grounding-only RL (fast, no VLM calls)
# ---------------------------------------------------------------------------
run_rl_grounding() {
    echo "=== RL fine-tuning (grounding reward, 10 epochs) ==="
    python "$PROJECT_DIR/src/train_guidance_rl.py" \
        --data "$PROJECT_DIR/data/vsr_guidance_full.json" \
        --sft-checkpoint "$(dirname $PROJECT_DIR)/checkpoints/guidance_sft/best.pt" \
        --reward-type grounding \
        --epochs 10 \
        --batch-size 32 \
        --samples 4 \
        --lr 1e-5 \
        --seed 42 \
        --output-dir "$(dirname $PROJECT_DIR)/checkpoints/guidance_rl"
}

# ---------------------------------------------------------------------------
# Stage: answer-aware RL (slow, ~3-4h, uses 200 training examples to limit VLM calls)
# ---------------------------------------------------------------------------
run_rl_combined() {
    echo "=== RL fine-tuning (combined reward, 5 epochs, max 200 train examples) ==="
    python "$PROJECT_DIR/src/train_guidance_rl.py" \
        --data "$PROJECT_DIR/data/vsr_guidance_full.json" \
        --sft-checkpoint "$(dirname $PROJECT_DIR)/checkpoints/guidance_sft/best.pt" \
        --reward-type combined \
        --epochs 5 \
        --batch-size 8 \
        --samples 2 \
        --max-train 200 \
        --lr 1e-5 \
        --seed 42 \
        --output-dir "$(dirname $PROJECT_DIR)/checkpoints/guidance_rl_combined"
}

# ---------------------------------------------------------------------------
# Stage: evaluate (grounding metrics only, no VLM)
# ---------------------------------------------------------------------------
run_eval() {
    echo "=== Evaluating guidance policy (grounding metrics, no VLM) ==="
    python "$PROJECT_DIR/src/eval_guidance_policy.py" \
        --checkpoint "$(dirname $PROJECT_DIR)/checkpoints/guidance_rl/best.pt" \
        --data "$PROJECT_DIR/data/vsr_guidance_full.json" \
        --model none \
        --baselines random full_image \
        --split test \
        --seed 42 \
        --save-viz \
        --viz-dir "$PROJECT_DIR/results/guidance_viz" \
        --output-dir "$PROJECT_DIR/results/guidance_raw"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$STAGE" in
    sft)          run_sft ;;
    rl_grounding) run_rl_grounding ;;
    rl_combined)  run_rl_combined ;;
    eval)         run_eval ;;
    all)
        run_sft
        run_rl_grounding
        run_eval
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid stages: sft | rl_grounding | rl_combined | eval | all"
        exit 1
        ;;
esac

echo "Job finished at $(date)"
