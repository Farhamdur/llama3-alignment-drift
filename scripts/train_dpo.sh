#!/bin/bash
# ============================================================
# Direct Preference Optimization (DPO) fine-tuning script
# Builds on the Stage-B LoRA adapter (helpful + safety tuned)
#
# Author: Md Farhamdur Reza
# ============================================================

set -euo pipefail

# ============================================================
# Paths
# ============================================================
TRAIN_JSONL="data_out/dpo_pku/dpo_train.jsonl"
EVAL_JSONL="data_out/dpo_pku/dpo_eval.jsonl"

# Stage B LoRA adapter (from train_sft.sh)
RESUME_ADAPTER="outputs/lora-helpful-safety"

# Create unique output directory for this DPO run
SAVE_DIR="outputs/dpo_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAVE_DIR"

LOGFILE="${SAVE_DIR}/train.log"

# ============================================================
# Hyperparameters
# ============================================================
BETA=0.1
LR=5e-7
EPOCHS=1
BATCH=2
GRAD_ACCUM=16
MAX_PROMPT_LEN=2048
MAX_TOTAL_LEN=3072

# ============================================================
# Environment Setup
# ============================================================
echo "[INFO] Activating environment..."

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Change this to your environment name
conda activate CS_DJ

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "============================================================"
echo " DPO TRAINING"
echo "============================================================"
echo " Train file:     ${TRAIN_JSONL}"
echo " Eval file:      ${EVAL_JSONL}"
echo " Resume from:    ${RESUME_ADAPTER}"
echo " Save to:        ${SAVE_DIR}"
echo "------------------------------------------------------------"
echo " Beta:           ${BETA}"
echo " LR:             ${LR}"
echo " Epochs:         ${EPOCHS}"
echo " Batch size:     ${BATCH}"
echo " Grad accum:     ${GRAD_ACCUM}"
echo " Max prompt len: ${MAX_PROMPT_LEN}"
echo " Max total len:  ${MAX_TOTAL_LEN}"
echo "============================================================"

# ============================================================
# Run Training
# ============================================================
echo "[INFO] Starting DPO training..."
echo "[INFO] Logging to $LOGFILE"
echo "------------------------------------------------------------"

python src/run_dpo.py \
  --train_jsonl "$TRAIN_JSONL" \
  --val_jsonl "$EVAL_JSONL" \
  --resume_adapter "$RESUME_ADAPTER" \
  --save_dir "$SAVE_DIR" \
  --beta "$BETA" \
  --lr "$LR" \
  --batch_size "$BATCH" \
  --grad_accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --max_prompt_length "$MAX_PROMPT_LEN" \
  --max_length "$MAX_TOTAL_LEN" \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 50 \
  2>&1 | tee "$LOGFILE"

# ============================================================
# Completion
# ============================================================
echo "============================================================"
echo "✅ DPO training completed"
echo "   Logs: ${LOGFILE}"
echo "   Output adapter: ${SAVE_DIR}"
echo "============================================================"
