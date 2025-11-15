#!/bin/bash
# ============================================================
# Fine-tune Llama-3.1-8B-Instruct for Helpful + Safe Alignment
#   Stage A → Helpful SFT
#   Stage B → Safety (Refusal) SFT
#
# Author: Md Farhamdur Reza
# ============================================================

set -euo pipefail

# ============================================================
# Paths
# ============================================================
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

STAGE_A_TRAIN="data_out/processed/stageA_train.jsonl"
STAGE_A_VAL="data_out/processed/stageA_val.jsonl"

STAGE_B_TRAIN="data_out/processed/stageB_train.jsonl"
STAGE_B_VAL="data_out/processed/stageB_val.jsonl"

OUT_DIR="outputs"
STAGE_A_OUT="${OUT_DIR}/lora-helpful"
STAGE_B_OUT="${OUT_DIR}/lora-helpful-safety"

# ============================================================
# Hardware
# ============================================================
CUDA_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

# ============================================================
# Hyperparameters
# ============================================================
BATCH_SIZE=2
GRAD_ACCUM=16
MAX_LENGTH=2048
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# ============================================================
# Environment
# ============================================================
echo "[INFO] Activating environment..."
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Change this to your env name
conda activate CS_DJ

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

mkdir -p "${OUT_DIR}/logs"

echo "============================================================"
echo " SAFE + HELPFUL ALIGNMENT TRAINING PIPELINE"
echo "============================================================"
echo " Base model:     ${BASE_MODEL}"
echo " GPUs:           ${CUDA_DEVICES}"
echo " Output dir:     ${OUT_DIR}"
echo "------------------------------------------------------------"
echo " Batch size:     ${BATCH_SIZE}"
echo " Grad accum:     ${GRAD_ACCUM}"
echo " Max length:     ${MAX_LENGTH}"
echo " LoRA r:         ${LORA_R}"
echo " LoRA alpha:     ${LORA_ALPHA}"
echo " LoRA dropout:   ${LORA_DROPOUT}"
echo "============================================================"

# ============================================================
# Stage A: Helpful SFT
# ============================================================
echo "[Stage A] Starting Helpful SFT..."
python src/run_sft.py \
    --base_model "${BASE_MODEL}" \
    --train_jsonl "${STAGE_A_TRAIN}" \
    --val_jsonl "${STAGE_A_VAL}" \
    --save_dir "${STAGE_A_OUT}" \
    --epochs 2 \
    --lr 1e-4 \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --bf16 \
    --max_length "${MAX_LENGTH}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    > "${OUT_DIR}/logs/stageA.log" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Stage A failed! Check ${OUT_DIR}/logs/stageA.log"
    exit 1
fi

echo "[Stage A] Completed successfully!"
echo "  → Logs:  ${OUT_DIR}/logs/stageA.log"
echo "  → Model: ${STAGE_A_OUT}"

# ============================================================
# Stage B: Safety (Refusal) SFT
# ============================================================
echo "[Stage B] Starting Safety SFT (Refusal tuning)..."

python src/run_sft.py \
    --base_model "${BASE_MODEL}" \
    --train_jsonl "${STAGE_B_TRAIN}" \
    --val_jsonl "${STAGE_B_VAL}" \
    --save_dir "${STAGE_B_OUT}" \
    --resume_adapter "${STAGE_A_OUT}" \
    --epochs 1 \
    --lr 5e-5 \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --bf16 \
    --max_length "${MAX_LENGTH}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    > "${OUT_DIR}/logs/stageB.log" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Stage B failed! Check ${OUT_DIR}/logs/stageB.log"
    exit 1
fi

echo "[Stage B] Completed successfully!"
echo "  → Logs:  ${OUT_DIR}/logs/stageB.log"
echo "  → Model: ${STAGE_B_OUT}"

# ============================================================
# Final Report
# ============================================================
echo "============================================================"
echo " FINAL TINY-EVAL METRICS (if present)"
echo "------------------------------------------------------------"
echo " Stage A:"
cat "${STAGE_A_OUT}/tiny_eval_metrics.json" 2>/dev/null || echo "  (not found)"
echo ""
echo " Stage B:"
cat "${STAGE_B_OUT}/tiny_eval_metrics.json" 2>/dev/null || echo "  (not found)"
echo "============================================================"
echo "✅ Fine-tuning pipeline completed successfully!"
echo "============================================================"
