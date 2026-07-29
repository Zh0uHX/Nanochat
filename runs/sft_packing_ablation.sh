#!/usr/bin/env bash
set -euo pipefail

# Fixed-budget SFT packing ablation. Run every strategy from the same base
# checkpoint and compare packing efficiency, tok/s, MFU, validation BPB, and
# downstream chat evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_MODEL_TAG="${BASE_MODEL_TAG:?set BASE_MODEL_TAG to an existing base checkpoint}"
BASE_MODEL_STEP="${BASE_MODEL_STEP:-}"
ABLATION_STEPS="${ABLATION_STEPS:-200}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
OPTIMIZER_KERNEL="${OPTIMIZER_KERNEL:-compiled}"
VALIDATION_PACKING_STRATEGY="${VALIDATION_PACKING_STRATEGY:-best_fit}"
RESULT_DIR="${RESULT_DIR:-benchmark_results/sft_packing_ablation}"
DRY_RUN="${DRY_RUN:-1}"

mkdir -p "${RESULT_DIR}"

DRY_RUN_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_RUN_ARGS+=(--dry-run)
fi

MODEL_STEP_ARGS=()
if [[ -n "${BASE_MODEL_STEP}" ]]; then
  MODEL_STEP_ARGS+=(--model-step="${BASE_MODEL_STEP}")
fi

for STRATEGY in sequential first_fit length_bucket best_fit; do
  OUTPUT_TAG="${BASE_MODEL_TAG}-packing-${STRATEGY}"
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.chat_sft -- \
    --model-tag="${BASE_MODEL_TAG}" \
    "${MODEL_STEP_ARGS[@]}" \
    --output-model-tag="${OUTPUT_TAG}" \
    --num-iterations="${ABLATION_STEPS}" \
    --device-batch-size="${DEVICE_BATCH_SIZE}" \
    --total-batch-size="${TOTAL_BATCH_SIZE}" \
    --packing-strategy="${STRATEGY}" \
    --validation-packing-strategy="${VALIDATION_PACKING_STRATEGY}" \
    --optimizer-kernel="${OPTIMIZER_KERNEL}" \
    --eval-every=-1 \
    --eval-tokens="${EVAL_TOKENS}" \
    --save-every=-1 \
    --result-output="${RESULT_DIR}/${STRATEGY}.json" \
    --run="dummy" \
    "${DRY_RUN_ARGS[@]}"
done
